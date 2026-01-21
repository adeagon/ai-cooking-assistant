"""LLM Integration tests for Phase 5 feedback commands.

These tests require Ollama to be running with llama3.3:70b model.
Run with: pytest tests/test_llm_chat_phase5.py -v -s -m llm
"""

import sqlite3
from io import StringIO

import pytest

from langchain_ollama import ChatOllama
from rich.console import Console

from src.app.cli import display_full_recipe, resolve_recipe_reference
from src.app.settings import Settings
from src.chains.chat_chain import build_chat_chain
from src.chains.extractors import ConstraintExtractor
from src.chains.retrieval import RetrievalRunnable
from src.domain.models import PreferenceProfile, RecipeFeedback, SessionState
from src.ingest.build_db import get_recipe_by_id
from src.memory.feedback_store import FeedbackStore
from src.memory.history_store import HistoryStore
from src.retrieval.recipe_cards import RecipeCardBuilder
from src.retrieval.rerank import RecipeReranker
from src.retrieval.retriever import RecipeRetriever

# Test user ID for LLM tests
LLM_TEST_USER_ID = "llm-test-user-00000000-0000-0000-0000-000000000001"


@pytest.fixture(scope="module")
def llm_setup():
    """Set up LLM and retrieval chain for tests.

    Requires:
    - Ollama running on localhost:11434
    - llama3.3:70b model pulled
    - Vector store populated
    - SQLite database with recipes
    """
    settings = Settings()

    # Initialize LLM
    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.llm_temperature,
        reasoning=False,  # Disable Qwen3 thinking mode
    )

    # Initialize retrieval components
    retriever = RecipeRetriever(settings.chroma_persist_dir, settings.embedding_model)
    reranker = RecipeReranker(settings.reranker_model)
    card_builder = RecipeCardBuilder(settings.sqlite_db_path)
    retrieval_chain = RetrievalRunnable(retriever, reranker, card_builder, settings)

    return {
        "llm": llm,
        "retrieval_chain": retrieval_chain,
        "settings": settings,
        "db_path": settings.sqlite_db_path,
    }


@pytest.fixture
def feedback_db(tmp_path):
    """Create a temporary database for feedback/history with test recipes."""
    db_path = tmp_path / "test_feedback.db"

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Create users table (required for multi-user stores)
    cursor.execute("""
        CREATE TABLE users (
            id TEXT PRIMARY KEY,
            username TEXT UNIQUE NOT NULL COLLATE NOCASE,
            password_hash TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE
        )
    """)

    # Insert test user
    cursor.execute(
        "INSERT INTO users (id, username, is_active) VALUES (?, ?, ?)",
        (LLM_TEST_USER_ID, "llm_test_user", True)
    )

    # Create minimal recipes table for foreign key constraint
    cursor.execute("""
        CREATE TABLE recipes (
            recipe_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            tags TEXT DEFAULT '[]'
        )
    """)

    conn.commit()
    conn.close()

    return db_path


@pytest.mark.llm
class TestPhase5LLMIntegration:
    """LLM integration tests for Phase 5 features.

    These tests require Ollama to be running and will make actual LLM calls.
    They demonstrate the full conversation flow with Phase 5 commands.
    """

    def test_feedback_workflow_with_llm(self, llm_setup, feedback_db):
        """Test complete feedback workflow with actual LLM.

        Demonstrates:
        1. User asks for chicken recipes -> LLM recommends recipes
        2. User likes recipe #1
        3. User asks for chicken recipes again -> liked recipe is excluded
        """
        console = Console()

        # Set up stores with user_id
        feedback_store = FeedbackStore(feedback_db, LLM_TEST_USER_ID)
        history_store = HistoryStore(feedback_db, LLM_TEST_USER_ID)

        profile = PreferenceProfile()
        session = SessionState()
        rolling_summary = ""

        # ==== TURN 1: Ask for chicken recipes ====
        console.print("\n" + "="*70)
        console.print("[bold cyan]TURN 1: Ask for chicken recipes[/bold cyan]")
        console.print("="*70 + "\n")

        exclude_ids = set()  # No exclusions yet

        chain = build_chat_chain(
            llm=llm_setup["llm"],
            retrieval_chain=llm_setup["retrieval_chain"],
            profile=profile,
            session=session,
            rolling_summary=rolling_summary,
            exclude_recipe_ids=exclude_ids,
        )

        user_input = "I want chicken recipes, something quick under 30 minutes"
        console.print(f"[green bold]User:[/green bold] {user_input}")

        response = chain.invoke({"user_input": user_input})
        console.print(f"\n[blue bold]Assistant:[/blue bold] {response}")

        # Capture recipe cards
        extractor = ConstraintExtractor()
        constraints = extractor.extract_constraints(user_input)
        retrieval_result = llm_setup["retrieval_chain"].invoke({
            "user_input": user_input,
            "constraints": constraints,
            "session": session,
            "exclude_recipe_ids": exclude_ids,
        })
        last_cards = retrieval_result.get("cards", [])

        console.print(f"\n[yellow bold]Captured {len(last_cards)} recipe cards:[/yellow bold]")
        for i, card in enumerate(last_cards, 1):
            console.print(f"  {i}. {card.title}")
            console.print(f"     ID: {card.recipe_id} | Rating: {card.rating_avg:.1f}/5 | Time: {card.time_total}min")

        # Verify we got recipes
        assert len(last_cards) > 0, "Should receive recipe recommendations"

        # ==== COMMAND: Like recipe #1 ====
        console.print("\n" + "="*70)
        console.print("[bold cyan]COMMAND: /like 1[/bold cyan]")
        console.print("="*70 + "\n")

        result = resolve_recipe_reference("1", last_cards)
        assert result is not None, "Should resolve recipe #1"
        recipe_id, title = result

        # Add recipe to database so foreign key constraint works
        conn = sqlite3.connect(feedback_db)
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO recipes (recipe_id, title) VALUES (?, ?)", (recipe_id, title))
        conn.commit()
        conn.close()

        feedback_store.add_feedback(RecipeFeedback(
            recipe_id=recipe_id,
            feedback_type="like"
        ))
        console.print(f"[green]OK Liked: {title}[/green]")
        console.print(f"[dim]   (Recipe ID: {recipe_id})[/dim]")

        # ==== TURN 2: Ask for chicken recipes again ====
        console.print("\n" + "="*70)
        console.print("[bold cyan]TURN 2: Ask for chicken recipes again (test exclusion)[/bold cyan]")
        console.print("="*70 + "\n")

        # Recompute exclusions
        exclude_ids = (
            feedback_store.get_liked_recipe_ids(limit=20) |
            feedback_store.get_disliked_recipe_ids() |
            history_store.get_recently_cooked_ids(days=7)
        )
        console.print(f"[yellow]Exclusion set: {exclude_ids}[/yellow]")
        console.print(f"[dim]Excluded recipe: {recipe_id} - {title}[/dim]\n")

        chain = build_chat_chain(
            llm=llm_setup["llm"],
            retrieval_chain=llm_setup["retrieval_chain"],
            profile=profile,
            session=session,
            rolling_summary=rolling_summary,
            exclude_recipe_ids=exclude_ids,
        )

        user_input = "show me more chicken recipes under 30 minutes"
        console.print(f"[green bold]User:[/green bold] {user_input}")

        response = chain.invoke({"user_input": user_input})
        console.print(f"\n[blue bold]Assistant:[/blue bold] {response}")

        # Capture new cards
        constraints = extractor.extract_constraints(user_input)
        retrieval_result = llm_setup["retrieval_chain"].invoke({
            "user_input": user_input,
            "constraints": constraints,
            "session": session,
            "exclude_recipe_ids": exclude_ids,
        })
        new_cards = retrieval_result.get("cards", [])

        console.print(f"\n[yellow bold]New recipe cards (should NOT include {recipe_id}):[/yellow bold]")
        for i, card in enumerate(new_cards, 1):
            console.print(f"  {i}. {card.title}")
            console.print(f"     ID: {card.recipe_id} | Rating: {card.rating_avg:.1f}/5 | Time: {card.time_total}min")
            # Verify liked recipe is not in new results
            assert card.recipe_id != recipe_id, f"Liked recipe {recipe_id} should be excluded!"

        console.print("\n[green bold]OK SUCCESS: Liked recipe was excluded from new results![/green bold]\n")

    def test_show_and_cooked_commands(self, llm_setup, feedback_db):
        """Test /show and /cooked commands with actual recipes.

        Demonstrates:
        1. Get pasta recommendations
        2. /show command displays full recipe
        3. /cooked command marks recipe as cooked
        4. Cooked recipe is excluded from next search
        """
        console = Console()

        feedback_store = FeedbackStore(feedback_db, LLM_TEST_USER_ID)
        history_store = HistoryStore(feedback_db, LLM_TEST_USER_ID)

        profile = PreferenceProfile()
        session = SessionState()
        rolling_summary = ""

        # ==== TURN 1: Ask for pasta recipes ====
        console.print("\n" + "="*70)
        console.print("[bold cyan]TURN 1: Ask for pasta recipes[/bold cyan]")
        console.print("="*70 + "\n")

        exclude_ids = set()

        chain = build_chat_chain(
            llm=llm_setup["llm"],
            retrieval_chain=llm_setup["retrieval_chain"],
            profile=profile,
            session=session,
            rolling_summary=rolling_summary,
            exclude_recipe_ids=exclude_ids,
        )

        user_input = "quick pasta recipes"
        console.print(f"[green bold]User:[/green bold] {user_input}")

        response = chain.invoke({"user_input": user_input})
        console.print(f"\n[blue bold]Assistant:[/blue bold] {response}")

        # Capture cards
        extractor = ConstraintExtractor()
        constraints = extractor.extract_constraints(user_input)
        retrieval_result = llm_setup["retrieval_chain"].invoke({
            "user_input": user_input,
            "constraints": constraints,
            "session": session,
            "exclude_recipe_ids": exclude_ids,
        })
        last_cards = retrieval_result.get("cards", [])

        console.print(f"\n[yellow bold]Captured {len(last_cards)} recipe cards[/yellow bold]")

        assert len(last_cards) > 0, "Should receive recipe recommendations"

        # ==== COMMAND: Show full recipe ====
        console.print("\n" + "="*70)
        console.print("[bold cyan]COMMAND: /show 1[/bold cyan]")
        console.print("="*70 + "\n")

        result = resolve_recipe_reference("1", last_cards)
        assert result is not None
        recipe_id, title = result

        # Get full recipe from database
        full_recipe = get_recipe_by_id(llm_setup["db_path"], recipe_id)
        assert full_recipe is not None, "Recipe should exist in database"

        # Display full recipe
        output = StringIO()
        temp_console = Console(file=output, force_terminal=False)
        display_full_recipe(full_recipe, temp_console)
        recipe_display = output.getvalue()

        console.print("[green]OK Full recipe displayed:[/green]")
        console.print(f"[dim]  Title: {full_recipe.title}[/dim]")
        console.print(f"[dim]  Ingredients: {len(full_recipe.ingredients)} items[/dim]")
        console.print(f"[dim]  Instructions: {len(full_recipe.instructions)} steps[/dim]")

        # ==== COMMAND: Mark as cooked ====
        console.print("\n" + "="*70)
        console.print("[bold cyan]COMMAND: /cooked 1[/bold cyan]")
        console.print("="*70 + "\n")

        # Add recipe to DB for foreign key
        conn = sqlite3.connect(feedback_db)
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO recipes (recipe_id, title) VALUES (?, ?)", (recipe_id, title))
        conn.commit()
        conn.close()

        history_store.add_cooked(recipe_id)
        console.print(f"[green]OK Marked as cooked: {title}[/green]")

        # ==== TURN 2: Verify exclusion ====
        console.print("\n" + "="*70)
        console.print("[bold cyan]TURN 2: Verify cooked recipe is excluded[/bold cyan]")
        console.print("="*70 + "\n")

        exclude_ids = (
            feedback_store.get_liked_recipe_ids(limit=20) |
            feedback_store.get_disliked_recipe_ids() |
            history_store.get_recently_cooked_ids(days=7)
        )

        console.print(f"[yellow]Recently cooked recipes: {history_store.get_recently_cooked_ids(days=7)}[/yellow]")
        assert recipe_id in exclude_ids, "Cooked recipe should be in exclusion set"

        console.print(f"[green]OK SUCCESS: Cooked recipe {recipe_id} is in exclusion set[/green]\n")

    def test_rating_and_history(self, llm_setup, feedback_db):
        """Test /rate command and /history display.

        Demonstrates:
        1. Get recipe recommendations
        2. /rate command stores rating
        3. /cooked marks recipe as cooked
        4. History shows cooked recipes
        """
        console = Console()

        feedback_store = FeedbackStore(feedback_db, LLM_TEST_USER_ID)
        history_store = HistoryStore(feedback_db, LLM_TEST_USER_ID)

        profile = PreferenceProfile()
        session = SessionState()

        # Get some recipes
        console.print("\n" + "="*70)
        console.print("[bold cyan]Getting recipe recommendations[/bold cyan]")
        console.print("="*70 + "\n")

        chain = build_chat_chain(
            llm=llm_setup["llm"],
            retrieval_chain=llm_setup["retrieval_chain"],
            profile=profile,
            session=session,
            rolling_summary="",
            exclude_recipe_ids=set(),
        )

        user_input = "healthy dinner recipes"
        console.print(f"[green bold]User:[/green bold] {user_input}")

        response = chain.invoke({"user_input": user_input})
        console.print(f"\n[blue bold]Assistant:[/blue bold] {response}")

        # Capture cards
        extractor = ConstraintExtractor()
        constraints = extractor.extract_constraints(user_input)
        retrieval_result = llm_setup["retrieval_chain"].invoke({
            "user_input": user_input,
            "constraints": constraints,
            "session": session,
            "exclude_recipe_ids": set(),
        })
        last_cards = retrieval_result.get("cards", [])

        assert len(last_cards) > 0

        # ==== COMMAND: Rate recipe 5 stars ====
        console.print("\n" + "="*70)
        console.print("[bold cyan]COMMAND: /rate 5 1[/bold cyan]")
        console.print("="*70 + "\n")

        result = resolve_recipe_reference("1", last_cards)
        assert result is not None
        recipe_id, title = result

        # Add to DB
        conn = sqlite3.connect(feedback_db)
        cursor = conn.cursor()
        cursor.execute("INSERT OR IGNORE INTO recipes (recipe_id, title) VALUES (?, ?)", (recipe_id, title))
        conn.commit()
        conn.close()

        feedback_store.add_feedback(RecipeFeedback(
            recipe_id=recipe_id,
            feedback_type="rate",
            rating=5
        ))
        console.print(f"[green]OK Rated {title}: 5/5[/green]")

        # ==== COMMAND: Mark as cooked ====
        console.print("\n" + "="*70)
        console.print("[bold cyan]COMMAND: /cooked 1[/bold cyan]")
        console.print("="*70 + "\n")

        history_store.add_cooked(recipe_id, notes="Delicious!")
        console.print(f"[green]OK Marked as cooked: {title}[/green]")

        # ==== COMMAND: View history ====
        console.print("\n" + "="*70)
        console.print("[bold cyan]COMMAND: /history[/bold cyan]")
        console.print("="*70 + "\n")

        history = history_store.get_cooking_history(limit=10)

        console.print("[yellow bold]Cooking History:[/yellow bold]")
        for entry in history:
            console.print(f"  - Recipe ID: {entry.recipe_id}")
            console.print(f"    Cooked at: {entry.cooked_at}")
            if entry.notes:
                console.print(f"    Notes: {entry.notes}")

        assert len(history) == 1, "Should have 1 cooked recipe"
        assert history[0].recipe_id == recipe_id
        assert history[0].notes == "Delicious!"

        console.print(f"\n[green]OK SUCCESS: History shows cooked recipe with notes[/green]\n")

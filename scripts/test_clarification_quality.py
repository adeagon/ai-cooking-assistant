"""Test clarification quality with reasoning=True.

This script tests vague/ambiguous queries that should trigger the clarification
branch, using the thoughtful LLM with reasoning enabled.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_ollama import ChatOllama
from src.app.settings import settings
from src.retrieval.retriever import RecipeRetriever
from src.retrieval.rerank import RecipeReranker
from src.retrieval.recipe_cards import RecipeCardBuilder
from src.chains.retrieval import RetrievalRunnable
from src.chains.chat_chain import build_chat_chain
from src.memory import ProfileStore, SessionStore, FeedbackStore, HistoryStore

# Vague queries that should trigger clarification
CLARIFICATION_QUERIES = [
    # Very vague
    "food",
    "dinner",
    "what should I eat",
    "I'm hungry",
    "cook something",

    # Slightly more context but still vague
    "something good",
    "make me dinner",
    "I don't know what to cook",
    "surprise me",
    "anything really",

    # Dish name alone (should clarify for preferences)
    "lasagna",
    "curry",
    "stir fry",
    "tacos",
    "soup",

    # Mood-based but vague
    "I'm feeling lazy",
    "it's been a long day",
    "I want to try something different",
    "feeling adventurous",
    "need comfort",
]


async def test_clarification_quality():
    """Test clarification responses for vague queries."""
    print("=" * 80)
    print("CLARIFICATION QUALITY TEST - Reasoning=True")
    print("=" * 80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model: {settings.ollama_model}")
    print()

    # Initialize components
    print("Initializing components...")

    # Main LLM for recommendations (fast)
    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.llm_temperature,
        num_predict=settings.llm_max_tokens,
        reasoning=False,
    )

    # LLM for clarification (thoughtful with reasoning)
    llm_clarification = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.llm_temperature,
        num_predict=settings.llm_max_tokens * 2,
        reasoning=True,  # Enable thinking for better clarification
    )

    # Retrieval components
    retriever = RecipeRetriever(
        chroma_dir=Path(settings.chroma_persist_dir),
        embedding_model=settings.embedding_model
    )
    reranker = RecipeReranker(model_name=settings.reranker_model)
    card_builder = RecipeCardBuilder(db_path=settings.sqlite_db_path)
    retrieval_chain = RetrievalRunnable(
        retriever=retriever,
        reranker=reranker,
        card_builder=card_builder,
        settings=settings
    )

    # Memory stores
    profile_store = ProfileStore(db_path=settings.sqlite_db_path)
    session_store = SessionStore(db_path=settings.sqlite_db_path)
    feedback_store = FeedbackStore(db_path=settings.sqlite_db_path)
    history_store = HistoryStore(db_path=settings.sqlite_db_path)

    profile = profile_store.load()

    print("Ready!\n")
    print("=" * 80)

    results = []

    for i, query in enumerate(CLARIFICATION_QUERIES, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{len(CLARIFICATION_QUERIES)}: \"{query}\"")
        print("=" * 80)

        # Fresh session for each query
        session_id = session_store.create()
        session = session_store.get(session_id)

        # Compute exclusions
        exclude_ids = (
            feedback_store.get_liked_recipe_ids(limit=20) |
            feedback_store.get_disliked_recipe_ids() |
            history_store.get_recently_cooked_ids(days=7)
        )

        # Build chain with clarification LLM
        chain = build_chat_chain(
            llm=llm,
            retrieval_chain=retrieval_chain,
            profile=profile,
            session=session,
            rolling_summary="",
            exclude_recipe_ids=exclude_ids,
            llm_clarification=llm_clarification,
        )

        # Get response
        try:
            result = await chain.ainvoke({"user_input": query})
            response = result.get("response", "")
            cards = result.get("cards", [])

            # Determine if it was clarification or recommendation
            was_clarification = len(cards) == 0

            print(f"\nUSER: {query}")
            print(f"\nASSISTANT: {response}")
            print(f"\n[Mode: {'CLARIFICATION' if was_clarification else 'RECOMMENDATION'}]")
            if cards:
                print(f"[Cards: {len(cards)}]")

            # Quality indicators for clarification
            quality_indicators = {
                "asks_question": "?" in response,
                "offers_options": any(word in response.lower() for word in ["or", "would you", "do you", "what about", "how about"]),
                "mentions_cuisines": any(cuisine in response.lower() for cuisine in ["italian", "mexican", "asian", "thai", "indian", "chinese", "french", "greek"]),
                "mentions_constraints": any(word in response.lower() for word in ["time", "quick", "ingredients", "dietary", "vegetarian", "spicy"]),
                "engaging_tone": any(word in response.lower() for word in ["tonight", "craving", "mood", "feeling", "love"]),
            }

            if was_clarification:
                quality_score = sum(quality_indicators.values())
                print(f"\n[Quality Score: {quality_score}/5]")
                for indicator, passed in quality_indicators.items():
                    status = "[x]" if passed else "[ ]"
                    print(f"  {status} {indicator.replace('_', ' ')}")

            results.append({
                "query": query,
                "response": response,
                "was_clarification": was_clarification,
                "quality_indicators": quality_indicators if was_clarification else None,
            })

        except Exception as e:
            print(f"\n[ERROR: {e}]")
            results.append({
                "query": query,
                "response": None,
                "error": str(e),
            })

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    clarifications = [r for r in results if r.get("was_clarification")]
    recommendations = [r for r in results if not r.get("was_clarification") and r.get("response")]
    errors = [r for r in results if r.get("error")]

    print(f"\nTotal queries: {len(CLARIFICATION_QUERIES)}")
    print(f"Triggered clarification: {len(clarifications)}")
    print(f"Gave recommendations: {len(recommendations)}")
    print(f"Errors: {len(errors)}")

    if clarifications:
        avg_quality = sum(
            sum(r["quality_indicators"].values())
            for r in clarifications
        ) / len(clarifications)
        print(f"\nAverage clarification quality: {avg_quality:.1f}/5")

    if recommendations:
        print(f"\nQueries that got recommendations instead of clarification:")
        for r in recommendations:
            print(f"  - \"{r['query']}\"")

    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    asyncio.run(test_clarification_quality())

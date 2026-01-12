"""CLI application for the recipe assistant."""

import typer
from rich.console import Console
from rich.panel import Panel
from src import __version__
from src.app.settings import settings
from src.app.logging_config import configure_logging, get_logger
from src.ingest import cli as ingest_cli

# Initialize Typer app
app = typer.Typer(
    name="recipe-assistant",
    help="Local recipe assistant using RAG with Llama 3.3 70B",
    add_completion=False
)

# Add ingest subcommand
app.add_typer(ingest_cli.app, name="ingest")

# Initialize Rich console for pretty output
console = Console()

# Configure logging
configure_logging(settings.log_level)
logger = get_logger(__name__)


@app.command()
def chat():
    """Start an interactive recipe assistant chat session."""
    import asyncio
    from pathlib import Path

    # Check if vector store and database exist
    chroma_dir = Path(settings.chroma_persist_dir)
    if not chroma_dir.exists():
        console.print(f"[red]ERROR: Vector store not found at: {chroma_dir}[/red]")
        console.print("[yellow]Run 'ingest embed' first to build the vector store[/yellow]")
        raise typer.Exit(1)

    if not settings.sqlite_db_path.exists():
        console.print(f"[red]ERROR: Database not found at: {settings.sqlite_db_path}[/red]")
        console.print("[yellow]Run 'ingest process' first to build the database[/yellow]")
        raise typer.Exit(1)

    # Run async chat session
    asyncio.run(async_chat_session())


def resolve_recipe_reference(ref: str, last_cards: list) -> tuple[str, str] | None:
    """Resolve recipe reference (number or name) to (recipe_id, title).

    Args:
        ref: Reference string (e.g., "1", "chicken tacos")
        last_cards: List of last recommended RecipeCard objects

    Returns:
        Tuple of (recipe_id, title) or None if not found
    """
    ref = ref.strip().strip('"\'')

    # Return None for empty reference
    if not ref:
        return None

    # Try as number first (1-indexed)
    if ref.isdigit():
        idx = int(ref) - 1
        if 0 <= idx < len(last_cards):
            return (last_cards[idx].recipe_id, last_cards[idx].title)
        return None

    # Try fuzzy match on title
    for card in last_cards:
        if ref.lower() in card.title.lower():
            return (card.recipe_id, card.title)

    return None


def display_full_recipe(recipe, console: Console):
    """Display complete recipe with ingredients and instructions.

    Args:
        recipe: Recipe object from database
        console: Rich console for output
    """
    from src.domain.models import Recipe

    rating_str = ""
    if recipe.rating_avg and recipe.rating_count:
        rating_str = f"Rating: {recipe.rating_avg:.1f}/5 ({recipe.rating_count} reviews)\n"

    time_str = ""
    if recipe.minutes:
        rating_str += f"Time: {recipe.minutes} minutes\n"

    ingredients_str = "\n".join(f"  • {ing}" for ing in recipe.ingredients)
    instructions_str = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(recipe.instructions))

    console.print(Panel(
        f"[bold]{recipe.title}[/bold]\n"
        f"{rating_str}{time_str}\n"
        f"[bold]Ingredients:[/bold]\n{ingredients_str}\n\n"
        f"[bold]Instructions:[/bold]\n{instructions_str}",
        title="Recipe Details",
        border_style="cyan"
    ))


async def async_chat_session():
    """Async chat session with LLM integration."""
    from pathlib import Path
    from langchain_ollama import ChatOllama
    from src.retrieval.retriever import RecipeRetriever
    from src.retrieval.rerank import RecipeReranker
    from src.retrieval.recipe_cards import RecipeCardBuilder
    from src.chains.retrieval import RetrievalRunnable
    from src.chains.chat_chain import build_chat_chain
    from src.memory import ProfileStore, SessionStore, RollingSummarizer, FeedbackStore, HistoryStore
    from src.domain.models import RecipeFeedback
    from src.ingest.build_db import get_recipe_by_id

    console.print(Panel.fit(
        "[bold cyan]Recipe Assistant[/bold cyan]\n"
        "Local recipe recommendation powered by RAG + Llama 3.3 70B\n\n"
        "Commands:\n"
        "  /new          - Start a new session\n"
        "  /prefs        - Show your preferences\n"
        "  /like <ref>   - Like a recipe (by number or name)\n"
        "  /dislike <ref>- Dislike a recipe\n"
        "  /rate <1-5> <ref> - Rate a recipe\n"
        "  /show <ref>   - Show full recipe details\n"
        "  /cooked <ref> - Mark recipe as cooked\n"
        "  /history      - Show cooking history\n"
        "  quit          - Exit the chat",
        border_style="cyan"
    ))

    logger.info("Starting chat session")

    try:
        # Initialize components
        console.print("[dim]Initializing LLM and retrieval components...[/dim]")

        # Initialize LLM
        llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            num_predict=settings.llm_max_tokens,
        )

        # Initialize retrieval components
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

        # Initialize memory stores
        profile_store = ProfileStore(db_path=settings.sqlite_db_path)
        session_store = SessionStore(db_path=settings.sqlite_db_path)
        feedback_store = FeedbackStore(db_path=settings.sqlite_db_path)
        history_store = HistoryStore(db_path=settings.sqlite_db_path)
        summarizer = RollingSummarizer()

        # Load profile and session
        profile = profile_store.load()
        session_id, session = session_store.get_or_create_current()
        rolling_summary = session_store.get_summary(session_id)

        # Track last recommended cards for feedback commands
        last_recommended_cards = []

        console.print("[green]Ready![/green]\n")

        logger.info("Chat components initialized", session_id=session_id)

    except Exception as e:
        console.print(f"[red]ERROR: Failed to initialize chat: {e}[/red]")
        if "Connection" in str(e) or "connect" in str(e).lower():
            console.print("[yellow]Is Ollama running? Start with: ollama serve[/yellow]")
        logger.exception("Chat initialization error")
        raise typer.Exit(1)

    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = console.input("\n[bold green]You:[/bold green] ")

            # Check for commands
            if user_input.strip().lower() in ("quit", "exit"):
                console.print("[yellow]Goodbye![/yellow]")
                logger.info("Chat session ended by user")
                break

            if user_input.strip().lower() == "/new":
                session_id = session_store.create()
                session = session_store.get(session_id)
                rolling_summary = ""
                console.print("[green]✓ Started new session[/green]")
                logger.info("New session created", session_id=session_id)
                continue

            if user_input.strip().lower() == "/prefs":
                console.print(f"\n[bold]Your Preferences:[/bold]")
                console.print(f"  Spice level: {profile.spice_level}")
                console.print(f"  Diet: {profile.diet}")
                if profile.avoid_ingredients:
                    console.print(f"  Avoid: {', '.join(profile.avoid_ingredients)}")
                if profile.preferred_cuisines:
                    console.print(f"  Cuisines: {', '.join(profile.preferred_cuisines)}")
                continue

            # /like command
            if user_input.strip().lower().startswith("/like"):
                ref = user_input[5:].strip()
                if not ref:
                    console.print("[yellow]Usage: /like <number or recipe name>[/yellow]")
                    continue
                result = resolve_recipe_reference(ref, last_recommended_cards)
                if result:
                    recipe_id, title = result
                    feedback_store.add_feedback(RecipeFeedback(
                        recipe_id=recipe_id,
                        feedback_type="like",
                        session_id=session_id
                    ))
                    console.print(f"[green]✓ Liked: {title}[/green]")
                else:
                    console.print(f"[yellow]Recipe not found: {ref}[/yellow]")
                continue

            # /dislike command
            if user_input.strip().lower().startswith("/dislike"):
                ref = user_input[8:].strip()
                if not ref:
                    console.print("[yellow]Usage: /dislike <number or recipe name>[/yellow]")
                    continue
                result = resolve_recipe_reference(ref, last_recommended_cards)
                if result:
                    recipe_id, title = result
                    feedback_store.add_feedback(RecipeFeedback(
                        recipe_id=recipe_id,
                        feedback_type="dislike",
                        session_id=session_id
                    ))
                    console.print(f"[yellow]✓ Disliked: {title}[/yellow]")
                else:
                    console.print(f"[yellow]Recipe not found: {ref}[/yellow]")
                continue

            # /rate command
            if user_input.strip().lower().startswith("/rate"):
                parts = user_input[5:].strip().split(maxsplit=1)
                if len(parts) < 2:
                    console.print("[yellow]Usage: /rate <1-5> <number or recipe name>[/yellow]")
                    continue
                try:
                    rating = int(parts[0])
                    if not 1 <= rating <= 5:
                        console.print("[yellow]Rating must be between 1 and 5[/yellow]")
                        continue
                    ref = parts[1]
                    result = resolve_recipe_reference(ref, last_recommended_cards)
                    if result:
                        recipe_id, title = result
                        feedback_store.add_feedback(RecipeFeedback(
                            recipe_id=recipe_id,
                            feedback_type="rate",
                            rating=rating,
                            session_id=session_id
                        ))
                        console.print(f"[green]✓ Rated {title}: {rating}/5[/green]")
                    else:
                        console.print(f"[yellow]Recipe not found: {ref}[/yellow]")
                except ValueError:
                    console.print("[yellow]Invalid rating. Must be a number 1-5[/yellow]")
                continue

            # /show command
            if user_input.strip().lower().startswith("/show"):
                ref = user_input[5:].strip()
                if not ref:
                    console.print("[yellow]Usage: /show <number or recipe name>[/yellow]")
                    continue
                result = resolve_recipe_reference(ref, last_recommended_cards)
                if result:
                    recipe_id, title = result
                    recipe = get_recipe_by_id(settings.sqlite_db_path, recipe_id)
                    if recipe:
                        display_full_recipe(recipe, console)
                    else:
                        console.print(f"[yellow]Recipe not found in database: {recipe_id}[/yellow]")
                else:
                    console.print(f"[yellow]Recipe not found: {ref}[/yellow]")
                continue

            # /cooked command
            if user_input.strip().lower().startswith("/cooked"):
                ref = user_input[7:].strip()
                if not ref:
                    console.print("[yellow]Usage: /cooked <number or recipe name>[/yellow]")
                    continue
                result = resolve_recipe_reference(ref, last_recommended_cards)
                if result:
                    recipe_id, title = result
                    history_store.add_cooked(recipe_id)
                    console.print(f"[green]✓ Marked as cooked: {title}[/green]")
                else:
                    console.print(f"[yellow]Recipe not found: {ref}[/yellow]")
                continue

            # /history command
            if user_input.strip().lower() == "/history":
                history = history_store.get_cooking_history(limit=10)
                if history:
                    console.print(f"\n[bold]Recent Cooking History:[/bold]")
                    for i, entry in enumerate(history, 1):
                        # Get recipe title
                        recipe = get_recipe_by_id(settings.sqlite_db_path, entry.recipe_id)
                        title = recipe.title if recipe else entry.recipe_id
                        cooked_str = entry.cooked_at.strftime("%Y-%m-%d") if entry.cooked_at else "Unknown"
                        console.print(f"  {i}. {title} (cooked: {cooked_str})")
                else:
                    console.print("[yellow]No cooking history yet[/yellow]")
                continue

            # Skip empty input
            if not user_input.strip():
                continue

            # Compute exclusion set from feedback and history
            exclude_ids = (
                feedback_store.get_liked_recipe_ids(limit=20) |
                feedback_store.get_disliked_recipe_ids() |
                history_store.get_recently_cooked_ids(days=7)
            )

            # Build chain with current context
            chain = build_chat_chain(
                llm=llm,
                retrieval_chain=retrieval_chain,
                profile=profile,
                session=session,
                rolling_summary=rolling_summary,
                exclude_recipe_ids=exclude_ids
            )

            # Invoke chain
            console.print("\n[dim]Thinking...[/dim]")

            response = await chain.ainvoke({"user_input": user_input})

            # Display response
            console.print(f"\n[bold blue]Assistant:[/bold blue] {response}")

            # Capture recipe cards if this was a recommendation (not clarification)
            # Check if response contains recipe recommendations by invoking retrieval
            from src.chains.extractors import ConstraintExtractor
            from src.chains.chat_chain import should_clarify
            extractor = ConstraintExtractor()
            constraints = extractor.extract_constraints(user_input)

            # If this wasn't a clarification, capture the cards for feedback commands
            input_data = {"user_input": user_input, "constraints": constraints, "session": session}
            if not should_clarify(input_data):
                try:
                    retrieval_result = retrieval_chain.invoke(input_data)
                    last_recommended_cards = retrieval_result.get("cards", [])
                except Exception as e:
                    logger.warning("Failed to capture recipe cards", error=str(e))
                    last_recommended_cards = []

            # Update rolling summary
            rolling_summary = summarizer.update_summary(rolling_summary, constraints, user_input)
            session_store.update_summary(session_id, rolling_summary)

            logger.info("Processed user turn", input_length=len(user_input), response_length=len(response))

        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            logger.info("Chat session interrupted")
            break
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {e}")
            if "Connection" in str(e) or "connect" in str(e).lower():
                console.print("[yellow]Lost connection to Ollama. Is it still running?[/yellow]")
            logger.exception("Chat error")


@app.command()
def version():
    """Show version information."""
    console.print(f"Recipe Assistant version [cyan]{__version__}[/cyan]")
    logger.info("Version command executed", version=__version__)


@app.command()
def config():
    """Show current configuration settings."""
    console.print("[bold]Current Configuration:[/bold]\n")
    console.print(f"Ollama URL: [cyan]{settings.ollama_base_url}[/cyan]")
    console.print(f"Ollama Model: [cyan]{settings.ollama_model}[/cyan]")
    console.print(f"Embedding Model: [cyan]{settings.embedding_model}[/cyan]")
    console.print(f"Reranker Model: [cyan]{settings.reranker_model}[/cyan]")
    console.print(f"Chroma Dir: [cyan]{settings.chroma_persist_dir}[/cyan]")
    console.print(f"SQLite DB: [cyan]{settings.sqlite_db_path}[/cyan]")
    console.print(f"k_retrieve: [cyan]{settings.k_retrieve}[/cyan]")
    console.print(f"k_rerank: [cyan]{settings.k_rerank}[/cyan]")
    console.print(f"k_context: [cyan]{settings.k_context}[/cyan]")
    console.print(f"Log Level: [cyan]{settings.log_level}[/cyan]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query (e.g., 'chicken tomato spicy')"),
    k: int = typer.Option(10, help="Number of results to return"),
    rerank: bool = typer.Option(False, "--rerank", "-r", help="Enable cross-encoder reranking"),
    cards: bool = typer.Option(False, "--cards", "-c", help="Show recipe cards (implies --rerank)")
):
    """Search for recipes using vector similarity with optional reranking and card display."""
    import time
    from pathlib import Path
    from src.retrieval.retriever import RecipeRetriever
    from src.retrieval.rerank import RecipeReranker
    from src.retrieval.recipe_cards import RecipeCardBuilder
    from rich.panel import Panel

    chroma_dir = Path(settings.chroma_persist_dir)

    # Check if vector store exists
    if not chroma_dir.exists():
        console.print(f"[red]ERROR Vector store not found at: {chroma_dir}[/red]")
        console.print("[yellow]Run 'ingest embed' first to build the vector store[/yellow]")
        raise typer.Exit(1)

    # Enable rerank if cards requested
    if cards:
        rerank = True

    mode = "cards" if cards else "rerank" if rerank else "vector"
    console.print(f"\n[cyan]Searching for:[/cyan] {query}")
    console.print(f"[dim]Mode: {mode} | Retrieving top {k} results...[/dim]\n")

    try:
        # Initialize retriever
        retriever = RecipeRetriever(
            chroma_dir=chroma_dir,
            embedding_model=settings.embedding_model
        )

        # Perform search
        start_time = time.time()

        # Step 1: Vector retrieval
        k_retrieve = settings.k_retrieve if rerank else k
        results = retriever.search(query, k=k_retrieve)

        # Step 2: Rerank (optional)
        if rerank and results:
            reranker = RecipeReranker(model_name=settings.reranker_model)
            results = reranker.rerank(query, results, top_k=settings.k_rerank)

        # Limit to k results if not using cards
        if not cards:
            results = results[:k]

        elapsed_ms = (time.time() - start_time) * 1000

        # Display results
        if not results:
            console.print("[yellow]No results found.[/yellow]")
        elif cards:
            # Step 3: Build and display cards
            builder = RecipeCardBuilder(db_path=settings.sqlite_db_path)
            recipe_cards = builder.build_cards(results[:settings.k_context], query)

            console.print(f"[bold]Found {len(recipe_cards)} recipe cards:[/bold]\n")

            for i, card in enumerate(recipe_cards, 1):
                # Build card content
                content = []

                # Rating
                if card.rating_avg:
                    stars = "★" * int(card.rating_avg) + "☆" * (5 - int(card.rating_avg))
                    rating_count = f"({card.rating_count} reviews)" if card.rating_count else ""
                    content.append(f"[yellow]{stars}[/yellow] {card.rating_avg:.1f}/5 {rating_count}")

                # Time
                if card.time_total:
                    content.append(f"[cyan]⏱ {card.time_total} minutes[/cyan]")

                # Tags
                if card.tags:
                    content.append(f"[dim]Tags: {', '.join(card.tags[:8])}[/dim]")

                # Key ingredients
                if card.key_ingredients:
                    ing_list = ', '.join(card.key_ingredients[:10])
                    content.append(f"[green]Key ingredients:[/green] {ing_list}")

                # Summary
                if card.one_sentence_summary:
                    content.append(f"\n{card.one_sentence_summary}")

                # Why match
                if card.why_match:
                    content.append(f"\n[bold cyan]Why this matches:[/bold cyan] {card.why_match}")

                console.print(Panel(
                    "\n".join(content),
                    title=f"[bold]{i}. {card.title}[/bold]",
                    border_style="blue"
                ))
        else:
            # Simple result list
            console.print(f"[bold]Found {len(results)} recipes in {elapsed_ms:.0f}ms:[/bold]\n")

            for i, result in enumerate(results, 1):
                rating_str = f"{result.rating_avg:.1f}" if result.rating_avg else "N/A"
                rating_count_str = f"({result.rating_count})" if result.rating_count else ""
                time_str = f"{result.minutes}m" if result.minutes else "N/A"

                console.print(
                    f"{i:2}. [bold]{result.title}[/bold] "
                    f"[dim]| Score: {result.score:.3f} | "
                    f"Rating: {rating_str} {rating_count_str} | "
                    f"Time: {time_str}[/dim]"
                )

        # Timing summary
        console.print(f"\n[dim]Completed in {elapsed_ms:.0f}ms[/dim]")

        # Warn if search is slow
        if elapsed_ms > 1000:
            console.print(f"[yellow]WARNING: Search took {elapsed_ms:.0f}ms (target <1000ms)[/yellow]")

        logger.info("Search completed", query=query, mode=mode, results_count=len(results), time_ms=elapsed_ms)

    except Exception as e:
        console.print(f"[red]ERROR Search failed: {e}[/red]")
        logger.exception("Search error")
        raise typer.Exit(1)


def main():
    """Entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()

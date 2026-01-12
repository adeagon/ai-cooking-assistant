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
    console.print(Panel.fit(
        "[bold cyan]Recipe Assistant[/bold cyan]\n"
        "Local recipe recommendation powered by RAG\n\n"
        "Type 'quit' or 'exit' to end the session.",
        border_style="cyan"
    ))

    logger.info("Starting chat session")

    while True:
        try:
            # Get user input
            user_input = console.input("\n[bold green]You:[/bold green] ")

            # Check for exit commands
            if user_input.strip().lower() in ("quit", "exit"):
                console.print("[yellow]Goodbye![/yellow]")
                logger.info("Chat session ended by user")
                break

            # Skip empty input
            if not user_input.strip():
                continue

            # Placeholder response (will be replaced with actual LLM integration)
            console.print(
                f"\n[bold blue]Assistant:[/bold blue] You said: {user_input}\n"
                "[dim]This is a placeholder response. "
                "LLM integration coming in later phases.[/dim]"
            )

            logger.debug("Processed user input", input_length=len(user_input))

        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            logger.info("Chat session interrupted")
            break
        except Exception as e:
            console.print(f"[red]Error:[/red] {e}")
            logger.error("Chat error", error=str(e))


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

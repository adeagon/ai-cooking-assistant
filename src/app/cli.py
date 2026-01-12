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
    k: int = typer.Option(10, help="Number of results to return")
):
    """Search for recipes using vector similarity."""
    import time
    from pathlib import Path
    from src.retrieval.retriever import RecipeRetriever

    chroma_dir = Path(settings.chroma_persist_dir)

    # Check if vector store exists
    if not chroma_dir.exists():
        console.print(f"[red]ERROR Vector store not found at: {chroma_dir}[/red]")
        console.print("[yellow]Run 'ingest embed' first to build the vector store[/yellow]")
        raise typer.Exit(1)

    console.print(f"\n[cyan]Searching for:[/cyan] {query}")
    console.print(f"[dim]Retrieving top {k} results...[/dim]\n")

    try:
        # Initialize retriever
        retriever = RecipeRetriever(
            chroma_dir=chroma_dir,
            embedding_model=settings.embedding_model
        )

        # Perform search
        start_time = time.time()
        results = retriever.search(query, k=k)
        elapsed_ms = (time.time() - start_time) * 1000

        # Display results
        if not results:
            console.print("[yellow]No results found.[/yellow]")
        else:
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

            # Warn if search is slow
            if elapsed_ms > 200:
                console.print(f"\n[yellow]WARNING: Search took {elapsed_ms:.0f}ms (target <200ms)[/yellow]")

        logger.info("Search completed", query=query, results_count=len(results), time_ms=elapsed_ms)

    except Exception as e:
        console.print(f"[red]ERROR Search failed: {e}[/red]")
        logger.exception("Search error")
        raise typer.Exit(1)


def main():
    """Entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()

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


def main():
    """Entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()

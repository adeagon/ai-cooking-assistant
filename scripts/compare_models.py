"""Compare search quality between embedding models."""

import time
from pathlib import Path
from src.retrieval.retriever import RecipeRetriever
from rich.console import Console
from rich.table import Table

console = Console()


def compare_search_results(queries: list[str], k: int = 10):
    """Compare search results for different models."""

    # Load both models
    console.print("\n[cyan]Loading models...[/cyan]")

    old_model = "all-MiniLM-L6-v2"
    new_model = "all-mpnet-base-v2"

    old_retriever = RecipeRetriever(
        chroma_dir=Path("data/chroma_backup_miniLM"),
        embedding_model=old_model
    )

    new_retriever = RecipeRetriever(
        chroma_dir=Path("data/chroma"),
        embedding_model=new_model
    )

    console.print(f"[green]Loaded both models[/green]\n")

    # Test each query
    for query in queries:
        console.print(f"\n[bold]Query: {query}[/bold]")
        console.print("─" * 80)

        # Old model
        start = time.time()
        old_results = old_retriever.search(query, k=k)
        old_time = (time.time() - start) * 1000

        # New model
        start = time.time()
        new_results = new_retriever.search(query, k=k)
        new_time = (time.time() - start) * 1000

        # Create comparison table
        table = Table(show_header=True, header_style="bold")
        table.add_column("Rank", style="dim", width=4)
        table.add_column(f"Old Model ({old_model})", width=35)
        table.add_column("Score", justify="right", width=6)
        table.add_column(f"New Model ({new_model})", width=35)
        table.add_column("Score", justify="right", width=6)

        for i in range(min(5, k)):
            old_title = old_results[i].title[:33] + "..." if len(old_results[i].title) > 33 else old_results[i].title
            new_title = new_results[i].title[:33] + "..." if len(new_results[i].title) > 33 else new_results[i].title

            table.add_row(
                str(i + 1),
                old_title,
                f"{old_results[i].score:.3f}",
                new_title,
                f"{new_results[i].score:.3f}"
            )

        console.print(table)

        # Show timing
        console.print(f"\n[dim]Old model: {old_time:.0f}ms | New model: {new_time:.0f}ms[/dim]")

        # Calculate improvements
        avg_old_score = sum(r.score for r in old_results[:5]) / 5
        avg_new_score = sum(r.score for r in new_results[:5]) / 5
        score_improvement = ((avg_new_score - avg_old_score) / avg_old_score) * 100

        console.print(f"[dim]Avg top-5 score: Old={avg_old_score:.3f}, New={avg_new_score:.3f} "
                     f"({score_improvement:+.1f}%)[/dim]")


if __name__ == "__main__":
    # Test queries covering different search types
    test_queries = [
        "chicken tomato spicy",
        "healthy vegetarian dinner under 30 minutes",
        "quick pasta with basil and garlic",
        "chocolate dessert easy",
        "asian stir fry vegetables",
        "comfort food soup",
    ]

    console.print("\n[bold cyan]Comparing Embedding Models[/bold cyan]")
    console.print(f"Testing {len(test_queries)} queries with k=10")

    compare_search_results(test_queries, k=10)

    console.print("\n[bold green]Comparison complete![/bold green]\n")

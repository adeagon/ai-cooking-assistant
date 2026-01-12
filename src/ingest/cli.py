"""CLI commands for data ingestion."""

import json
from pathlib import Path
import typer
from rich.console import Console
from rich.progress import track
from src.app.settings import settings
from src.app.logging_config import get_logger
from src.ingest.download import download_foodcom_dataset
from src.ingest.load_foodcom import load_recipes, compute_ratings
from src.ingest.normalize import extract_key_ingredients
from src.ingest.filters import apply_quality_filters
from src.ingest.build_db import create_tables, insert_recipes, get_recipe_by_id, get_stats
from src.domain.models import Recipe

app = typer.Typer(help="Data ingestion commands")
console = Console()
logger = get_logger(__name__)


@app.command()
def download():
    """Download Food.com dataset from Kaggle."""
    raw_dir = Path("data/raw")

    console.print("[cyan]Downloading Food.com dataset...[/cyan]")

    try:
        download_foodcom_dataset(raw_dir)
        console.print("[green]OK Dataset downloaded successfully[/green]")
        console.print(f"Location: {raw_dir}")
    except Exception as e:
        console.print(f"[red]ERROR Download failed: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def process(
    min_rating_count: int = typer.Option(3, help="Minimum number of ratings"),
    min_rating_avg: float = typer.Option(3.5, help="Minimum average rating")
):
    """Process raw data: parse, normalize, filter, save."""
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    db_dir = Path("data/sqlite")

    recipes_csv = raw_dir / "RAW_recipes.csv"
    interactions_csv = raw_dir / "RAW_interactions.csv"

    if not recipes_csv.exists():
        console.print(f"[red]ERROR Recipes file not found: {recipes_csv}[/red]")
        console.print("[yellow]Run 'ingest download' first[/yellow]")
        raise typer.Exit(1)

    if not interactions_csv.exists():
        console.print(f"[red]ERROR Interactions file not found: {interactions_csv}[/red]")
        console.print("[yellow]Run 'ingest download' first[/yellow]")
        raise typer.Exit(1)

    # Create output directories
    processed_dir.mkdir(parents=True, exist_ok=True)
    db_dir.mkdir(parents=True, exist_ok=True)

    console.print("[cyan]Step 1/5: Computing ratings...[/cyan]")
    rating_stats = compute_ratings(recipes_csv, interactions_csv)
    console.print(f"[green]OK Computed ratings for {len(rating_stats)} recipes[/green]")

    console.print("[cyan]Step 2/5: Loading and filtering recipes...[/cyan]")
    filtered_recipes = []
    total_recipes = 0

    for recipe_dict in track(
        load_recipes(recipes_csv),
        description="Processing recipes",
        console=console
    ):
        total_recipes += 1
        recipe_id = recipe_dict['recipe_id']
        stats = rating_stats.get(recipe_id)

        if apply_quality_filters(recipe_dict, stats, min_rating_count, min_rating_avg):
            filtered_recipes.append((recipe_dict, stats))

    console.print(f"[green]OK Filtered {len(filtered_recipes)} recipes from {total_recipes} total[/green]")

    console.print("[cyan]Step 3/5: Normalizing ingredients...[/cyan]")
    processed_recipes = []

    for recipe_dict, stats in track(
        filtered_recipes,
        description="Normalizing ingredients",
        console=console
    ):
        normalized_ingredients = extract_key_ingredients(recipe_dict['ingredients'])

        recipe = Recipe(
            recipe_id=recipe_dict['recipe_id'],
            title=recipe_dict['name'],
            ingredients=recipe_dict['ingredients'],
            ingredients_normalized=normalized_ingredients,
            instructions=recipe_dict['steps'],
            tags=recipe_dict['tags'],
            rating_avg=stats.rating_avg,
            rating_count=stats.rating_count,
            minutes=recipe_dict['minutes'],
            n_steps=recipe_dict['n_steps'],
            n_ingredients=recipe_dict['n_ingredients'],
            source="foodcom"
        )
        processed_recipes.append(recipe)

    console.print(f"[green]OK Normalized {len(processed_recipes)} recipes[/green]")

    console.print("[cyan]Step 4/5: Saving to JSONL...[/cyan]")
    jsonl_path = processed_dir / "recipes.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for recipe in processed_recipes:
            f.write(recipe.model_dump_json() + '\n')

    console.print(f"[green]OK Saved to {jsonl_path}[/green]")

    console.print("[cyan]Step 5/5: Building SQLite database...[/cyan]")
    db_path = db_dir / "recipes.db"
    create_tables(db_path)
    count = insert_recipes(db_path, iter(processed_recipes))

    console.print(f"[green]OK Inserted {count} recipes into {db_path}[/green]")
    console.print("\n[bold green]Processing complete![/bold green]")


@app.command()
def stats():
    """Show dataset statistics."""
    db_path = Path("data/sqlite/recipes.db")

    if not db_path.exists():
        console.print(f"[red]ERROR Database not found: {db_path}[/red]")
        console.print("[yellow]Run 'ingest process' first[/yellow]")
        raise typer.Exit(1)

    stats_dict = get_stats(db_path)

    console.print("\n[bold]Dataset Statistics:[/bold]\n")
    console.print(f"Total recipes: [cyan]{stats_dict['total_recipes']:,}[/cyan]")
    console.print(f"Average rating: [cyan]{stats_dict['avg_rating']}[/cyan]")
    console.print(f"Average cooking time: [cyan]{stats_dict['avg_minutes']} minutes[/cyan]")
    console.print(f"\nDatabase: [cyan]{db_path}[/cyan]")


@app.command()
def sample(recipe_id: str):
    """Show a sample recipe by ID."""
    db_path = Path("data/sqlite/recipes.db")

    if not db_path.exists():
        console.print(f"[red]ERROR Database not found: {db_path}[/red]")
        console.print("[yellow]Run 'ingest process' first[/yellow]")
        raise typer.Exit(1)

    recipe = get_recipe_by_id(db_path, recipe_id)

    if recipe is None:
        console.print(f"[red]ERROR Recipe not found: {recipe_id}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]{recipe.title}[/bold]")
    console.print(f"Rating: [cyan]{recipe.rating_avg}[/cyan] ({recipe.rating_count} ratings)")
    console.print(f"Time: [cyan]{recipe.minutes} minutes[/cyan]")
    console.print(f"Tags: [cyan]{', '.join(recipe.tags[:5])}[/cyan]")

    console.print(f"\n[bold]Ingredients ({len(recipe.ingredients)}):[/bold]")
    for ing in recipe.ingredients[:10]:
        console.print(f"  • {ing}")

    console.print(f"\n[bold]Normalized Ingredients:[/bold]")
    console.print(f"  {', '.join(recipe.ingredients_normalized[:15])}")

    console.print(f"\n[bold]Instructions ({len(recipe.instructions)} steps):[/bold]")
    for i, step in enumerate(recipe.instructions[:5], 1):
        console.print(f"  {i}. {step}")

    if len(recipe.instructions) > 5:
        console.print(f"  ... ({len(recipe.instructions) - 5} more steps)")

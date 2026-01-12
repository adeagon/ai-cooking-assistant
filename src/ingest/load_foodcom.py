"""Load and parse Food.com CSV files."""

import ast
from pathlib import Path
from typing import Iterator
import pandas as pd
from src.domain.models import RatingStats
from src.app.logging_config import get_logger

logger = get_logger(__name__)


def load_recipes(csv_path: Path, chunksize: int = 10000) -> Iterator[dict]:
    """Load RAW_recipes.csv and yield recipe dicts.

    Args:
        csv_path: Path to RAW_recipes.csv
        chunksize: Number of rows to read at a time

    Yields:
        Recipe dicts with parsed ingredients, steps, and tags
    """
    logger.info("Loading recipes from CSV", path=str(csv_path))

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        for _, row in chunk.iterrows():
            try:
                # Parse string representations of Python lists
                ingredients = ast.literal_eval(row['ingredients']) if pd.notna(row['ingredients']) else []
                steps = ast.literal_eval(row['steps']) if pd.notna(row['steps']) else []
                tags = ast.literal_eval(row['tags']) if pd.notna(row['tags']) else []

                yield {
                    'recipe_id': str(row['id']),
                    'name': row['name'],
                    'minutes': int(row['minutes']) if pd.notna(row['minutes']) else None,
                    'contributor_id': str(row['contributor_id']) if pd.notna(row['contributor_id']) else None,
                    'submitted': row['submitted'],
                    'tags': tags,
                    'nutrition': ast.literal_eval(row['nutrition']) if pd.notna(row['nutrition']) else [],
                    'n_steps': int(row['n_steps']) if pd.notna(row['n_steps']) else None,
                    'steps': steps,
                    'description': row['description'] if pd.notna(row['description']) else '',
                    'ingredients': ingredients,
                    'n_ingredients': int(row['n_ingredients']) if pd.notna(row['n_ingredients']) else None,
                }
            except (ValueError, SyntaxError) as e:
                logger.warning(
                    "Failed to parse recipe",
                    recipe_id=row['id'],
                    error=str(e)
                )
                continue


def load_interactions(csv_path: Path, chunksize: int = 50000) -> Iterator[dict]:
    """Load RAW_interactions.csv and yield interaction dicts.

    Args:
        csv_path: Path to RAW_interactions.csv
        chunksize: Number of rows to read at a time

    Yields:
        Interaction dicts with user_id, recipe_id, rating
    """
    logger.info("Loading interactions from CSV", path=str(csv_path))

    for chunk in pd.read_csv(csv_path, chunksize=chunksize):
        for _, row in chunk.iterrows():
            yield {
                'user_id': str(row['user_id']),
                'recipe_id': str(row['recipe_id']),
                'date': row['date'],
                'rating': int(row['rating']) if pd.notna(row['rating']) else None,
                'review': row['review'] if pd.notna(row['review']) else '',
            }


def compute_ratings(
    recipes_csv: Path,
    interactions_csv: Path
) -> dict[str, RatingStats]:
    """Aggregate interactions to compute rating statistics per recipe.

    Args:
        recipes_csv: Path to RAW_recipes.csv
        interactions_csv: Path to RAW_interactions.csv

    Returns:
        Dict mapping recipe_id to RatingStats
    """
    logger.info("Computing rating statistics")

    # Load interactions and compute stats
    interactions_df = pd.read_csv(interactions_csv)

    # Filter out ratings of 0 (these are reviews without ratings)
    rated_interactions = interactions_df[interactions_df['rating'] > 0]

    # Compute aggregate stats
    rating_stats = rated_interactions.groupby('recipe_id').agg({
        'rating': ['mean', 'count']
    }).reset_index()

    rating_stats.columns = ['recipe_id', 'rating_avg', 'rating_count']

    # Convert to dict
    stats_dict = {}
    for _, row in rating_stats.iterrows():
        recipe_id = str(int(row['recipe_id']))  # Convert to int first to match load_recipes format
        stats_dict[recipe_id] = RatingStats(
            rating_avg=float(row['rating_avg']),
            rating_count=int(row['rating_count'])
        )

    logger.info(
        "Rating statistics computed",
        total_interactions=len(interactions_df),
        rated_interactions=len(rated_interactions),
        recipes_with_ratings=len(stats_dict)
    )

    return stats_dict

"""Generate embeddings for recipes using sentence-transformers."""

import json
from pathlib import Path
from typing import Iterator
import numpy as np
from sentence_transformers import SentenceTransformer
from src.domain.models import Recipe
from src.app.logging_config import get_logger

logger = get_logger(__name__)


def build_embedding_text(recipe: Recipe) -> str:
    """
    Build embedding text from recipe.

    Format: "{title}. Tags: {tags}. Ingredients: {ingredients_normalized}"

    Args:
        recipe: Recipe object with title, tags, ingredients_normalized

    Returns:
        Formatted embedding text string
    """
    tags_str = ", ".join(recipe.tags) if recipe.tags else ""
    ingredients_str = ", ".join(recipe.ingredients_normalized) if recipe.ingredients_normalized else ""

    parts = [recipe.title]
    if tags_str:
        parts.append(f"Tags: {tags_str}")
    if ingredients_str:
        parts.append(f"Ingredients: {ingredients_str}")

    return ". ".join(parts)


def load_recipes_from_jsonl(path: Path) -> Iterator[Recipe]:
    """
    Load recipes from JSONL file.

    Args:
        path: Path to recipes.jsonl file

    Yields:
        Recipe objects
    """
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                recipe_dict = json.loads(line)
                yield Recipe(**recipe_dict)


def generate_embeddings_batch(
    texts: list[str],
    model: SentenceTransformer,
    show_progress: bool = False
) -> np.ndarray:
    """
    Generate embeddings for a batch of texts.

    Args:
        texts: List of text strings to embed
        model: Loaded SentenceTransformer model
        show_progress: Whether to show progress bar

    Returns:
        numpy array of embeddings with shape (len(texts), embedding_dim)
    """
    embeddings = model.encode(
        texts,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True  # Normalize for cosine similarity
    )
    return embeddings


def process_recipes_in_batches(
    recipes_path: Path,
    model_name: str,
    batch_size: int = 500
) -> Iterator[tuple[list[Recipe], np.ndarray]]:
    """
    Process recipes in batches and generate embeddings.

    Args:
        recipes_path: Path to recipes.jsonl file
        model_name: Name of sentence-transformers model
        batch_size: Number of recipes to process per batch

    Yields:
        Tuples of (recipe_batch, embeddings_batch)
    """
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    recipe_batch = []
    text_batch = []

    for recipe in load_recipes_from_jsonl(recipes_path):
        recipe_batch.append(recipe)
        text_batch.append(build_embedding_text(recipe))

        if len(recipe_batch) >= batch_size:
            embeddings = generate_embeddings_batch(text_batch, model)
            yield recipe_batch, embeddings
            recipe_batch = []
            text_batch = []

    # Process final batch if not empty
    if recipe_batch:
        embeddings = generate_embeddings_batch(text_batch, model)
        yield recipe_batch, embeddings

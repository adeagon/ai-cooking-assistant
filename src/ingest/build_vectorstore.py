"""Build and persist ChromaDB vector store from recipe embeddings."""

from pathlib import Path
import numpy as np
import chromadb
from chromadb.config import Settings as ChromaSettings
from src.domain.models import Recipe
from src.ingest.build_embeddings import process_recipes_in_batches
from src.app.logging_config import get_logger

logger = get_logger(__name__)

# Priority order for cuisine extraction (first match wins)
CUISINE_PRIORITY = [
    "italian", "mexican", "chinese", "indian", "thai", "japanese",
    "french", "greek", "mediterranean", "korean", "vietnamese",
    "american", "southern-united-states", "asian", "european",
    "middle-eastern", "spanish", "german", "british", "irish",
    "cajun", "creole", "caribbean", "african", "brazilian",
]


def _extract_primary_cuisine(tags: list[str]) -> str:
    """Extract primary cuisine from recipe tags.

    Args:
        tags: List of recipe tags (lowercase)

    Returns:
        Primary cuisine string, or empty string if none found
    """
    for cuisine in CUISINE_PRIORITY:
        if cuisine in tags:
            return cuisine
    # Check for regional variants
    if "north-american" in tags:
        return "american"
    return ""  # Empty string for ChromaDB (doesn't support None in filters)


def get_or_create_collection(client: chromadb.Client, collection_name: str = "recipes"):
    """
    Get or create ChromaDB collection.

    Args:
        client: ChromaDB client instance
        collection_name: Name of the collection

    Returns:
        ChromaDB Collection object
    """
    try:
        # Try to get existing collection
        collection = client.get_collection(name=collection_name)
        logger.info(f"Using existing collection: {collection_name}")
    except Exception:
        # Create new collection if it doesn't exist
        collection = client.create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}  # Cosine similarity for semantic search
        )
        logger.info(f"Created new collection: {collection_name}")

    return collection


def build_vectorstore(
    recipes_path: Path,
    chroma_dir: Path,
    embedding_model: str,
    batch_size: int = 500,
    collection_name: str = "recipes"
) -> int:
    """
    Build ChromaDB vector store from recipes.

    Args:
        recipes_path: Path to recipes.jsonl file
        chroma_dir: ChromaDB persistence directory
        embedding_model: Name of sentence-transformers model
        batch_size: Number of recipes to process per batch
        collection_name: Name of the ChromaDB collection

    Returns:
        Total number of recipes indexed
    """
    # Create persist directory if it doesn't exist
    chroma_dir.mkdir(parents=True, exist_ok=True)

    # Initialize ChromaDB with persistent storage
    logger.info(f"Initializing ChromaDB at: {chroma_dir}")
    client = chromadb.PersistentClient(
        path=str(chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False)
    )

    # Get or create collection
    collection = get_or_create_collection(client, collection_name)

    total_indexed = 0

    # Process recipes in batches
    for recipe_batch, embeddings_batch in process_recipes_in_batches(
        recipes_path, embedding_model, batch_size
    ):
        # Prepare batch data for ChromaDB
        ids = []
        documents = []
        metadatas = []
        embeddings_list = []

        for recipe, embedding in zip(recipe_batch, embeddings_batch):
            ids.append(recipe.recipe_id)
            # Store the embedding text as the document
            from src.ingest.build_embeddings import build_embedding_text
            documents.append(build_embedding_text(recipe))

            # Store metadata for filtering
            tags_lower = [t.lower() for t in recipe.tags] if recipe.tags else []

            metadata = {
                "title": recipe.title,
                "tags": ", ".join(recipe.tags) if recipe.tags else "",
                # Structured filterable fields
                "is_vegetarian": "vegetarian" in tags_lower,
                "is_vegan": "vegan" in tags_lower,
                "cuisine": _extract_primary_cuisine(tags_lower),
            }
            # Add numeric fields only if they exist
            if recipe.rating_avg is not None:
                metadata["rating_avg"] = float(recipe.rating_avg)
            if recipe.rating_count is not None:
                metadata["rating_count"] = int(recipe.rating_count)
            if recipe.minutes is not None:
                metadata["minutes"] = int(recipe.minutes)

            metadatas.append(metadata)
            embeddings_list.append(embedding.tolist())

        # Upsert batch to ChromaDB
        collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings_list
        )

        total_indexed += len(recipe_batch)
        logger.info(f"Indexed {total_indexed} recipes...")

    logger.info(f"Completed! Total recipes indexed: {total_indexed}")
    return total_indexed

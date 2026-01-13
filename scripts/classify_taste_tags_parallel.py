"""LLM-based taste classification for recipes with parallel processing.

This script uses Ollama/Qwen to classify recipes with missing taste tags:
- light: Light, refreshing meals (salads, simple proteins, low-cal)
- hearty: Filling, substantial meals (stews, casseroles, meat dishes)
- mild: Not spicy, family-friendly
- rich: Creamy, indulgent, decadent

Uses asyncio for parallel processing (8 concurrent requests by default).

Usage:
    # Start Ollama first
    ollama serve

    # Run classification (can be interrupted and resumed)
    python scripts/classify_taste_tags_parallel.py

    # Run with different concurrency
    python scripts/classify_taste_tags_parallel.py --workers 4
"""

import argparse
import asyncio
import json
import sqlite3
import time
from pathlib import Path

import httpx

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:14b"

TASTE_TAGS = ["light", "hearty", "mild", "rich"]

CLASSIFICATION_PROMPT = """Classify this recipe's taste profile.

Definitions:
- light: Light, refreshing (salads, simple proteins, low-calorie)
- hearty: Filling, substantial (stews, casseroles, meat-heavy)
- mild: Not spicy, family-friendly flavors
- rich: Creamy, indulgent, decadent (butter, cream, cheese)

Recipe: {title}
Ingredients: {ingredients}
Existing tags: {tags}

Reply in this exact format:
TAGS: <comma-separated tags or "none">
CONFIDENCE: <high/medium/low>

Example: "TAGS: hearty, mild
CONFIDENCE: high"
"""


async def classify_recipe_async(
    client: httpx.AsyncClient,
    recipe_id: str,
    title: str,
    ingredients: list,
    tags: list,
) -> tuple[str, list[str], str]:
    """Use LLM to classify recipe taste profile asynchronously.

    Args:
        client: Async HTTP client
        recipe_id: Recipe ID
        title: Recipe title
        ingredients: List of ingredients
        tags: Existing tags on the recipe

    Returns:
        Tuple of (recipe_id, taste_tags, confidence_level)
    """
    prompt = CLASSIFICATION_PROMPT.format(
        title=title,
        ingredients=", ".join(ingredients[:10]),
        tags=", ".join(tags[:10]),
    )

    try:
        response = await client.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=60.0,
        )
        response.raise_for_status()

        result = response.json()["response"].strip()

        # Parse structured response
        new_tags = []
        confidence = "low"

        for line in result.split("\n"):
            line = line.strip().lower()
            if line.startswith("tags:"):
                tag_str = line.replace("tags:", "").strip()
                if tag_str != "none":
                    for tag in tag_str.split(","):
                        tag = tag.strip()
                        if tag in TASTE_TAGS:
                            new_tags.append(tag)
            elif line.startswith("confidence:"):
                conf = line.replace("confidence:", "").strip()
                if conf in ("high", "medium", "low"):
                    confidence = conf

        return recipe_id, new_tags, confidence

    except Exception as e:
        print(f"  Error classifying {recipe_id}: {e}", flush=True)
        return recipe_id, [], "error"


async def process_batch(
    client: httpx.AsyncClient,
    batch: list[tuple],
) -> list[tuple[str, list[str], str]]:
    """Process a batch of recipes concurrently.

    Args:
        client: Async HTTP client
        batch: List of (recipe_id, title, ingredients, tags) tuples

    Returns:
        List of (recipe_id, taste_tags, confidence) tuples
    """
    tasks = [
        classify_recipe_async(client, recipe_id, title, ingredients, tags)
        for recipe_id, title, ingredients, tags in batch
    ]
    return await asyncio.gather(*tasks)


async def main_async(num_workers: int = 8):
    """Run taste classification on all recipes with parallel processing."""
    db_path = Path("data/sqlite/recipes.db")
    progress_file = Path("data/classification_progress.json")

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return

    # Load progress if resuming
    processed_ids = set()
    if progress_file.exists():
        processed_ids = set(json.loads(progress_file.read_text()))
        print(f"Resuming from {len(processed_ids)} previously processed recipes", flush=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get all recipes to classify
    cursor.execute("""
        SELECT recipe_id, title, ingredients_raw, tags
        FROM recipes
        WHERE tags IS NOT NULL
    """)

    # Build list of recipes to process
    recipes_to_process = []
    skipped_existing = 0

    for recipe_id, title, ingredients_json, tags_json in cursor:
        if recipe_id in processed_ids:
            continue

        tags = json.loads(tags_json) if tags_json else []
        ingredients = json.loads(ingredients_json) if ingredients_json else []

        # Skip if already has taste tags
        if any(t in TASTE_TAGS for t in tags):
            processed_ids.add(recipe_id)
            skipped_existing += 1
            continue

        recipes_to_process.append((recipe_id, title, ingredients, tags))

    print(f"Found {len(recipes_to_process)} recipes to classify", flush=True)
    print(f"Skipped {skipped_existing} recipes (already have taste tags)", flush=True)
    print(f"Using {num_workers} parallel workers", flush=True)
    print("Press Ctrl+C to stop (progress will be saved)\n", flush=True)

    total = 0
    updated = 0
    medium_confidence = 0
    batch_updates = []
    start_time = time.time()

    try:
        async with httpx.AsyncClient() as client:
            # Process in batches
            for i in range(0, len(recipes_to_process), num_workers):
                batch = recipes_to_process[i:i + num_workers]
                results = await process_batch(client, batch)

                for recipe_id, new_taste_tags, confidence in results:
                    total += 1

                    if confidence == "error":
                        continue

                    # Find original tags for this recipe
                    original_tags = None
                    for rid, _, _, tags in batch:
                        if rid == recipe_id:
                            original_tags = tags
                            break

                    if original_tags is None:
                        continue

                    # Only accept HIGH confidence classifications
                    if new_taste_tags and confidence == "high":
                        updated_tags = original_tags + new_taste_tags
                        batch_updates.append((json.dumps(updated_tags), recipe_id))
                        updated += 1
                    elif new_taste_tags and confidence == "medium":
                        medium_confidence += 1

                    processed_ids.add(recipe_id)

                # Progress logging every 100 recipes
                if total % 100 < num_workers:
                    elapsed = time.time() - start_time
                    rate = total / elapsed if elapsed > 0 else 0
                    remaining = len(recipes_to_process) - total
                    eta_seconds = remaining / rate if rate > 0 else 0
                    eta_hours = eta_seconds / 3600

                    print(
                        f"Processed {total}/{len(recipes_to_process)} recipes, "
                        f"{updated} updated, {medium_confidence} medium | "
                        f"{rate:.1f} recipes/sec | ETA: {eta_hours:.1f}h",
                        flush=True,
                    )

                # Batch save every 500 recipes
                if total % 500 < num_workers:
                    if batch_updates:
                        cursor.executemany(
                            "UPDATE recipes SET tags = ? WHERE recipe_id = ?",
                            batch_updates,
                        )
                        conn.commit()
                        batch_updates = []
                    progress_file.write_text(json.dumps(list(processed_ids)))
                    print(f"  Progress saved ({len(processed_ids)} total processed)", flush=True)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving progress...", flush=True)
    except httpx.ConnectError:
        print("\nError: Cannot connect to Ollama. Is it running?", flush=True)
        print("Start it with: ollama serve", flush=True)

    # Final batch
    if batch_updates:
        cursor.executemany(
            "UPDATE recipes SET tags = ? WHERE recipe_id = ?",
            batch_updates,
        )
        conn.commit()

    progress_file.write_text(json.dumps(list(processed_ids)))
    conn.close()

    elapsed = time.time() - start_time
    rate = total / elapsed if elapsed > 0 else 0

    print(f"\nDone!", flush=True)
    print(f"  Total processed: {total}", flush=True)
    print(f"  Updated with taste tags: {updated}", flush=True)
    print(f"  Skipped (already had tags): {skipped_existing}", flush=True)
    print(f"  Medium confidence (not applied): {medium_confidence}", flush=True)
    print(f"  Time elapsed: {elapsed/60:.1f} minutes", flush=True)
    print(f"  Average rate: {rate:.1f} recipes/sec", flush=True)
    print(f"  Progress saved to: {progress_file}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Classify recipe taste tags in parallel")
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    args = parser.parse_args()

    asyncio.run(main_async(num_workers=args.workers))


if __name__ == "__main__":
    main()

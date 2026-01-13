"""LLM-based taste classification for recipes.

This script uses Ollama/Qwen to classify recipes with missing taste tags:
- light: Light, refreshing meals (salads, simple proteins, low-cal)
- hearty: Filling, substantial meals (stews, casseroles, meat dishes)
- mild: Not spicy, family-friendly
- rich: Creamy, indulgent, decadent

Usage:
    # Start Ollama first
    ollama serve

    # Run classification (can be interrupted and resumed)
    python scripts/classify_taste_tags.py

    # Check progress
    cat data/classification_progress.json | python -c "import sys,json; print(len(json.load(sys.stdin)))"
"""

import json
import sqlite3
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


def classify_recipe(title: str, ingredients: list, tags: list) -> tuple[list[str], str]:
    """Use LLM to classify recipe taste profile.

    Args:
        title: Recipe title
        ingredients: List of ingredients
        tags: Existing tags on the recipe

    Returns:
        Tuple of (taste_tags, confidence_level)
    """
    prompt = CLASSIFICATION_PROMPT.format(
        title=title,
        ingredients=", ".join(ingredients[:10]),
        tags=", ".join(tags[:10]),
    )

    response = httpx.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=30.0,
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

    return new_tags, confidence


def main():
    """Run taste classification on all recipes."""
    db_path = Path("data/sqlite/recipes.db")
    progress_file = Path("data/classification_progress.json")

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return

    # Load progress if resuming
    processed_ids = set()
    if progress_file.exists():
        processed_ids = set(json.loads(progress_file.read_text()))
        print(f"Resuming from {len(processed_ids)} previously processed recipes")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get recipes to classify (skip already processed)
    cursor.execute("""
        SELECT recipe_id, title, ingredients_raw, tags
        FROM recipes
        WHERE tags IS NOT NULL
    """)

    total = 0
    updated = 0
    skipped_existing = 0
    medium_confidence = 0
    batch_updates = []

    print("Starting taste classification...")
    print("Press Ctrl+C to stop (progress will be saved)\n")

    try:
        for recipe_id, title, ingredients_json, tags_json in cursor:
            if recipe_id in processed_ids:
                continue

            total += 1
            tags = json.loads(tags_json) if tags_json else []
            ingredients = json.loads(ingredients_json) if ingredients_json else []

            # Skip if already has taste tags
            if any(t in TASTE_TAGS for t in tags):
                processed_ids.add(recipe_id)
                skipped_existing += 1
                continue

            try:
                new_taste_tags, confidence = classify_recipe(title, ingredients, tags)

                # Only accept HIGH confidence classifications
                if new_taste_tags and confidence == "high":
                    updated_tags = tags + new_taste_tags
                    batch_updates.append((json.dumps(updated_tags), recipe_id))
                    updated += 1
                elif new_taste_tags and confidence == "medium":
                    # Log medium confidence for potential review
                    print(f"  [MEDIUM] {title}: {new_taste_tags}")
                    medium_confidence += 1

                processed_ids.add(recipe_id)

                # Progress logging
                if total % 100 == 0:
                    print(
                        f"Processed {total} recipes, {updated} updated, "
                        f"{medium_confidence} medium confidence"
                    )

                # Batch save every 500 recipes
                if total % 500 == 0:
                    cursor.executemany(
                        "UPDATE recipes SET tags = ? WHERE recipe_id = ?",
                        batch_updates,
                    )
                    conn.commit()
                    batch_updates = []
                    progress_file.write_text(json.dumps(list(processed_ids)))
                    print(f"  Progress saved ({len(processed_ids)} total processed)")

            except httpx.ConnectError:
                print("\nError: Cannot connect to Ollama. Is it running?")
                print("Start it with: ollama serve")
                break
            except Exception as e:
                print(f"Error classifying {recipe_id}: {e}")
                continue

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving progress...")

    # Final batch
    if batch_updates:
        cursor.executemany(
            "UPDATE recipes SET tags = ? WHERE recipe_id = ?",
            batch_updates,
        )
        conn.commit()

    progress_file.write_text(json.dumps(list(processed_ids)))
    conn.close()

    print(f"\nDone!")
    print(f"  Total processed: {total}")
    print(f"  Updated with taste tags: {updated}")
    print(f"  Skipped (already had tags): {skipped_existing}")
    print(f"  Medium confidence (not applied): {medium_confidence}")
    print(f"  Progress saved to: {progress_file}")


if __name__ == "__main__":
    main()

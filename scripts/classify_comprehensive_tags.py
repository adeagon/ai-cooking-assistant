"""LLM-based recipe classification for TASTE and OCCASION tags.

This script uses Ollama with Qwen3 to classify recipes with taste and occasion tags.
Uses the chat API with thinking disabled for fast inference (~0.4 recipes/sec).
Dietary tags (vegetarian, vegan) are handled by apply_ingredient_rules.py instead.

TASTE PROFILE (pick 1-2):
- sweet: Sweet flavors (desserts, fruit-based, honey, sugar)
- savory: Savory/umami flavors (not sweet)
- spicy: Hot, spicy (chili, pepper, hot sauce)
- mild: Not spicy, gentle flavors
- rich: Rich, indulgent (butter, cream, cheese, fatty)
- light: Light, refreshing (salads, simple, low-calorie)

OCCASION/USE CASE (pick 2-3):
- kid-friendly: Appeals to children
- comfort-food: Comforting, satisfying, nostalgic
- weeknight: Quick/easy for busy weeknights
- dinner-party: Suitable for entertaining
- holiday-event: Special occasion/holiday
- inexpensive: Budget-friendly
- for-1-or-2: Small portions
- for-large-groups: Serves many people
- one-dish-meal: Complete meal in one dish

Uses asyncio for parallel processing.

Usage:
    # Start Ollama first
    ollama serve

    # Run classification (can be interrupted and resumed)
    python scripts/classify_comprehensive_tags.py

    # Run with different concurrency
    python scripts/classify_comprehensive_tags.py --workers 4

    # Test on 10 samples first
    python scripts/classify_comprehensive_tags.py --test
"""

import argparse
import asyncio
import json
import re
import sqlite3
import time
from pathlib import Path

import httpx

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
MODEL = "qwen3:14b"
# Disable thinking mode for faster inference (Qwen3 generates ~900 hidden tokens otherwise)
DISABLE_THINKING = True

# Tags we classify with LLM (taste + occasion)
# Dietary tags are handled by apply_ingredient_rules.py
CLASSIFIABLE_TAGS = {
    "taste": ["sweet", "savory", "spicy", "mild", "rich", "light"],
    "occasion": ["kid-friendly", "comfort-food", "weeknight", "dinner-party",
                 "holiday-event", "inexpensive", "for-1-or-2", "for-large-groups",
                 "one-dish-meal"],
}

# Tags to check for skip logic (if recipe has these, skip LLM)
TASTE_TAGS = {"sweet", "savory", "spicy", "mild", "rich", "light"}
OCCASION_TAGS = {"kid-friendly", "comfort-food", "weeknight", "dinner-party",
                 "holiday-event", "inexpensive", "for-1-or-2", "for-large-groups",
                 "one-dish-meal"}

# Cuisine tags - classified separately for recipes missing cuisine
CUISINE_TAGS = {
    "american", "southern-united-states", "southwestern-united-states", "cajun", "tex-mex",
    "mexican", "caribbean", "cuban", "brazilian",
    "italian", "french", "spanish", "greek", "mediterranean",
    "german", "british", "irish", "eastern-european", "russian", "polish", "scandinavian",
    "chinese", "japanese", "korean", "thai", "vietnamese", "indonesian",
    "indian", "middle-eastern", "lebanese", "turkish", "moroccan",
    "african", "ethiopian"
}

# Flatten for quick lookup
ALL_CLASSIFIABLE = set()
for category_tags in CLASSIFIABLE_TAGS.values():
    ALL_CLASSIFIABLE.update(category_tags)

# Post-processing rules
MUTUAL_EXCLUSIONS = [
    ("spicy", "mild"),      # Can't be both spicy and mild
    ("rich", "light"),      # Can't be both rich and light
]

IMPLICATIONS = [
    ("vegan", "vegetarian"),  # Vegan implies vegetarian
]


def apply_tag_rules(tags: list[str]) -> list[str]:
    """Apply mutual exclusion and implication rules to clean up tags."""
    tags_set = set(tags)

    # Apply mutual exclusions (keep first one found in original list)
    for tag_a, tag_b in MUTUAL_EXCLUSIONS:
        if tag_a in tags_set and tag_b in tags_set:
            tags_set.remove(tag_b)

    # Apply implications (add missing implied tags)
    for tag_a, tag_b in IMPLICATIONS:
        if tag_a in tags_set and tag_b not in tags_set:
            tags_set.add(tag_b)

    return list(tags_set)


# Cuisine classification prompt (used separately for recipes without cuisine tags)
CUISINE_PROMPT = """What cuisine is this recipe? Pick the single most specific cuisine that fits.

Recipe: {title}
Ingredients: {ingredients}

CUISINES (pick exactly one):
- american, southern-united-states, cajun, tex-mex, southwestern-united-states
- mexican, caribbean, cuban, brazilian
- italian, french, spanish, greek, mediterranean
- german, british, irish, eastern-european, russian, polish, scandinavian
- chinese, japanese, korean, thai, vietnamese, indonesian
- indian, middle-eastern, lebanese, turkish, moroccan
- african, ethiopian

Guidelines:
- Pick the most SPECIFIC cuisine (e.g., "italian" not "mediterranean" for pasta dishes)
- If the recipe is generic with no clear cultural origin, reply: CUISINE: american
- Look for key ingredients: soy sauce/ginger=asian, cumin/cilantro=mexican, olive oil/feta=greek, etc.

Reply format:
CUISINE: <single cuisine>
CONFIDENCE: <high/medium/low>
"""


async def classify_cuisine(client, recipe_id, title, ingredients):
    """Classify cuisine for a single recipe."""
    try:
        prompt = CUISINE_PROMPT.format(
            title=title,
            ingredients=", ".join(ingredients[:12]),
        )
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if DISABLE_THINKING:
            payload["think"] = False

        response = await client.post(
            OLLAMA_CHAT_URL,
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()
        result = response.json()["message"]["content"].strip()

        cuisine = None
        confidence = "low"

        for line in result.split("\n"):
            line = line.strip().lower()
            if line.startswith("cuisine:"):
                c = line.replace("cuisine:", "").strip()
                if c in CUISINE_TAGS:
                    cuisine = c
            elif line.startswith("confidence:"):
                conf = line.replace("confidence:", "").strip()
                if conf in ("high", "medium", "low"):
                    confidence = conf

        return (recipe_id, cuisine, confidence)

    except Exception as e:
        return (recipe_id, None, "error")


CLASSIFICATION_PROMPT = """Classify this recipe's TASTE and OCCASION. Be selective and precise.

Recipe: {title}
Ingredients: {ingredients}

TASTE (pick 1-2 dominant flavors):
- sweet: Desserts, baked goods, or dishes where sweetness dominates (NOT savory dishes with a hint of sugar)
- savory: Main dishes, sides, appetizers that are not sweet
- spicy: Contains chili peppers, hot sauce, cayenne, jalapeño (NOT black pepper, cinnamon, or warm baking spices)
- mild: Gentle flavors, not spicy or bold
- rich: Heavy, indulgent (butter, cream, cheese, fried)
- light: Fresh, low-fat, salads, steamed vegetables

OCCASION (pick 2-3 best fits):
- weeknight: Quick, easy, everyday cooking
- comfort-food: Hearty, nostalgic, satisfying
- kid-friendly: Appeals to children, not too complex
- dinner-party: Impressive enough to serve guests (NOT sandwiches, basic salads, or casual snacks)
- holiday-event: Special occasions, celebrations
- inexpensive: Budget-friendly ingredients
- for-1-or-2: Small portions, single servings
- for-large-groups: Feeds a crowd, potluck-friendly
- one-dish-meal: Complete meal in one dish

Reply format:
TAGS: <3-5 comma-separated tags>
CONFIDENCE: <high/medium/low>
"""


async def classify_one(client, recipe_id, title, ingredients):
    """Classify a single recipe."""
    try:
        prompt = CLASSIFICATION_PROMPT.format(
            title=title,
            ingredients=", ".join(ingredients[:15]),
        )
        payload = {
            "model": MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        if DISABLE_THINKING:
            payload["think"] = False

        response = await client.post(
            OLLAMA_CHAT_URL,
            json=payload,
            timeout=60.0,
        )
        response.raise_for_status()
        result = response.json()["message"]["content"].strip()

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
                        # Only accept tags we know about
                        if tag in ALL_CLASSIFIABLE:
                            new_tags.append(tag)
            elif line.startswith("confidence:"):
                conf = line.replace("confidence:", "").strip()
                if conf in ("high", "medium", "low"):
                    confidence = conf

        # Apply post-processing rules
        new_tags = apply_tag_rules(new_tags)

        return (recipe_id, new_tags, confidence)

    except Exception as e:
        return (recipe_id, [], "error")


async def process_batch(client, batch):
    """Process a batch of recipes in parallel."""
    tasks = [
        classify_one(client, recipe_id, title, ingredients)
        for recipe_id, title, ingredients, _ in batch
    ]
    return await asyncio.gather(*tasks)


async def main(num_workers=8):
    """Main classification loop."""
    db_path = Path("data/sqlite/recipes.db")
    progress_file = Path("data/comprehensive_classification_progress.json")

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

    # Get all recipes
    cursor.execute("""
        SELECT recipe_id, title, ingredients_raw, tags
        FROM recipes
        WHERE tags IS NOT NULL
    """)

    # Build list of recipes to process
    # Each entry: (recipe_id, title, ingredients, tags, needs_taste_occasion, needs_cuisine)
    recipes_to_process = []
    skipped_complete = 0
    needs_taste_occasion_count = 0
    needs_cuisine_count = 0

    for recipe_id, title, ingredients_json, tags_json in cursor:
        if recipe_id in processed_ids:
            continue

        tags = json.loads(tags_json) if tags_json else []
        ingredients = json.loads(ingredients_json) if ingredients_json else []

        # Check what tags this recipe already has
        has_taste = any(t in TASTE_TAGS for t in tags)
        has_occasion = any(t in OCCASION_TAGS for t in tags)
        has_cuisine = any(t in CUISINE_TAGS for t in tags)

        needs_taste_occasion = not (has_taste and has_occasion)
        needs_cuisine = not has_cuisine

        # Skip if recipe already has everything
        if not needs_taste_occasion and not needs_cuisine:
            processed_ids.add(recipe_id)
            skipped_complete += 1
            continue

        if needs_taste_occasion:
            needs_taste_occasion_count += 1
        if needs_cuisine:
            needs_cuisine_count += 1

        recipes_to_process.append((recipe_id, title, ingredients, tags, needs_taste_occasion, needs_cuisine))

    print(f"Found {len(recipes_to_process)} recipes to classify", flush=True)
    print(f"Skipped {skipped_complete} recipes (already complete)", flush=True)
    print(f"  - {needs_taste_occasion_count} need taste/occasion tags", flush=True)
    print(f"  - {needs_cuisine_count} need cuisine tags", flush=True)
    print(f"Using {num_workers} parallel workers", flush=True)
    print("Press Ctrl+C to stop (progress will be saved)\n", flush=True)

    total = 0
    updated = 0
    cuisine_updated = 0
    medium_confidence = 0
    batch_updates = []
    start_time = time.time()

    try:
        async with httpx.AsyncClient() as client:
            # Process in batches
            for i in range(0, len(recipes_to_process), num_workers):
                batch = recipes_to_process[i:i + num_workers]

                # Process taste/occasion for recipes that need it
                taste_occasion_batch = [
                    (rid, title, ingredients, tags)
                    for rid, title, ingredients, tags, needs_to, needs_c in batch
                    if needs_to
                ]
                taste_occasion_results = {}
                if taste_occasion_batch:
                    results = await process_batch(client, taste_occasion_batch)
                    for recipe_id, new_tags, confidence in results:
                        taste_occasion_results[recipe_id] = (new_tags, confidence)

                # Process cuisine for recipes that need it
                cuisine_batch = [
                    (rid, title, ingredients)
                    for rid, title, ingredients, tags, needs_to, needs_c in batch
                    if needs_c
                ]
                cuisine_results = {}
                if cuisine_batch:
                    cuisine_tasks = [
                        classify_cuisine(client, rid, title, ingredients)
                        for rid, title, ingredients in cuisine_batch
                    ]
                    results = await asyncio.gather(*cuisine_tasks)
                    for recipe_id, cuisine, confidence in results:
                        cuisine_results[recipe_id] = (cuisine, confidence)

                # Combine results and update database
                for recipe_id, title, ingredients, original_tags, needs_to, needs_c in batch:
                    total += 1
                    all_new_tags = []

                    # Add taste/occasion tags
                    if needs_to and recipe_id in taste_occasion_results:
                        new_tags, confidence = taste_occasion_results[recipe_id]
                        if new_tags and confidence in ("high", "medium"):
                            all_new_tags.extend(new_tags)
                            if confidence == "medium":
                                medium_confidence += 1

                    # Add cuisine tag
                    if needs_c and recipe_id in cuisine_results:
                        cuisine, confidence = cuisine_results[recipe_id]
                        if cuisine and confidence in ("high", "medium"):
                            all_new_tags.append(cuisine)
                            cuisine_updated += 1

                    # Update database if we have new tags
                    if all_new_tags:
                        existing_set = set(original_tags)
                        new_unique = [t for t in all_new_tags if t not in existing_set]

                        if new_unique:
                            updated_tags = original_tags + new_unique
                            batch_updates.append((json.dumps(updated_tags), recipe_id))
                            updated += 1

                    processed_ids.add(recipe_id)

                    # Progress logging every 100 recipes
                    if total % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = total / elapsed if elapsed > 0 else 0
                        remaining = len(recipes_to_process) - total
                        eta_seconds = remaining / rate if rate > 0 else 0
                        eta_hours = eta_seconds / 3600

                        print(
                            f"Processed {total}/{len(recipes_to_process)} recipes, "
                            f"{updated} updated ({cuisine_updated} cuisine) | "
                            f"{rate:.1f} recipes/sec | ETA: {eta_hours:.1f}h",
                            flush=True
                        )

                    # Batch save every 500 recipes
                    if total % 500 == 0:
                        if batch_updates:
                            cursor.executemany(
                                "UPDATE recipes SET tags = ? WHERE recipe_id = ?",
                                batch_updates
                            )
                            conn.commit()
                            batch_updates = []

                        progress_file.write_text(json.dumps(list(processed_ids)))
                        print(f"  Progress saved ({len(processed_ids)} total processed)", flush=True)

        # Final save
        if batch_updates:
            cursor.executemany(
                "UPDATE recipes SET tags = ? WHERE recipe_id = ?",
                batch_updates
            )
            conn.commit()

        progress_file.write_text(json.dumps(list(processed_ids)))

        elapsed = time.time() - start_time
        print(f"\nDone! Processed {total} recipes in {elapsed/60:.1f} minutes", flush=True)
        print(f"Updated {updated} recipes with new tags ({updated/total*100:.1f}%)", flush=True)
        print(f"Medium confidence: {medium_confidence} ({medium_confidence/total*100:.1f}%)", flush=True)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving progress...", flush=True)

        if batch_updates:
            cursor.executemany(
                "UPDATE recipes SET tags = ? WHERE recipe_id = ?",
                batch_updates
            )
            conn.commit()

        progress_file.write_text(json.dumps(list(processed_ids)))
        print(f"Progress saved ({len(processed_ids)} recipes processed)", flush=True)

    finally:
        conn.close()


async def test_on_samples(num_samples: int = 10):
    """Test classification on a few sample recipes."""
    db_path = Path("data/sqlite/recipes.db")

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get ALL recipes and filter to those needing classification
    cursor.execute("""
        SELECT recipe_id, title, ingredients_raw, tags
        FROM recipes
        WHERE tags IS NOT NULL AND ingredients_raw IS NOT NULL
        ORDER BY RANDOM()
    """)

    # Filter to those missing taste/occasion OR cuisine, collect exactly num_samples
    samples = []
    for recipe_id, title, ingredients_json, tags_json in cursor:
        tags = json.loads(tags_json) if tags_json else []
        ingredients = json.loads(ingredients_json) if ingredients_json else []

        has_taste = any(t in TASTE_TAGS for t in tags)
        has_occasion = any(t in OCCASION_TAGS for t in tags)
        has_cuisine = any(t in CUISINE_TAGS for t in tags)

        needs_taste_occasion = not (has_taste and has_occasion)
        needs_cuisine = not has_cuisine

        if needs_taste_occasion or needs_cuisine:
            samples.append((recipe_id, title, ingredients, tags, needs_taste_occasion, needs_cuisine))
            if len(samples) >= num_samples:
                break

    conn.close()

    if not samples:
        print("No recipes found needing classification")
        return

    print(f"\n{'='*70}")
    print(f"TESTING LLM CLASSIFICATION ON {len(samples)} SAMPLES")
    print(f"{'='*70}\n")

    async with httpx.AsyncClient() as client:
        for recipe_id, title, ingredients, existing_tags, needs_to, needs_c in samples:
            print(f"\nRecipe: {title[:60]}")
            print(f"Ingredients: {', '.join(ingredients[:6])}...")

            existing_taste_occasion = [t for t in existing_tags if t in TASTE_TAGS or t in OCCASION_TAGS]
            existing_cuisine = [t for t in existing_tags if t in CUISINE_TAGS]
            print(f"Existing taste/occasion: {existing_taste_occasion}")
            print(f"Existing cuisine: {existing_cuisine}")

            # Taste/occasion classification
            if needs_to:
                _, new_tags, confidence = await classify_one(client, recipe_id, title, ingredients)
                print(f"Taste/Occasion predicted: {new_tags} ({confidence})")

                # Check for conflicts
                if "spicy" in new_tags and "mild" in new_tags:
                    print("  WARNING: spicy + mild conflict")
                if "rich" in new_tags and "light" in new_tags:
                    print("  WARNING: rich + light conflict")
            else:
                print("Taste/Occasion: SKIPPED (already has tags)")

            # Cuisine classification
            if needs_c:
                _, cuisine, conf = await classify_cuisine(client, recipe_id, title, ingredients)
                print(f"Cuisine predicted: {cuisine} ({conf})")
            else:
                print("Cuisine: SKIPPED (already has tags)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM recipe classification for taste/occasion")
    parser.add_argument("--workers", type=int, default=4, help="Number of parallel workers")
    parser.add_argument("--test", type=int, nargs="?", const=10, help="Test on N samples (default 10)")
    parser.add_argument("--model", type=str, default=None, help=f"Ollama model to use (default: {MODEL})")
    parser.add_argument("--think", action="store_true", help="Enable thinking mode (slower but may improve quality)")
    args = parser.parse_args()

    # Override globals based on args
    if args.model:
        MODEL = args.model
    if args.think:
        DISABLE_THINKING = False

    print(f"Using model: {MODEL}, thinking: {'enabled' if not DISABLE_THINKING else 'disabled'}")

    if args.test:
        asyncio.run(test_on_samples(num_samples=args.test))
    else:
        asyncio.run(main(num_workers=args.workers))

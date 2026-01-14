"""Apply ingredient-based rules for vegetarian/vegan classification.

This script uses deterministic ingredient matching to add missing dietary tags.
It's faster and more accurate than LLM classification for these specific tags.

Rules:
- Vegetarian: No actual meat or seafood (broth/stock is OK)
- Vegan: No meat, seafood, dairy, or eggs

Usage:
    python scripts/apply_ingredient_rules.py           # Process all recipes
    python scripts/apply_ingredient_rules.py --test   # Test on 50 samples only
"""

import argparse
import json
import sqlite3
from pathlib import Path

# Actual meat keywords (NOT broth) - disqualifies vegetarian
MEAT_KEYWORDS = {
    "chicken", "beef", "pork", "bacon", "ham", "turkey", "lamb",
    "sausage", "steak", "ground beef", "ground pork", "pepperoni",
    "salami", "prosciutto", "duck", "veal", "venison", "chorizo",
    "ribs", "brisket", "meatball", "meatloaf", "hot dog", "bratwurst",
    "kielbasa", "pancetta", "guanciale", "lard", "fatback"
}

# Seafood - disqualifies vegetarian
SEAFOOD_KEYWORDS = {
    "fish", "salmon", "tuna", "shrimp", "crab", "lobster", "clam",
    "mussel", "oyster", "scallop", "cod", "tilapia", "halibut",
    "anchovy", "sardine", "calamari", "squid", "octopus", "prawn",
    "crawfish", "crayfish", "mahi", "snapper", "trout", "bass",
    "swordfish", "mackerel", "haddock", "catfish"
}

# These indicate a broth/stock ingredient
BROTH_KEYWORDS = {"broth", "stock", "bouillon", "base"}

# Animal-based broths - these are NOT vegetarian
ANIMAL_BROTH_KEYWORDS = {"chicken", "beef", "pork", "turkey", "ham", "bone", "fish", "seafood", "lamb", "veal"}

# Dairy and eggs - disqualifies vegan (but not vegetarian)
DAIRY_EGG_KEYWORDS = {
    "egg", "eggs", "milk", "cheese", "butter", "cream", "yogurt",
    "sour cream", "cream cheese", "parmesan", "cheddar", "mozzarella",
    "feta", "ricotta", "half-and-half", "whipping cream", "heavy cream",
    "buttermilk", "ghee", "mayonnaise", "mayo", "custard", "ice cream",
    "whey", "casein", "lactose"
}


def is_plant_based_broth(ingredient: str) -> bool:
    """Check if a broth/stock ingredient is plant-based (vegetable, mushroom, etc.)."""
    ing_lower = ingredient.lower()

    # Must contain broth/stock keyword to be considered a broth
    if not any(broth in ing_lower for broth in BROTH_KEYWORDS):
        return False

    # Check if it's animal-based - if so, NOT plant-based
    return not any(animal in ing_lower for animal in ANIMAL_BROTH_KEYWORDS)


def has_actual_meat(ingredients: list[str]) -> bool:
    """Check for actual meat (only plant-based broth/stock is OK for vegetarian)."""
    for ing in ingredients:
        ing_lower = ing.lower()

        # If it's a plant-based broth (vegetable, mushroom, etc.), skip it
        if is_plant_based_broth(ing):
            continue

        # Check for meat keywords
        for meat in MEAT_KEYWORDS:
            if meat in ing_lower:
                return True

    return False


def has_seafood(ingredients: list[str]) -> bool:
    """Check for seafood ingredients."""
    ingredients_text = " ".join(ingredients).lower()
    return any(sf in ingredients_text for sf in SEAFOOD_KEYWORDS)


def has_dairy_or_eggs(ingredients: list[str]) -> bool:
    """Check for dairy or egg ingredients."""
    ingredients_text = " ".join(ingredients).lower()
    return any(de in ingredients_text for de in DAIRY_EGG_KEYWORDS)


def apply_dietary_rules(ingredients: list[str], existing_tags: list[str]) -> list[str]:
    """
    Determine which dietary tags to add based on ingredients.

    Returns list of new tags to add (empty if none needed).
    """
    new_tags = []

    # Skip if already tagged
    has_vegetarian = "vegetarian" in existing_tags
    has_vegan = "vegan" in existing_tags

    # Check ingredients
    contains_meat = has_actual_meat(ingredients)
    contains_seafood = has_seafood(ingredients)
    contains_dairy_eggs = has_dairy_or_eggs(ingredients)

    # Apply rules
    if not contains_meat and not contains_seafood:
        if not has_vegetarian:
            new_tags.append("vegetarian")

        if not contains_dairy_eggs and not has_vegan:
            new_tags.append("vegan")

    return new_tags


def test_on_samples(cursor, num_samples: int = 50):
    """Test ingredient rules on random samples and report accuracy."""
    print(f"\n{'='*60}")
    print(f"TESTING INGREDIENT RULES ON {num_samples} SAMPLES")
    print(f"{'='*60}\n")

    # Get samples that have existing vegetarian/vegan tags for validation
    cursor.execute("""
        SELECT recipe_id, title, ingredients_raw, tags
        FROM recipes
        WHERE (tags LIKE '%"vegetarian"%' OR tags LIKE '%"vegan"%')
        ORDER BY RANDOM()
        LIMIT ?
    """, (num_samples // 2,))

    tagged_samples = cursor.fetchall()

    # Get samples without tags
    cursor.execute("""
        SELECT recipe_id, title, ingredients_raw, tags
        FROM recipes
        WHERE tags NOT LIKE '%"vegetarian"%' AND tags NOT LIKE '%"vegan"%'
        ORDER BY RANDOM()
        LIMIT ?
    """, (num_samples // 2,))

    untagged_samples = cursor.fetchall()

    all_samples = tagged_samples + untagged_samples

    correct = 0
    incorrect = 0
    new_tags_added = 0

    print("Sample Results:")
    print("-" * 60)

    for recipe_id, title, ingredients_json, tags_json in all_samples[:20]:  # Show first 20
        ingredients = json.loads(ingredients_json) if ingredients_json else []
        tags = json.loads(tags_json) if tags_json else []

        # What we would predict
        new_tags = apply_dietary_rules(ingredients, [])  # Ignore existing for prediction

        # Check against existing tags
        existing_veg = "vegetarian" in tags
        predicted_veg = "vegetarian" in new_tags

        # For validation: if recipe has vegetarian tag, our rules should agree
        # (unless it's a false positive in the dataset)
        match = "?" if not existing_veg else ("OK" if predicted_veg else "MISS")

        print(f"\n{title[:50]}")
        print(f"  Ingredients: {', '.join(ingredients[:5])}...")
        print(f"  Existing tags: vegetarian={existing_veg}, vegan={'vegan' in tags}")
        print(f"  Predicted: {new_tags if new_tags else 'none'}")
        print(f"  Status: {match}")

        if existing_veg:
            if predicted_veg:
                correct += 1
            else:
                incorrect += 1
        else:
            if new_tags:
                new_tags_added += 1

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Recipes with existing vegetarian tag: {correct + incorrect}")
    print(f"  Correctly identified: {correct}")
    print(f"  Missed (false negatives): {incorrect}")
    if correct + incorrect > 0:
        print(f"  Recall: {correct / (correct + incorrect) * 100:.1f}%")
    print(f"\nRecipes without vegetarian tag:")
    print(f"  Would add vegetarian: {new_tags_added}")


def process_all_recipes(cursor, conn, dry_run: bool = False):
    """Process all recipes and add missing dietary tags."""
    print(f"\n{'='*60}")
    print("PROCESSING ALL RECIPES")
    print(f"{'='*60}\n")

    # Get all recipes
    cursor.execute("""
        SELECT recipe_id, ingredients_raw, tags
        FROM recipes
        WHERE ingredients_raw IS NOT NULL
    """)

    total = 0
    updated = 0
    batch_updates = []

    for recipe_id, ingredients_json, tags_json in cursor:
        total += 1

        ingredients = json.loads(ingredients_json) if ingredients_json else []
        tags = json.loads(tags_json) if tags_json else []

        new_tags = apply_dietary_rules(ingredients, tags)

        if new_tags:
            updated_tags = tags + new_tags
            batch_updates.append((json.dumps(updated_tags), recipe_id))
            updated += 1

        if total % 10000 == 0:
            print(f"Processed {total} recipes, {updated} would be updated...", flush=True)

    print(f"\nTotal recipes: {total}")
    print(f"Would update: {updated} ({updated/total*100:.1f}%)")

    if not dry_run and batch_updates:
        print(f"\nApplying {len(batch_updates)} updates to database...")
        cursor.executemany(
            "UPDATE recipes SET tags = ? WHERE recipe_id = ?",
            batch_updates
        )
        conn.commit()
        print("Done!")
    elif dry_run:
        print("\n(Dry run - no changes made)")


def clear_dietary_tags(cursor, conn):
    """Remove existing vegetarian/vegan tags added by this script."""
    print(f"\n{'='*60}")
    print("CLEARING EXISTING VEGETARIAN/VEGAN TAGS")
    print(f"{'='*60}\n")

    # Get all recipes with vegetarian or vegan tags
    cursor.execute("""
        SELECT recipe_id, tags
        FROM recipes
        WHERE tags LIKE '%"vegetarian"%' OR tags LIKE '%"vegan"%'
    """)

    batch_updates = []
    for recipe_id, tags_json in cursor:
        tags = json.loads(tags_json) if tags_json else []
        # Remove vegetarian and vegan tags
        new_tags = [t for t in tags if t.lower() not in ("vegetarian", "vegan")]
        if len(new_tags) != len(tags):
            batch_updates.append((json.dumps(new_tags), recipe_id))

    print(f"Found {len(batch_updates)} recipes with vegetarian/vegan tags to clear")

    if batch_updates:
        cursor.executemany(
            "UPDATE recipes SET tags = ? WHERE recipe_id = ?",
            batch_updates
        )
        conn.commit()
        print("Tags cleared successfully!")


def main():
    parser = argparse.ArgumentParser(description="Apply ingredient-based dietary rules")
    parser.add_argument("--test", action="store_true", help="Test on 50 samples only")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually update database")
    parser.add_argument("--reset", action="store_true", help="Clear existing vegetarian/vegan tags before re-applying")
    args = parser.parse_args()

    db_path = Path("data/sqlite/recipes.db")

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        if args.test:
            test_on_samples(cursor)
        else:
            if args.reset:
                clear_dietary_tags(cursor, conn)
            process_all_recipes(cursor, conn, dry_run=args.dry_run)
    finally:
        conn.close()


if __name__ == "__main__":
    main()

"""Apply nutrition-based rules to add dietary tags.

This script uses actual nutrition data (PDV = Percent Daily Value) to add tags:
- low-calorie: < 300% PDV (~600 cal per serving)
- low-fat: < 15% PDV
- low-sodium: < 20% PDV
- low-carb: < 15% PDV
- high-protein: > 50% PDV
- low-saturated-fat: < 25% PDV

Thresholds are based on analysis of existing Food.com tags.

Usage:
    python scripts/apply_nutrition_rules.py
    python scripts/apply_nutrition_rules.py --test        # Test on 50 samples
    python scripts/apply_nutrition_rules.py --dry-run     # Show stats without changes
    python scripts/apply_nutrition_rules.py --reset       # Clear and re-apply tags
"""

import argparse
import json
import sqlite3
from pathlib import Path

DB_PATH = Path("data/sqlite/recipes.db")

# Thresholds based on analysis of existing Food.com tags
# Using conservative thresholds (around P50-P75 of tagged recipes)
NUTRITION_RULES = {
    "low-calorie": ("calories_pdv", "<", 300),      # ~600 cal per serving
    "low-fat": ("total_fat_pdv", "<", 15),          # <15% daily value
    "low-sodium": ("sodium_pdv", "<", 20),          # <20% daily value
    "low-carb": ("carbs_pdv", "<", 15),             # <15% daily value
    "high-protein": ("protein_pdv", ">", 50),       # >50% daily value
    "low-saturated-fat": ("saturated_fat_pdv", "<", 25),  # <25% daily value
}

# Tags that this script manages (for --reset)
MANAGED_TAGS = set(NUTRITION_RULES.keys())


def get_recipes_needing_tags(cursor: sqlite3.Cursor) -> list[tuple]:
    """Get recipes that might need nutrition tags."""
    cursor.execute("""
        SELECT recipe_id, tags,
               calories_pdv, total_fat_pdv, sugar_pdv,
               sodium_pdv, protein_pdv, saturated_fat_pdv, carbs_pdv
        FROM recipes
        WHERE calories_pdv IS NOT NULL
    """)
    return cursor.fetchall()


def calculate_nutrition_tags(
    calories: float,
    fat: float,
    sugar: float,
    sodium: float,
    protein: float,
    sat_fat: float,
    carbs: float,
) -> list[str]:
    """Calculate which nutrition tags apply based on values."""
    tags = []

    if calories is not None and calories < 300:
        tags.append("low-calorie")
    if fat is not None and fat < 15:
        tags.append("low-fat")
    if sodium is not None and sodium < 20:
        tags.append("low-sodium")
    if carbs is not None and carbs < 15:
        tags.append("low-carb")
    if protein is not None and protein > 50:
        tags.append("high-protein")
    if sat_fat is not None and sat_fat < 25:
        tags.append("low-saturated-fat")

    return tags


def clear_nutrition_tags(cursor: sqlite3.Cursor, conn: sqlite3.Connection):
    """Remove nutrition tags added by this script."""
    print("Clearing existing nutrition tags...")

    cursor.execute("SELECT recipe_id, tags FROM recipes WHERE tags IS NOT NULL")
    updates = []

    for recipe_id, tags_json in cursor:
        tags = json.loads(tags_json) if tags_json else []
        original_len = len(tags)
        tags = [t for t in tags if t not in MANAGED_TAGS]

        if len(tags) != original_len:
            updates.append((json.dumps(tags), recipe_id))

    if updates:
        cursor.executemany("UPDATE recipes SET tags = ? WHERE recipe_id = ?", updates)
        conn.commit()
        print(f"  Cleared nutrition tags from {len(updates):,} recipes")
    else:
        print("  No nutrition tags to clear")


def apply_nutrition_rules(dry_run: bool = False, test_mode: bool = False, reset: bool = False):
    """Apply nutrition-based tagging rules."""
    print("=" * 60)
    print("APPLYING NUTRITION-BASED TAGGING RULES")
    print("=" * 60)

    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Clear existing tags if reset
        if reset:
            clear_nutrition_tags(cursor, conn)

        # Get recipes
        print("\n1. Loading recipes with nutrition data...")
        recipes = get_recipes_needing_tags(cursor)
        print(f"   Found {len(recipes):,} recipes with nutrition data")

        if test_mode:
            import random
            recipes = random.sample(recipes, min(50, len(recipes)))
            print(f"   TEST MODE: Using {len(recipes)} samples")

        # Process recipes
        print("\n2. Calculating nutrition tags...")
        stats = {tag: 0 for tag in NUTRITION_RULES}
        updates = []
        new_tags_added = 0

        for row in recipes:
            recipe_id = row[0]
            existing_tags = json.loads(row[1]) if row[1] else []
            calories, fat, sugar, sodium, protein, sat_fat, carbs = row[2:9]

            # Calculate new tags
            new_tags = calculate_nutrition_tags(calories, fat, sugar, sodium, protein, sat_fat, carbs)

            # Count stats
            for tag in new_tags:
                stats[tag] += 1

            # Find tags to add (not already present)
            tags_to_add = [t for t in new_tags if t not in existing_tags]

            if tags_to_add:
                updated_tags = existing_tags + tags_to_add
                updates.append((json.dumps(updated_tags), recipe_id))
                new_tags_added += len(tags_to_add)

        # Show stats
        print("\n   Tag distribution (recipes qualifying):")
        for tag, count in sorted(stats.items(), key=lambda x: -x[1]):
            pct = count / len(recipes) * 100
            print(f"     {tag:<20}: {count:>6,} ({pct:>5.1f}%)")

        print(f"\n   Recipes needing updates: {len(updates):,}")
        print(f"   New tags to add: {new_tags_added:,}")

        # Apply updates
        if dry_run:
            print("\n   DRY RUN - no changes made")
        elif updates:
            print("\n3. Applying updates...")
            cursor.executemany("UPDATE recipes SET tags = ? WHERE recipe_id = ?", updates)
            conn.commit()
            print(f"   Updated {len(updates):,} recipes")

        # Show sample results in test mode
        if test_mode and updates:
            print("\n=== SAMPLE RESULTS ===")
            for tags_json, recipe_id in updates[:5]:
                cursor.execute("SELECT title FROM recipes WHERE recipe_id = ?", (recipe_id,))
                title = cursor.fetchone()[0]
                new_tags = [t for t in json.loads(tags_json) if t in MANAGED_TAGS]
                print(f"  {title[:40]:<40} -> {new_tags}")

    finally:
        conn.close()

    print("\nDone!")


def show_comparison():
    """Compare our rules vs existing Food.com tags."""
    print("\n=== COMPARISON: Our Rules vs Food.com Tags ===")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for tag, (col, op, threshold) in NUTRITION_RULES.items():
        # Count recipes with existing tag
        cursor.execute("SELECT COUNT(*) FROM recipes WHERE tags LIKE ?", (f'%{tag}%',))
        existing_count = cursor.fetchone()[0]

        # Count recipes that would qualify under our rules
        if op == "<":
            cursor.execute(f"SELECT COUNT(*) FROM recipes WHERE {col} < ? AND {col} IS NOT NULL", (threshold,))
        else:
            cursor.execute(f"SELECT COUNT(*) FROM recipes WHERE {col} > ? AND {col} IS NOT NULL", (threshold,))
        qualifying_count = cursor.fetchone()[0]

        print(f"  {tag:<20}: Existing={existing_count:>6,}  Qualifying={qualifying_count:>6,}  Diff={qualifying_count - existing_count:>+6,}")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply nutrition-based tagging rules")
    parser.add_argument("--dry-run", action="store_true", help="Show stats without making changes")
    parser.add_argument("--test", action="store_true", help="Test on 50 random samples")
    parser.add_argument("--reset", action="store_true", help="Clear and re-apply nutrition tags")
    parser.add_argument("--compare", action="store_true", help="Compare our rules vs existing tags")
    args = parser.parse_args()

    if args.compare:
        show_comparison()
    else:
        apply_nutrition_rules(dry_run=args.dry_run, test_mode=args.test, reset=args.reset)

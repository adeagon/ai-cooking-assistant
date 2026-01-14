"""Add nutrition data from RAW_recipes.csv to SQLite database.

This script:
1. Adds nutrition columns to the recipes table
2. Loads nutrition data from RAW_recipes.csv
3. Updates recipes with their nutrition values

Nutrition values are Percent Daily Value (PDV):
- calories_pdv: Calories as % of 2000 cal daily value
- total_fat_pdv: Total fat as % DV
- sugar_pdv: Sugar as % DV
- sodium_pdv: Sodium as % DV
- protein_pdv: Protein as % DV
- saturated_fat_pdv: Saturated fat as % DV
- carbs_pdv: Carbohydrates as % DV

Usage:
    python scripts/add_nutrition_data.py
    python scripts/add_nutrition_data.py --dry-run
"""

import argparse
import ast
import sqlite3
from pathlib import Path

import pandas as pd

DB_PATH = Path("data/sqlite/recipes.db")
RAW_CSV_PATH = Path("data/raw/RAW_recipes.csv")

NUTRITION_COLUMNS = [
    ("calories_pdv", "REAL"),
    ("total_fat_pdv", "REAL"),
    ("sugar_pdv", "REAL"),
    ("sodium_pdv", "REAL"),
    ("protein_pdv", "REAL"),
    ("saturated_fat_pdv", "REAL"),
    ("carbs_pdv", "REAL"),
]


def add_columns_if_not_exist(cursor: sqlite3.Cursor) -> list[str]:
    """Add nutrition columns to recipes table if they don't exist."""
    # Get existing columns
    cursor.execute("PRAGMA table_info(recipes)")
    existing_cols = {row[1] for row in cursor.fetchall()}

    added = []
    for col_name, col_type in NUTRITION_COLUMNS:
        if col_name not in existing_cols:
            cursor.execute(f"ALTER TABLE recipes ADD COLUMN {col_name} {col_type}")
            added.append(col_name)
            print(f"  Added column: {col_name} ({col_type})")

    return added


def load_nutrition_from_csv() -> dict[str, list[float]]:
    """Load nutrition data from RAW_recipes.csv.

    Returns:
        Dict mapping recipe_id to [calories, fat, sugar, sodium, protein, sat_fat, carbs]
    """
    print(f"Loading nutrition data from {RAW_CSV_PATH}...")

    df = pd.read_csv(RAW_CSV_PATH, usecols=["id", "nutrition"])

    nutrition_map = {}
    errors = 0

    for _, row in df.iterrows():
        recipe_id = str(row["id"])
        try:
            nutrition = ast.literal_eval(row["nutrition"])
            if len(nutrition) == 7:
                nutrition_map[recipe_id] = nutrition
            else:
                errors += 1
        except (ValueError, SyntaxError):
            errors += 1

    print(f"  Loaded {len(nutrition_map):,} recipes with nutrition data")
    if errors:
        print(f"  Skipped {errors} recipes with invalid nutrition format")

    return nutrition_map


def update_recipes(cursor: sqlite3.Cursor, nutrition_map: dict[str, list[float]], dry_run: bool) -> int:
    """Update recipes with nutrition data."""
    # Get all recipe IDs in our database
    cursor.execute("SELECT recipe_id FROM recipes")
    db_recipe_ids = {row[0] for row in cursor.fetchall()}

    # Find matches
    matched_ids = db_recipe_ids & set(nutrition_map.keys())
    print(f"  Matched {len(matched_ids):,} of {len(db_recipe_ids):,} recipes in database")

    if dry_run:
        print("  DRY RUN - no changes made")
        return len(matched_ids)

    # Batch update
    updates = []
    for recipe_id in matched_ids:
        nutr = nutrition_map[recipe_id]
        updates.append((
            nutr[0],  # calories
            nutr[1],  # total_fat
            nutr[2],  # sugar
            nutr[3],  # sodium
            nutr[4],  # protein
            nutr[5],  # saturated_fat
            nutr[6],  # carbs
            recipe_id,
        ))

    cursor.executemany("""
        UPDATE recipes SET
            calories_pdv = ?,
            total_fat_pdv = ?,
            sugar_pdv = ?,
            sodium_pdv = ?,
            protein_pdv = ?,
            saturated_fat_pdv = ?,
            carbs_pdv = ?
        WHERE recipe_id = ?
    """, updates)

    return len(updates)


def show_stats(cursor: sqlite3.Cursor):
    """Show nutrition statistics after update."""
    print("\n=== NUTRITION STATISTICS ===")

    for col in ["calories_pdv", "total_fat_pdv", "sugar_pdv", "sodium_pdv", "protein_pdv", "saturated_fat_pdv", "carbs_pdv"]:
        cursor.execute(f"""
            SELECT
                COUNT(*) as count,
                AVG({col}) as avg,
                MIN({col}) as min,
                MAX({col}) as max
            FROM recipes
            WHERE {col} IS NOT NULL
        """)
        row = cursor.fetchone()
        count, avg, min_val, max_val = row
        col_short = col.replace("_pdv", "")
        print(f"  {col_short:<12}: count={count:,}, avg={avg:.1f}%, min={min_val:.1f}%, max={max_val:.1f}%")


def main(dry_run: bool = False):
    print("=" * 60)
    print("ADDING NUTRITION DATA TO DATABASE")
    print("=" * 60)

    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}")
        return

    if not RAW_CSV_PATH.exists():
        print(f"Error: Raw CSV not found at {RAW_CSV_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        # Step 1: Add columns
        print("\n1. Adding nutrition columns to schema...")
        added = add_columns_if_not_exist(cursor)
        if not added:
            print("  All columns already exist")
        conn.commit()

        # Step 2: Load nutrition data
        print("\n2. Loading nutrition data from CSV...")
        nutrition_map = load_nutrition_from_csv()

        # Step 3: Update recipes
        print("\n3. Updating recipes with nutrition data...")
        updated = update_recipes(cursor, nutrition_map, dry_run)
        print(f"  Updated {updated:,} recipes")

        if not dry_run:
            conn.commit()
            print("\n  Changes committed to database")

            # Show stats
            show_stats(cursor)

    finally:
        conn.close()

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add nutrition data to recipes database")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without making changes")
    args = parser.parse_args()

    main(dry_run=args.dry_run)

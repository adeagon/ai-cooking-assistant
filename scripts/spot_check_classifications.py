"""Spot-check script to validate taste classifications.

This script displays a random sample of recipes with taste tags
for manual verification. Target accuracy: >85%.

Usage:
    python scripts/spot_check_classifications.py
"""

import json
import sqlite3
from pathlib import Path

TASTE_TAGS = ["light", "hearty", "mild", "rich"]


def main():
    """Run spot check validation on taste classifications."""
    db_path = Path("data/sqlite/recipes.db")

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get 50 random recipes that have new taste tags
    cursor.execute("""
        SELECT title, ingredients_raw, tags FROM recipes
        WHERE tags LIKE '%"light"%' OR tags LIKE '%"hearty"%'
           OR tags LIKE '%"mild"%' OR tags LIKE '%"rich"%'
        ORDER BY RANDOM() LIMIT 50
    """)

    results = cursor.fetchall()
    conn.close()

    if not results:
        print("No recipes found with taste tags. Run classify_taste_tags.py first.")
        return

    print("=== SPOT CHECK: Review these recipes ===")
    print("For each recipe, evaluate if the taste tags are correct.")
    print("Enter: y=correct, n=incorrect, s=skip, q=quit\n")

    correct = 0
    incorrect = 0
    skipped = 0
    reviewed = 0

    for i, (title, ingredients_json, tags_json) in enumerate(results, 1):
        ingredients = json.loads(ingredients_json) if ingredients_json else []
        tags = json.loads(tags_json) if tags_json else []
        taste_tags = [t for t in tags if t in TASTE_TAGS]

        if not taste_tags:
            continue

        print(f"\n--- Recipe {i}/{len(results)} ---")
        print(f"Title: {title}")
        print(f"Ingredients: {', '.join(ingredients[:8])}")
        print(f"Taste tags: {taste_tags}")

        while True:
            response = input("Correct? (y/n/s/q): ").strip().lower()
            if response in ("y", "n", "s", "q"):
                break
            print("Please enter y, n, s, or q")

        if response == "q":
            print("\nQuitting early...")
            break
        elif response == "y":
            correct += 1
            reviewed += 1
        elif response == "n":
            incorrect += 1
            reviewed += 1
        elif response == "s":
            skipped += 1

    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"Reviewed: {reviewed}")
    print(f"Correct: {correct}")
    print(f"Incorrect: {incorrect}")
    print(f"Skipped: {skipped}")

    if reviewed > 0:
        accuracy = correct / reviewed * 100
        print(f"\nAccuracy: {accuracy:.1f}%")

        if accuracy >= 85:
            print("Status: PASS (target: >85%)")
        else:
            print("Status: FAIL (target: >85%)")
            print("Consider reviewing classification prompt or confidence threshold.")
    else:
        print("\nNo recipes reviewed.")


if __name__ == "__main__":
    main()

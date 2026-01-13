"""Benchmark different models for taste classification."""

import json
import sqlite3
import time
from pathlib import Path

import httpx

OLLAMA_URL = "http://localhost:11434/api/generate"
TASTE_TAGS = ["light", "hearty", "mild", "rich"]

CLASSIFICATION_PROMPT = """Classify this recipe's taste profile.

Definitions:
- light: Light, refreshing (salads, simple proteins, low-calorie)
- hearty: Filling, substantial (stews, casseroles, meat-heavy)
- mild: Not spicy, family-friendly flavors
- rich: Creamy, indulgent, decadent (butter, cream, cheese)

Recipe: {title}
Ingredients: {ingredients}

Reply in this exact format:
TAGS: <comma-separated tags or "none">
CONFIDENCE: <high/medium/low>
"""


def classify_recipe(model: str, title: str, ingredients: list) -> tuple[list[str], str, float]:
    """Classify a recipe and return tags, confidence, and time taken."""
    prompt = CLASSIFICATION_PROMPT.format(
        title=title,
        ingredients=", ".join(ingredients[:10]),
    )

    start = time.time()
    response = httpx.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=60.0,
    )
    elapsed = time.time() - start
    response.raise_for_status()

    result = response.json()["response"].strip()

    # Parse response
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

    return new_tags, confidence, elapsed


def main():
    db_path = Path("data/sqlite/recipes.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get 20 random recipes
    cursor.execute("""
        SELECT recipe_id, title, ingredients_raw
        FROM recipes
        WHERE tags IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 20
    """)

    recipes = []
    for recipe_id, title, ingredients_json in cursor:
        ingredients = json.loads(ingredients_json) if ingredients_json else []
        recipes.append((recipe_id, title, ingredients))

    conn.close()

    models = ["qwen2.5:14b", "qwen2.5:7b"]
    results = {model: {"times": [], "high": 0, "medium": 0, "low": 0, "tags": []} for model in models}

    print(f"Benchmarking {len(recipes)} recipes across {len(models)} models\n")
    print("=" * 70)

    for i, (recipe_id, title, ingredients) in enumerate(recipes, 1):
        print(f"\n[{i}/20] {title[:50]}...")

        for model in models:
            tags, conf, elapsed = classify_recipe(model, title, ingredients)
            results[model]["times"].append(elapsed)
            results[model][conf] += 1
            results[model]["tags"].append((tags, conf))

            print(f"  {model:15} -> {tags} ({conf}) [{elapsed:.1f}s]")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for model in models:
        r = results[model]
        avg_time = sum(r["times"]) / len(r["times"])
        total_time = sum(r["times"])
        high_rate = r["high"] / 20 * 100

        print(f"\n{model}:")
        print(f"  Avg time/recipe: {avg_time:.2f}s")
        print(f"  Total time:      {total_time:.1f}s")
        print(f"  High confidence: {r['high']}/20 ({high_rate:.0f}%)")
        print(f"  Medium:          {r['medium']}/20")
        print(f"  Low:             {r['low']}/20")

    # Agreement check
    print("\n" + "-" * 70)
    print("AGREEMENT (same tags between models):")
    agree = 0
    for i in range(20):
        tags_14b = set(results["qwen2.5:14b"]["tags"][i][0])
        tags_7b = set(results["qwen2.5:7b"]["tags"][i][0])
        if tags_14b == tags_7b:
            agree += 1
    print(f"  {agree}/20 recipes ({agree/20*100:.0f}%) have identical tags")

    # Speed comparison
    speedup = sum(results["qwen2.5:14b"]["times"]) / sum(results["qwen2.5:7b"]["times"])
    print(f"\n7b is {speedup:.1f}x faster than 14b")


if __name__ == "__main__":
    main()

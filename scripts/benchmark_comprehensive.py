"""Benchmark comprehensive classification with different models and worker counts."""

import asyncio
import json
import sqlite3
import time
from pathlib import Path

import httpx

OLLAMA_URL = "http://localhost:11434/api/generate"

# Models to test
MODELS = ["qwen2.5:7b", "qwen2.5:14b", "llama3.3:70b"]

# Worker counts to test
WORKER_COUNTS = [2, 4, 8, 12]

# All classifiable tags
CLASSIFIABLE_TAGS = {
    "taste": ["sweet", "savory", "spicy", "mild", "rich", "light"],
    "dietary": ["vegetarian", "vegan", "gluten-free", "low-carb", "low-sodium",
                "low-calorie", "low-fat", "high-protein", "healthy"],
    "difficulty": ["easy", "beginner-cook"],
    "occasion": ["kid-friendly", "comfort-food", "weeknight", "dinner-party",
                 "holiday-event", "inexpensive", "for-1-or-2", "for-large-groups",
                 "one-dish-meal"],
}

ALL_CLASSIFIABLE = set()
for category_tags in CLASSIFIABLE_TAGS.values():
    ALL_CLASSIFIABLE.update(category_tags)

CLASSIFICATION_PROMPT = """Classify this recipe with ALL applicable tags.

Recipe: {title}
Ingredients: {ingredients}

TASTE PROFILE (select all that apply):
- sweet, savory, spicy, mild, rich, light

DIETARY (select all that apply):
- vegetarian, vegan, gluten-free, low-carb, low-sodium, low-calorie, low-fat, high-protein, healthy

DIFFICULTY (select one if obvious):
- easy, beginner-cook

OCCASION/USE (select all that apply):
- kid-friendly, comfort-food, weeknight, dinner-party, holiday-event, inexpensive, for-1-or-2, for-large-groups, one-dish-meal

Reply in this exact format (comma-separated tags, or "none" if uncertain):
TAGS: <tags>
CONFIDENCE: <high/medium/low>

Example: "TAGS: savory, mild, vegetarian, healthy, easy, weeknight\nCONFIDENCE: high"
"""


async def classify_one(client, model, title, ingredients):
    """Classify a single recipe."""
    prompt = CLASSIFICATION_PROMPT.format(
        title=title,
        ingredients=", ".join(ingredients[:15]),
    )
    try:
        response = await client.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120.0,
        )
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
                        if tag in ALL_CLASSIFIABLE:
                            new_tags.append(tag)
            elif line.startswith("confidence:"):
                conf = line.replace("confidence:", "").strip()
                if conf in ("high", "medium", "low"):
                    confidence = conf

        return (new_tags, confidence, result)
    except Exception as e:
        return ([], "error", str(e))


async def benchmark_model_workers(model, recipes, num_workers):
    """Benchmark a model with specific worker count."""
    start = time.time()
    results = []

    async with httpx.AsyncClient() as client:
        for i in range(0, len(recipes), num_workers):
            batch = recipes[i:i + num_workers]
            tasks = [classify_one(client, model, title, ing) for _, title, ing in batch]
            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

    elapsed = time.time() - start
    rate = len(recipes) / elapsed

    # Analyze results
    high_conf = sum(1 for _, conf, _ in results if conf == "high")
    medium_conf = sum(1 for _, conf, _ in results if conf == "medium")
    low_conf = sum(1 for _, conf, _ in results if conf == "low")
    errors = sum(1 for _, conf, _ in results if conf == "error")

    # Count average tags per recipe (high confidence only)
    high_conf_tags = [tags for tags, conf, _ in results if conf == "high"]
    avg_tags = sum(len(t) for t in high_conf_tags) / len(high_conf_tags) if high_conf_tags else 0

    return {
        "model": model,
        "workers": num_workers,
        "elapsed": elapsed,
        "rate": rate,
        "high_conf": high_conf,
        "medium_conf": medium_conf,
        "low_conf": low_conf,
        "errors": errors,
        "avg_tags": avg_tags,
        "sample_results": results[:5]  # Keep first 5 for quality review
    }


async def main():
    db_path = Path("data/sqlite/recipes.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get 50 random recipes for benchmarking
    cursor.execute("""
        SELECT recipe_id, title, ingredients_raw
        FROM recipes
        WHERE tags IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 50
    """)

    recipes = []
    for recipe_id, title, ingredients_json in cursor:
        ingredients = json.loads(ingredients_json) if ingredients_json else []
        recipes.append((recipe_id, title, ingredients))
    conn.close()

    print(f"Benchmarking {len(recipes)} recipes with comprehensive classification")
    print("=" * 70)

    all_results = []

    for model in MODELS:
        print(f"\n{'=' * 70}")
        print(f"MODEL: {model}")
        print("=" * 70)

        for workers in WORKER_COUNTS:
            print(f"\nTesting {workers} workers...", end=" ", flush=True)
            try:
                result = await benchmark_model_workers(model, recipes, workers)
                all_results.append(result)

                high_pct = result["high_conf"] / len(recipes) * 100
                print(
                    f"{result['rate']:.2f} recipes/sec | "
                    f"High: {result['high_conf']} ({high_pct:.0f}%) | "
                    f"Avg tags: {result['avg_tags']:.1f}"
                )
            except Exception as e:
                print(f"ERROR: {e}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Model':<15} {'Workers':<8} {'Rate':<12} {'High%':<10} {'Avg Tags':<10} {'ETA (h)':<10}")
    print("-" * 70)

    for r in all_results:
        high_pct = r["high_conf"] / len(recipes) * 100
        eta_hours = 86899 / r["rate"] / 3600  # Approximate total recipes
        print(
            f"{r['model']:<15} {r['workers']:<8} {r['rate']:<12.2f} "
            f"{high_pct:<10.0f} {r['avg_tags']:<10.1f} {eta_hours:<10.1f}"
        )

    # Best by rate vs quality
    if all_results:
        best_rate = max(all_results, key=lambda x: x["rate"])
        best_quality = max(all_results, key=lambda x: x["high_conf"] / len(recipes) * x["avg_tags"])

        print("\n" + "=" * 70)
        print(f"BEST SPEED: {best_rate['model']} with {best_rate['workers']} workers")
        print(f"  Rate: {best_rate['rate']:.2f} recipes/sec")
        print(f"  High confidence: {best_rate['high_conf']}/{len(recipes)}")

        print(f"\nBEST QUALITY: {best_quality['model']} with {best_quality['workers']} workers")
        print(f"  High confidence: {best_quality['high_conf']}/{len(recipes)}")
        print(f"  Avg tags (high conf): {best_quality['avg_tags']:.1f}")

    # Show sample classifications from best quality model
    if all_results:
        print("\n" + "=" * 70)
        print("SAMPLE CLASSIFICATIONS (from best quality model):")
        print("=" * 70)

        for i, (recipe_id, title, ingredients) in enumerate(recipes[:5]):
            tags, conf, raw = best_quality["sample_results"][i]
            print(f"\nRecipe: {title[:50]}")
            print(f"Ingredients: {', '.join(ingredients[:5])}")
            print(f"Tags: {', '.join(tags) if tags else 'none'}")
            print(f"Confidence: {conf}")


if __name__ == "__main__":
    asyncio.run(main())

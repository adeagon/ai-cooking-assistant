"""Benchmark different worker counts for parallel classification."""

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

Reply in this exact format:
TAGS: <comma-separated tags or "none">
CONFIDENCE: <high/medium/low>
"""


async def classify_one(client, title, ingredients):
    prompt = CLASSIFICATION_PROMPT.format(
        title=title,
        ingredients=", ".join(ingredients[:10]),
    )
    response = await client.post(
        OLLAMA_URL,
        json={"model": MODEL, "prompt": prompt, "stream": False},
        timeout=60.0,
    )
    return response.json()["response"]


async def benchmark_workers(recipes, num_workers):
    """Benchmark with specific worker count."""
    start = time.time()

    async with httpx.AsyncClient() as client:
        for i in range(0, len(recipes), num_workers):
            batch = recipes[i:i + num_workers]
            tasks = [classify_one(client, title, ing) for _, title, ing in batch]
            await asyncio.gather(*tasks)

    elapsed = time.time() - start
    rate = len(recipes) / elapsed
    return elapsed, rate


async def main():
    db_path = Path("data/sqlite/recipes.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get 100 random recipes for benchmarking
    cursor.execute("""
        SELECT recipe_id, title, ingredients_raw
        FROM recipes
        WHERE tags IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 100
    """)

    recipes = []
    for recipe_id, title, ingredients_json in cursor:
        ingredients = json.loads(ingredients_json) if ingredients_json else []
        recipes.append((recipe_id, title, ingredients))
    conn.close()

    print(f"Benchmarking {len(recipes)} recipes with different worker counts\n")
    print("=" * 50)

    worker_counts = [4, 8, 12, 16, 24]
    results = []

    for workers in worker_counts:
        print(f"\nTesting {workers} workers...", end=" ", flush=True)
        elapsed, rate = await benchmark_workers(recipes, workers)
        results.append((workers, elapsed, rate))
        print(f"{rate:.2f} recipes/sec ({elapsed:.1f}s total)")

    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"{'Workers':<10} {'Rate (r/s)':<12} {'Time (s)':<10} {'ETA (h)':<10}")
    print("-" * 50)

    for workers, elapsed, rate in results:
        eta_hours = 87000 / rate / 3600  # ~87K remaining recipes
        print(f"{workers:<10} {rate:<12.2f} {elapsed:<10.1f} {eta_hours:<10.1f}")

    best = max(results, key=lambda x: x[2])
    print(f"\nOptimal: {best[0]} workers ({best[2]:.2f} recipes/sec)")


if __name__ == "__main__":
    asyncio.run(main())

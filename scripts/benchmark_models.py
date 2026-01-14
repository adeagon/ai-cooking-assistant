"""Benchmark LLM models for recipe classification.

Compares qwen2.5:14b, qwen3:14b, and qwen3:8b across different parallelism levels.

Usage:
    python scripts/benchmark_models.py
    python scripts/benchmark_models.py --samples 50
    python scripts/benchmark_models.py --models qwen3:14b qwen3:8b
"""

import argparse
import asyncio
import json
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

OLLAMA_URL = "http://localhost:11434/api/generate"

# Tags we classify
TASTE_TAGS = {"sweet", "savory", "spicy", "mild", "rich", "light"}
OCCASION_TAGS = {"kid-friendly", "comfort-food", "weeknight", "dinner-party",
                 "holiday-event", "inexpensive", "for-1-or-2", "for-large-groups",
                 "one-dish-meal"}
ALL_CLASSIFIABLE = TASTE_TAGS | OCCASION_TAGS

CLASSIFICATION_PROMPT = """Classify this recipe's TASTE and OCCASION. Be selective and precise.

Recipe: {title}
Ingredients: {ingredients}

TASTE (pick 1-2 dominant flavors):
- sweet: Desserts, baked goods, or dishes where sweetness dominates
- savory: Main dishes, sides, appetizers that are not sweet
- spicy: Contains chili peppers, hot sauce, cayenne, jalapeno
- mild: Gentle flavors, not spicy or bold
- rich: Heavy, indulgent (butter, cream, cheese, fried)
- light: Fresh, low-fat, salads, steamed vegetables

OCCASION (pick 2-3 best fits):
- weeknight: Quick, easy, everyday cooking
- comfort-food: Hearty, nostalgic, satisfying
- kid-friendly: Appeals to children, not too complex
- dinner-party: Impressive enough to serve guests
- holiday-event: Special occasions, celebrations
- inexpensive: Budget-friendly ingredients
- for-1-or-2: Small portions, single servings
- for-large-groups: Feeds a crowd, potluck-friendly
- one-dish-meal: Complete meal in one dish

Reply format:
TAGS: <3-5 comma-separated tags>
CONFIDENCE: <high/medium/low>
"""


@dataclass
class BenchmarkResult:
    """Results from a single benchmark run."""
    model: str
    workers: int
    num_recipes: int
    total_time: float
    successful: int
    failed: int
    avg_time_per_recipe: float
    recipes_per_second: float
    classifications: list = field(default_factory=list)


def parse_response(response_text: str) -> tuple[list[str], str]:
    """Parse LLM response to extract tags and confidence."""
    tags = []
    confidence = "low"

    for line in response_text.split("\n"):
        line = line.strip().lower()
        if line.startswith("tags:"):
            tag_str = line.replace("tags:", "").strip()
            if tag_str != "none":
                for tag in tag_str.split(","):
                    tag = tag.strip()
                    if tag in ALL_CLASSIFIABLE:
                        tags.append(tag)
        elif line.startswith("confidence:"):
            conf = line.replace("confidence:", "").strip()
            if conf in ("high", "medium", "low"):
                confidence = conf

    return tags, confidence


async def classify_one(client: httpx.AsyncClient, model: str, recipe_id: str,
                       title: str, ingredients: list[str]) -> dict:
    """Classify a single recipe."""
    start_time = time.time()

    try:
        prompt = CLASSIFICATION_PROMPT.format(
            title=title,
            ingredients=", ".join(ingredients[:15]),
        )
        response = await client.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120.0,
        )
        response.raise_for_status()
        result = response.json()
        response_text = result.get("response", "").strip()

        tags, confidence = parse_response(response_text)
        elapsed = time.time() - start_time

        # Get token counts from response if available
        eval_count = result.get("eval_count", 0)
        prompt_eval_count = result.get("prompt_eval_count", 0)

        return {
            "recipe_id": recipe_id,
            "title": title,
            "tags": tags,
            "confidence": confidence,
            "success": True,
            "time": elapsed,
            "output_tokens": eval_count,
            "input_tokens": prompt_eval_count,
            "raw_response": response_text[:200],
        }

    except Exception as e:
        elapsed = time.time() - start_time
        return {
            "recipe_id": recipe_id,
            "title": title,
            "tags": [],
            "confidence": "error",
            "success": False,
            "time": elapsed,
            "error": str(e),
        }


async def run_benchmark(model: str, recipes: list[tuple], workers: int) -> BenchmarkResult:
    """Run benchmark for a single model/worker configuration."""
    print(f"\n{'='*60}")
    print(f"BENCHMARKING: {model} with {workers} workers")
    print(f"{'='*60}")

    results = []
    start_time = time.time()

    async with httpx.AsyncClient() as client:
        # Process in batches
        for i in range(0, len(recipes), workers):
            batch = recipes[i:i + workers]

            tasks = [
                classify_one(client, model, recipe_id, title, ingredients)
                for recipe_id, title, ingredients in batch
            ]

            batch_results = await asyncio.gather(*tasks)
            results.extend(batch_results)

            # Progress
            done = len(results)
            elapsed = time.time() - start_time
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  Progress: {done}/{len(recipes)} ({rate:.2f} recipes/sec)", end="\r")

    total_time = time.time() - start_time
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    print()  # New line after progress

    return BenchmarkResult(
        model=model,
        workers=workers,
        num_recipes=len(recipes),
        total_time=total_time,
        successful=successful,
        failed=failed,
        avg_time_per_recipe=total_time / len(recipes) if recipes else 0,
        recipes_per_second=len(recipes) / total_time if total_time > 0 else 0,
        classifications=results,
    )


def get_sample_recipes(db_path: Path, num_samples: int) -> list[tuple]:
    """Get random sample recipes for benchmarking."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT recipe_id, title, ingredients_raw
        FROM recipes
        WHERE ingredients_raw IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
    """, (num_samples,))

    recipes = []
    for recipe_id, title, ingredients_json in cursor:
        ingredients = json.loads(ingredients_json) if ingredients_json else []
        recipes.append((recipe_id, title, ingredients))

    conn.close()
    return recipes


def print_summary(all_results: list[BenchmarkResult]):
    """Print summary comparison of all benchmark results."""
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)

    # Group by model
    models = {}
    for r in all_results:
        if r.model not in models:
            models[r.model] = []
        models[r.model].append(r)

    # Header
    print(f"\n{'Model':<15} {'Workers':<10} {'Time(s)':<10} {'Rate(r/s)':<12} {'Success':<10} {'Failed':<8}")
    print("-" * 70)

    for model in sorted(models.keys()):
        for r in sorted(models[model], key=lambda x: x.workers):
            print(f"{r.model:<15} {r.workers:<10} {r.total_time:<10.1f} {r.recipes_per_second:<12.2f} {r.successful:<10} {r.failed:<8}")

    # Best configurations by model
    print("\n" + "-"*70)
    print("BEST CONFIGURATION PER MODEL:")
    for model in sorted(models.keys()):
        best = max(models[model], key=lambda x: x.recipes_per_second)
        print(f"  {model}: {best.workers} workers -> {best.recipes_per_second:.2f} recipes/sec")

    # Overall fastest
    fastest = max(all_results, key=lambda x: x.recipes_per_second)
    print(f"\n  OVERALL FASTEST: {fastest.model} w/{fastest.workers} workers ({fastest.recipes_per_second:.2f} recipes/sec)")

    # Quality comparison - confidence distribution
    print("\n" + "-"*70)
    print("CONFIDENCE DISTRIBUTION (at 4 workers):")
    for model in sorted(models.keys()):
        for r in models[model]:
            if r.workers == 4:
                high = sum(1 for c in r.classifications if c.get("confidence") == "high")
                med = sum(1 for c in r.classifications if c.get("confidence") == "medium")
                low = sum(1 for c in r.classifications if c.get("confidence") == "low")
                total = len(r.classifications)
                print(f"  {model}: high={high} ({high/total*100:.0f}%), medium={med} ({med/total*100:.0f}%), low={low} ({low/total*100:.0f}%)")

    # Tag statistics
    print("\n" + "-"*70)
    print("AVERAGE TAGS PER RECIPE (at 4 workers):")
    for model in sorted(models.keys()):
        for r in models[model]:
            if r.workers == 4:
                total_tags = sum(len(c.get("tags", [])) for c in r.classifications)
                avg_tags = total_tags / len(r.classifications) if r.classifications else 0
                print(f"  {model}: {avg_tags:.2f} tags/recipe")


def print_sample_comparison(all_results: list[BenchmarkResult]):
    """Print sample classifications from each model for comparison."""
    print("\n" + "="*80)
    print("SAMPLE CLASSIFICATIONS COMPARISON (first 10 recipes at 4 workers)")
    print("="*80)

    # Get models at 4 workers
    results_4w = [r for r in all_results if r.workers == 4]
    if not results_4w:
        print("No results at 4 workers")
        return

    # Get first 10 recipe IDs
    sample_ids = [c["recipe_id"] for c in results_4w[0].classifications[:10]]

    for recipe_id in sample_ids:
        # Find the title
        title = ""
        for r in results_4w:
            for c in r.classifications:
                if c["recipe_id"] == recipe_id:
                    title = c.get("title", "")[:50]
                    break
            if title:
                break

        print(f"\n{title}")
        print("-" * 60)

        for r in results_4w:
            for c in r.classifications:
                if c["recipe_id"] == recipe_id:
                    tags = c.get("tags", [])
                    conf = c.get("confidence", "")
                    time_s = c.get("time", 0)
                    print(f"  {r.model:<15}: {str(tags):<50} ({conf}) [{time_s:.1f}s]")
                    break


def save_results(all_results: list[BenchmarkResult], output_path: Path):
    """Save benchmark results to JSON."""
    data = []
    for r in all_results:
        data.append({
            "model": r.model,
            "workers": r.workers,
            "num_recipes": r.num_recipes,
            "total_time": r.total_time,
            "successful": r.successful,
            "failed": r.failed,
            "avg_time_per_recipe": r.avg_time_per_recipe,
            "recipes_per_second": r.recipes_per_second,
            "classifications": r.classifications,
        })

    output_path.write_text(json.dumps(data, indent=2))
    print(f"\nResults saved to {output_path}")


async def main(models: list[str], num_samples: int, worker_counts: list[int]):
    """Run all benchmarks."""
    db_path = Path("data/sqlite/recipes.db")

    if not db_path.exists():
        print(f"Error: Database not found at {db_path}")
        return

    # Get sample recipes (same for all tests)
    print(f"Loading {num_samples} random sample recipes...")
    recipes = get_sample_recipes(db_path, num_samples)
    print(f"Loaded {len(recipes)} recipes for benchmarking")

    all_results = []

    for model in models:
        for workers in worker_counts:
            result = await run_benchmark(model, recipes, workers)
            all_results.append(result)

            print(f"  Completed: {result.total_time:.1f}s total, "
                  f"{result.recipes_per_second:.2f} recipes/sec, "
                  f"{result.successful} success, {result.failed} failed")

    # Print summaries
    print_summary(all_results)
    print_sample_comparison(all_results)

    # Save results
    output_path = Path("data/benchmark_results.json")
    save_results(all_results, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark LLM models for recipe classification")
    parser.add_argument("--models", nargs="+",
                        default=["qwen2.5:14b", "qwen3:14b", "qwen3:8b"],
                        help="Models to benchmark")
    parser.add_argument("--samples", type=int, default=100,
                        help="Number of sample recipes")
    parser.add_argument("--workers", nargs="+", type=int, default=[2, 4, 8],
                        help="Worker counts to test")
    args = parser.parse_args()

    print("="*80)
    print("LLM MODEL BENCHMARK FOR RECIPE CLASSIFICATION")
    print("="*80)
    print(f"Models: {args.models}")
    print(f"Samples: {args.samples}")
    print(f"Worker counts: {args.workers}")

    asyncio.run(main(args.models, args.samples, args.workers))

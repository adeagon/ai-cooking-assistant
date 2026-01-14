"""Benchmark model accuracy using ingredient-based ground truth rules."""

import asyncio
import json
import sqlite3
import time
from pathlib import Path

import httpx

OLLAMA_URL = "http://localhost:11434/api/generate"

# Models to test
MODELS = ["qwen2.5:7b", "qwen2.5:14b"]

# Tags we can verify with ingredient rules
VERIFIABLE_TAGS = {
    "vegetarian", "vegan", "gluten-free", "high-protein", "spicy"
}

# Ingredient-based ground truth rules
CONTAINS_MEAT = {
    "chicken", "beef", "pork", "lamb", "turkey", "bacon", "ham", "sausage",
    "steak", "ground beef", "ground pork", "ground turkey", "prosciutto",
    "pepperoni", "salami", "chorizo", "duck", "veal", "venison", "bison"
}

CONTAINS_SEAFOOD = {
    "fish", "salmon", "tuna", "shrimp", "crab", "lobster", "clam", "mussel",
    "oyster", "scallop", "cod", "tilapia", "halibut", "anchovy", "sardine",
    "calamari", "squid", "octopus", "prawn", "crawfish"
}

CONTAINS_DAIRY_EGGS = {
    "egg", "eggs", "milk", "cheese", "butter", "cream", "yogurt", "sour cream",
    "cream cheese", "parmesan", "cheddar", "mozzarella", "feta", "ricotta",
    "half-and-half", "whipping cream", "heavy cream", "buttermilk", "ghee"
}

CONTAINS_GLUTEN = {
    "flour", "bread", "pasta", "noodle", "wheat", "barley", "rye", "couscous",
    "cracker", "breadcrumb", "panko", "tortilla", "pita", "biscuit", "cake",
    "cookie", "pastry", "croissant", "bagel", "muffin", "pizza dough",
    "soy sauce", "teriyaki", "hoisin"  # These typically contain wheat
}

HIGH_PROTEIN_INGREDIENTS = {
    "chicken", "beef", "pork", "turkey", "fish", "salmon", "tuna", "shrimp",
    "egg", "eggs", "tofu", "tempeh", "lentil", "beans", "chickpea",
    "greek yogurt", "cottage cheese", "protein powder"
}

SPICY_INGREDIENTS = {
    "jalapeno", "habanero", "serrano", "cayenne", "chili", "hot sauce",
    "sriracha", "tabasco", "red pepper flakes", "ghost pepper", "chipotle",
    "wasabi", "horseradish", "hot pepper", "thai chili", "scotch bonnet"
}


def get_ground_truth(ingredients_list: list[str]) -> dict[str, bool]:
    """Determine ground truth tags based on ingredients."""
    ingredients_lower = " ".join(ingredients_list).lower()

    has_meat = any(meat in ingredients_lower for meat in CONTAINS_MEAT)
    has_seafood = any(sf in ingredients_lower for sf in CONTAINS_SEAFOOD)
    has_dairy_eggs = any(de in ingredients_lower for de in CONTAINS_DAIRY_EGGS)
    has_gluten = any(gl in ingredients_lower for gl in CONTAINS_GLUTEN)
    has_protein = any(hp in ingredients_lower for hp in HIGH_PROTEIN_INGREDIENTS)
    has_spicy = any(sp in ingredients_lower for sp in SPICY_INGREDIENTS)

    return {
        "vegetarian": not has_meat and not has_seafood,
        "vegan": not has_meat and not has_seafood and not has_dairy_eggs,
        "gluten-free": not has_gluten,
        "high-protein": has_protein,
        "spicy": has_spicy,
    }


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

Reply in this exact format (comma-separated tags from the lists above ONLY):
TAGS: <tags>
CONFIDENCE: <high/medium/low>
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
        tags = []
        confidence = "low"

        for line in result.split("\n"):
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

        return (tags, confidence, result)
    except Exception as e:
        return ([], "error", str(e))


async def benchmark_model(model, recipes, num_workers=4):
    """Benchmark a model's accuracy."""
    start = time.time()

    results = []
    async with httpx.AsyncClient() as client:
        for i in range(0, len(recipes), num_workers):
            batch = recipes[i:i + num_workers]
            tasks = [
                classify_one(client, model, title, ingredients)
                for _, title, ingredients, _ in batch
            ]
            batch_results = await asyncio.gather(*tasks)

            for j, (tags, confidence, raw) in enumerate(batch_results):
                recipe_id, title, ingredients, ground_truth = batch[j]
                results.append({
                    "recipe_id": recipe_id,
                    "title": title,
                    "ingredients": ingredients,
                    "ground_truth": ground_truth,
                    "predicted_tags": tags,
                    "confidence": confidence,
                    "raw_response": raw
                })

    elapsed = time.time() - start

    # Calculate accuracy metrics for verifiable tags
    metrics = {tag: {"tp": 0, "fp": 0, "tn": 0, "fn": 0} for tag in VERIFIABLE_TAGS}

    for r in results:
        gt = r["ground_truth"]
        predicted = set(r["predicted_tags"])

        for tag in VERIFIABLE_TAGS:
            should_have = gt[tag]
            has_tag = tag in predicted

            if should_have and has_tag:
                metrics[tag]["tp"] += 1
            elif should_have and not has_tag:
                metrics[tag]["fn"] += 1
            elif not should_have and has_tag:
                metrics[tag]["fp"] += 1
            else:
                metrics[tag]["tn"] += 1

    return {
        "model": model,
        "elapsed": elapsed,
        "rate": len(recipes) / elapsed,
        "results": results,
        "metrics": metrics
    }


def print_metrics(benchmark_result, num_recipes):
    """Print accuracy metrics."""
    model = benchmark_result["model"]
    metrics = benchmark_result["metrics"]

    print(f"\n{'=' * 70}")
    print(f"MODEL: {model}")
    print(f"{'=' * 70}")
    print(f"Speed: {benchmark_result['rate']:.2f} recipes/sec")
    print(f"\n{'Tag':<15} {'Precision':<12} {'Recall':<12} {'Accuracy':<12} {'F1':<12}")
    print("-" * 60)

    total_correct = 0
    total_possible = 0

    for tag in VERIFIABLE_TAGS:
        m = metrics[tag]
        tp, fp, tn, fn = m["tp"], m["fp"], m["tn"], m["fn"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        accuracy = (tp + tn) / num_recipes
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        total_correct += tp + tn
        total_possible += num_recipes

        print(f"{tag:<15} {precision:<12.1%} {recall:<12.1%} {accuracy:<12.1%} {f1:<12.2f}")

    overall_accuracy = total_correct / total_possible
    print(f"\n{'OVERALL':<15} {'':<12} {'':<12} {overall_accuracy:<12.1%}")


async def main():
    db_path = Path("data/sqlite/recipes.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get 30 random recipes
    cursor.execute("""
        SELECT recipe_id, title, ingredients_raw
        FROM recipes
        WHERE tags IS NOT NULL AND ingredients_raw IS NOT NULL
        ORDER BY RANDOM()
        LIMIT 30
    """)

    recipes = []
    for recipe_id, title, ingredients_json in cursor:
        ingredients = json.loads(ingredients_json) if ingredients_json else []
        if ingredients:
            ground_truth = get_ground_truth(ingredients)
            recipes.append((recipe_id, title, ingredients, ground_truth))
    conn.close()

    print(f"Benchmarking ACCURACY on {len(recipes)} recipes")
    print("Using ingredient-based ground truth for: vegetarian, vegan, gluten-free, high-protein, spicy")

    all_results = []
    for model in MODELS:
        result = await benchmark_model(model, recipes)
        all_results.append(result)
        print_metrics(result, len(recipes))

    # Show sample errors
    print("\n" + "=" * 70)
    print("SAMPLE ERRORS (False Positives - tagged but shouldn't be)")
    print("=" * 70)

    for result in all_results:
        print(f"\n--- {result['model']} ---")
        error_count = 0
        for r in result["results"]:
            gt = r["ground_truth"]
            predicted = set(r["predicted_tags"])

            # Find false positives
            for tag in VERIFIABLE_TAGS:
                if tag in predicted and not gt[tag]:
                    if error_count < 3:
                        print(f"\nRecipe: {r['title'][:50]}")
                        print(f"Ingredients: {', '.join(r['ingredients'][:5])}...")
                        print(f"FALSE POSITIVE: '{tag}' tagged but shouldn't be")
                        error_count += 1
                    break


if __name__ == "__main__":
    asyncio.run(main())

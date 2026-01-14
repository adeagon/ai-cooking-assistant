"""Validation script for LLM recipe classification.

Runs classification on a stratified sample and measures accuracy against
ingredient-based ground truth before committing to full classification.

Usage:
    # Run full validation (500 recipes, ~20-30 min)
    python scripts/validate_classification.py

    # Quick validation (100 recipes, ~5 min)
    python scripts/validate_classification.py --quick

    # Export ground truth for manual review
    python scripts/validate_classification.py --export-csv
"""

import argparse
import asyncio
import csv
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import httpx

# ============================================================================
# Configuration
# ============================================================================

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5:14b"
DB_PATH = Path("data/sqlite/recipes.db")

# Sample sizes for stratified sampling
SAMPLE_CONFIG = {
    "full": {
        "simple": 50,
        "medium": 50,
        "complex": 50,
        "main_dish": 50,
        "dessert": 50,
        "appetizer_side": 50,
        "with_cuisine": 50,
        "without_cuisine": 50,
        "random": 100,
    },
    "quick": {
        "simple": 10,
        "medium": 10,
        "complex": 10,
        "main_dish": 10,
        "dessert": 10,
        "appetizer_side": 10,
        "with_cuisine": 10,
        "without_cuisine": 10,
        "random": 20,
    },
}

# Tag sets for classification
TASTE_TAGS = {"sweet", "savory", "spicy", "mild", "rich", "light"}
OCCASION_TAGS = {
    "kid-friendly", "comfort-food", "weeknight", "dinner-party",
    "holiday-event", "inexpensive", "for-1-or-2", "for-large-groups",
    "one-dish-meal"
}
CUISINE_TAGS = {
    "american", "southern-united-states", "southwestern-united-states", "cajun", "tex-mex",
    "mexican", "caribbean", "cuban", "brazilian",
    "italian", "french", "spanish", "greek", "mediterranean",
    "german", "british", "irish", "eastern-european", "russian", "polish", "scandinavian",
    "chinese", "japanese", "korean", "thai", "vietnamese", "indonesian",
    "indian", "middle-eastern", "lebanese", "turkish", "moroccan",
    "african", "ethiopian"
}

# Ground truth ingredient rules
SPICY_INGREDIENTS = {
    "jalapeno", "jalapeño", "habanero", "serrano", "cayenne", "chili", "chile",
    "hot sauce", "sriracha", "tabasco", "red pepper flakes", "ghost pepper",
    "chipotle", "wasabi", "hot pepper", "thai chili", "scotch bonnet",
    "gochujang", "sambal", "harissa"
}

SWEET_INDICATORS = {
    "dessert", "cookie", "cake", "pie", "candy", "brownie", "cupcake",
    "frosting", "icing", "pudding", "ice cream", "sorbet", "custard",
    "chocolate chip", "sugar cookie", "cheesecake", "fudge", "caramel"
}

RICH_INGREDIENTS = {
    "heavy cream", "whipping cream", "cream cheese", "butter", "mascarpone",
    "brie", "camembert", "bacon fat", "lard", "duck fat", "deep fried",
    "double cream", "clotted cream"
}

LIGHT_TAGS = {"low-calorie", "low-fat", "diet", "healthy", "salad", "steamed"}

# Cuisine indicator ingredients
CUISINE_INDICATORS = {
    "italian": {"parmesan", "mozzarella", "basil", "oregano", "marinara", "pasta", "prosciutto", "ricotta"},
    "mexican": {"cumin", "cilantro", "jalapeno", "tortilla", "salsa", "taco", "enchilada", "queso"},
    "chinese": {"soy sauce", "sesame oil", "ginger", "five spice", "hoisin", "oyster sauce", "wok"},
    "indian": {"curry", "turmeric", "garam masala", "cumin", "coriander", "naan", "paneer", "ghee"},
    "thai": {"fish sauce", "coconut milk", "thai basil", "lemongrass", "galangal", "pad thai"},
    "japanese": {"miso", "sake", "mirin", "nori", "wasabi", "sushi", "teriyaki", "dashi"},
    "korean": {"gochujang", "kimchi", "sesame", "korean chili", "bulgogi", "bibimbap"},
    "greek": {"feta", "olive oil", "oregano", "lemon", "tzatziki", "pita", "gyro"},
    "french": {"wine", "shallot", "dijon", "tarragon", "gruyere", "beurre", "croissant"},
}


# ============================================================================
# Ground Truth Functions
# ============================================================================

def get_ground_truth_spicy(ingredients: list[str], title: str) -> bool:
    """Determine if recipe is spicy based on ingredients."""
    text = " ".join(ingredients).lower() + " " + title.lower()
    return any(spicy in text for spicy in SPICY_INGREDIENTS)


def get_ground_truth_sweet(ingredients: list[str], title: str, tags: list[str]) -> bool:
    """Determine if recipe is sweet based on title and tags."""
    text = title.lower() + " " + " ".join(tags).lower()
    return any(sweet in text for sweet in SWEET_INDICATORS)


def get_ground_truth_rich(ingredients: list[str]) -> bool:
    """Determine if recipe is rich based on ingredients."""
    text = " ".join(ingredients).lower()
    return any(rich in text for rich in RICH_INGREDIENTS)


def get_ground_truth_light(tags: list[str]) -> bool:
    """Determine if recipe is light based on existing tags."""
    tags_lower = {t.lower() for t in tags}
    return bool(tags_lower & LIGHT_TAGS)


def get_ground_truth_cuisine(ingredients: list[str], title: str) -> str | None:
    """Guess cuisine from ingredients (returns None if uncertain)."""
    text = " ".join(ingredients).lower() + " " + title.lower()

    cuisine_scores = {}
    for cuisine, indicators in CUISINE_INDICATORS.items():
        score = sum(1 for ind in indicators if ind in text)
        if score >= 2:  # Need at least 2 indicators
            cuisine_scores[cuisine] = score

    if cuisine_scores:
        return max(cuisine_scores, key=cuisine_scores.get)
    return None


# ============================================================================
# LLM Classification
# ============================================================================

TASTE_OCCASION_PROMPT = """Classify this recipe's TASTE and OCCASION. Be selective and precise.

Recipe: {title}
Ingredients: {ingredients}

TASTE (pick 1-2 dominant flavors):
- sweet: Desserts, baked goods, or dishes where sweetness dominates
- savory: Main dishes, sides, appetizers that are not sweet
- spicy: Contains chili peppers, hot sauce, cayenne, jalapeño (NOT black pepper)
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

CUISINE_PROMPT = """What cuisine is this recipe? Pick the single most specific cuisine.

Recipe: {title}
Ingredients: {ingredients}

CUISINES (pick exactly one):
- american, southern-united-states, cajun, tex-mex
- mexican, caribbean, cuban, brazilian
- italian, french, spanish, greek, mediterranean
- german, british, irish, eastern-european
- chinese, japanese, korean, thai, vietnamese
- indian, middle-eastern, lebanese, turkish, moroccan
- african, ethiopian

Reply format:
CUISINE: <single cuisine>
CONFIDENCE: <high/medium/low>
"""


async def classify_recipe(client: httpx.AsyncClient, recipe_id: str, title: str, ingredients: list[str]) -> dict:
    """Classify a recipe with LLM for taste/occasion and cuisine."""
    result = {
        "recipe_id": recipe_id,
        "title": title,
        "taste_tags": [],
        "occasion_tags": [],
        "cuisine": None,
        "taste_occasion_confidence": "error",
        "cuisine_confidence": "error",
    }

    # Classify taste/occasion
    try:
        prompt = TASTE_OCCASION_PROMPT.format(
            title=title,
            ingredients=", ".join(ingredients[:15]),
        )
        response = await client.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=60.0,
        )
        response.raise_for_status()
        text = response.json()["response"].strip()

        for line in text.split("\n"):
            line = line.strip().lower()
            if line.startswith("tags:"):
                tags_str = line.replace("tags:", "").strip()
                for tag in tags_str.split(","):
                    tag = tag.strip()
                    if tag in TASTE_TAGS:
                        result["taste_tags"].append(tag)
                    elif tag in OCCASION_TAGS:
                        result["occasion_tags"].append(tag)
            elif line.startswith("confidence:"):
                result["taste_occasion_confidence"] = line.replace("confidence:", "").strip()
    except Exception as e:
        result["taste_occasion_confidence"] = f"error: {e}"

    # Classify cuisine
    try:
        prompt = CUISINE_PROMPT.format(
            title=title,
            ingredients=", ".join(ingredients[:12]),
        )
        response = await client.post(
            OLLAMA_URL,
            json={"model": MODEL, "prompt": prompt, "stream": False},
            timeout=60.0,
        )
        response.raise_for_status()
        text = response.json()["response"].strip()

        for line in text.split("\n"):
            line = line.strip().lower()
            if line.startswith("cuisine:"):
                cuisine = line.replace("cuisine:", "").strip()
                if cuisine in CUISINE_TAGS:
                    result["cuisine"] = cuisine
            elif line.startswith("confidence:"):
                result["cuisine_confidence"] = line.replace("confidence:", "").strip()
    except Exception as e:
        result["cuisine_confidence"] = f"error: {e}"

    return result


# ============================================================================
# Stratified Sampling
# ============================================================================

def get_stratified_sample(cursor, config: dict) -> list[tuple]:
    """Get stratified sample of recipes.

    Returns list of (recipe_id, title, ingredients, tags, stratum) tuples.
    """
    samples = []

    # Simple recipes (few ingredients, quick)
    cursor.execute("""
        SELECT recipe_id, title, ingredients_raw, tags
        FROM recipes
        WHERE n_ingredients <= 5 AND minutes <= 30
        AND ingredients_raw IS NOT NULL AND tags IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
    """, (config["simple"],))
    for row in cursor:
        samples.append((*row, "simple"))

    # Medium complexity
    cursor.execute("""
        SELECT recipe_id, title, ingredients_raw, tags
        FROM recipes
        WHERE n_ingredients BETWEEN 6 AND 10 AND minutes BETWEEN 30 AND 60
        AND ingredients_raw IS NOT NULL AND tags IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
    """, (config["medium"],))
    for row in cursor:
        samples.append((*row, "medium"))

    # Complex recipes
    cursor.execute("""
        SELECT recipe_id, title, ingredients_raw, tags
        FROM recipes
        WHERE (n_ingredients >= 15 OR minutes >= 120)
        AND ingredients_raw IS NOT NULL AND tags IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
    """, (config["complex"],))
    for row in cursor:
        samples.append((*row, "complex"))

    # Main dishes
    cursor.execute("""
        SELECT recipe_id, title, ingredients_raw, tags
        FROM recipes
        WHERE tags LIKE '%main-dish%'
        AND ingredients_raw IS NOT NULL AND tags IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
    """, (config["main_dish"],))
    for row in cursor:
        samples.append((*row, "main_dish"))

    # Desserts
    cursor.execute("""
        SELECT recipe_id, title, ingredients_raw, tags
        FROM recipes
        WHERE (tags LIKE '%dessert%' OR tags LIKE '%cookie%' OR tags LIKE '%cake%')
        AND ingredients_raw IS NOT NULL AND tags IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
    """, (config["dessert"],))
    for row in cursor:
        samples.append((*row, "dessert"))

    # Appetizers/Sides
    cursor.execute("""
        SELECT recipe_id, title, ingredients_raw, tags
        FROM recipes
        WHERE (tags LIKE '%appetizer%' OR tags LIKE '%side-dish%')
        AND ingredients_raw IS NOT NULL AND tags IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
    """, (config["appetizer_side"],))
    for row in cursor:
        samples.append((*row, "appetizer_side"))

    # With cuisine tags (use LIKE since SQLite doesn't have REGEXP by default)
    cursor.execute("""
        SELECT recipe_id, title, ingredients_raw, tags
        FROM recipes
        WHERE (tags LIKE '%italian%' OR tags LIKE '%mexican%' OR tags LIKE '%chinese%'
               OR tags LIKE '%indian%' OR tags LIKE '%thai%' OR tags LIKE '%japanese%'
               OR tags LIKE '%french%' OR tags LIKE '%greek%' OR tags LIKE '%korean%')
        AND ingredients_raw IS NOT NULL AND tags IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
    """, (config["with_cuisine"],))
    for row in cursor:
        samples.append((*row, "with_cuisine"))

    # Without cuisine tags (need classification)
    cursor.execute("""
        SELECT recipe_id, title, ingredients_raw, tags
        FROM recipes
        WHERE tags NOT LIKE '%italian%' AND tags NOT LIKE '%mexican%'
        AND tags NOT LIKE '%chinese%' AND tags NOT LIKE '%indian%'
        AND tags NOT LIKE '%thai%' AND tags NOT LIKE '%japanese%'
        AND tags NOT LIKE '%french%' AND tags NOT LIKE '%greek%'
        AND tags NOT LIKE '%korean%' AND tags NOT LIKE '%american%'
        AND ingredients_raw IS NOT NULL AND tags IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
    """, (config["without_cuisine"],))
    for row in cursor:
        samples.append((*row, "without_cuisine"))

    # Random for unbiased coverage
    cursor.execute("""
        SELECT recipe_id, title, ingredients_raw, tags
        FROM recipes
        WHERE ingredients_raw IS NOT NULL AND tags IS NOT NULL
        ORDER BY RANDOM()
        LIMIT ?
    """, (config["random"],))
    for row in cursor:
        samples.append((*row, "random"))

    return samples


# ============================================================================
# Metrics Calculation
# ============================================================================

@dataclass
class ClassificationMetrics:
    tag: str
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0

    @property
    def f1(self) -> float:
        if self.precision + self.recall == 0:
            return 0
        return 2 * self.precision * self.recall / (self.precision + self.recall)

    @property
    def accuracy(self) -> float:
        total = self.tp + self.fp + self.tn + self.fn
        return (self.tp + self.tn) / total if total > 0 else 0


def calculate_metrics(results: list[dict], samples: list[tuple]) -> dict[str, ClassificationMetrics]:
    """Calculate accuracy metrics for each verifiable tag."""
    metrics = {
        "spicy": ClassificationMetrics("spicy"),
        "sweet": ClassificationMetrics("sweet"),
        "rich": ClassificationMetrics("rich"),
        "light": ClassificationMetrics("light"),
        "cuisine": ClassificationMetrics("cuisine"),
    }

    # Build lookup for samples
    sample_lookup = {}
    for recipe_id, title, ingredients_json, tags_json, stratum in samples:
        ingredients = json.loads(ingredients_json) if ingredients_json else []
        tags = json.loads(tags_json) if tags_json else []
        sample_lookup[recipe_id] = (title, ingredients, tags)

    for result in results:
        recipe_id = result["recipe_id"]
        if recipe_id not in sample_lookup:
            continue

        title, ingredients, tags = sample_lookup[recipe_id]

        # Spicy
        gt_spicy = get_ground_truth_spicy(ingredients, title)
        pred_spicy = "spicy" in result["taste_tags"]
        if gt_spicy and pred_spicy:
            metrics["spicy"].tp += 1
        elif gt_spicy and not pred_spicy:
            metrics["spicy"].fn += 1
        elif not gt_spicy and pred_spicy:
            metrics["spicy"].fp += 1
        else:
            metrics["spicy"].tn += 1

        # Sweet
        gt_sweet = get_ground_truth_sweet(ingredients, title, tags)
        pred_sweet = "sweet" in result["taste_tags"]
        if gt_sweet and pred_sweet:
            metrics["sweet"].tp += 1
        elif gt_sweet and not pred_sweet:
            metrics["sweet"].fn += 1
        elif not gt_sweet and pred_sweet:
            metrics["sweet"].fp += 1
        else:
            metrics["sweet"].tn += 1

        # Rich
        gt_rich = get_ground_truth_rich(ingredients)
        pred_rich = "rich" in result["taste_tags"]
        if gt_rich and pred_rich:
            metrics["rich"].tp += 1
        elif gt_rich and not pred_rich:
            metrics["rich"].fn += 1
        elif not gt_rich and pred_rich:
            metrics["rich"].fp += 1
        else:
            metrics["rich"].tn += 1

        # Light
        gt_light = get_ground_truth_light(tags)
        pred_light = "light" in result["taste_tags"]
        if gt_light and pred_light:
            metrics["light"].tp += 1
        elif gt_light and not pred_light:
            metrics["light"].fn += 1
        elif not gt_light and pred_light:
            metrics["light"].fp += 1
        else:
            metrics["light"].tn += 1

        # Cuisine (only if we have ground truth)
        gt_cuisine = get_ground_truth_cuisine(ingredients, title)
        if gt_cuisine:
            pred_cuisine = result["cuisine"]
            if gt_cuisine == pred_cuisine:
                metrics["cuisine"].tp += 1
            elif pred_cuisine:
                metrics["cuisine"].fp += 1
            else:
                metrics["cuisine"].fn += 1

    return metrics


def print_metrics_report(metrics: dict[str, ClassificationMetrics], total: int):
    """Print formatted metrics report."""
    print("\n" + "=" * 70)
    print("CLASSIFICATION ACCURACY REPORT")
    print("=" * 70)
    print(f"\n{'Tag':<15} {'Precision':<12} {'Recall':<12} {'F1':<12} {'Samples':<10}")
    print("-" * 60)

    f1_values = []
    for tag, m in metrics.items():
        samples_for_tag = m.tp + m.fp + m.tn + m.fn
        if samples_for_tag > 0:
            print(f"{tag:<15} {m.precision:<12.1%} {m.recall:<12.1%} {m.f1:<12.2f} {samples_for_tag:<10}")
            f1_values.append(m.f1)

    if f1_values:
        avg_f1 = sum(f1_values) / len(f1_values)
        print("-" * 60)
        print(f"{'AVERAGE F1':<15} {'':<12} {'':<12} {avg_f1:<12.2f}")

    print("\n" + "=" * 70)
    print("GO/NO-GO ASSESSMENT")
    print("=" * 70)

    # Check thresholds
    taste_f1 = sum(m.f1 for tag, m in metrics.items() if tag in ("spicy", "sweet", "rich", "light")) / 4
    cuisine_f1 = metrics["cuisine"].f1

    print(f"\nTaste F1 (target >= 0.75): {taste_f1:.2f} {'PASS' if taste_f1 >= 0.75 else 'FAIL'}")
    print(f"Cuisine F1 (target >= 0.80): {cuisine_f1:.2f} {'PASS' if cuisine_f1 >= 0.80 else 'FAIL'}")

    if taste_f1 >= 0.75 and cuisine_f1 >= 0.80:
        print("\n>>> RECOMMENDATION: PROCEED with full classification")
    elif taste_f1 >= 0.70 or cuisine_f1 >= 0.75:
        print("\n>>> RECOMMENDATION: PROCEED with caution (review false positives)")
    else:
        print("\n>>> RECOMMENDATION: REVISE prompts and re-validate")


# ============================================================================
# Main
# ============================================================================

async def run_validation(quick: bool = False, export_csv: bool = False):
    """Run the full validation pipeline."""
    config = SAMPLE_CONFIG["quick" if quick else "full"]
    total_samples = sum(config.values())

    print(f"{'=' * 70}")
    print(f"LLM CLASSIFICATION VALIDATION")
    print(f"{'=' * 70}")
    print(f"Mode: {'Quick' if quick else 'Full'} ({total_samples} recipes)")
    print(f"Model: {MODEL}")

    # Connect to database
    if not DB_PATH.exists():
        print(f"Error: Database not found at {DB_PATH}")
        return

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Get stratified sample
    print("\n1. Selecting stratified sample...")
    samples = get_stratified_sample(cursor, config)
    print(f"   Selected {len(samples)} recipes across {len(config)} strata")

    # Show stratum distribution
    strata_counts = {}
    for _, _, _, _, stratum in samples:
        strata_counts[stratum] = strata_counts.get(stratum, 0) + 1
    for stratum, count in sorted(strata_counts.items()):
        print(f"   - {stratum}: {count}")

    if export_csv:
        # Export for manual review
        csv_path = Path("data/validation_samples.csv")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["recipe_id", "title", "ingredients_preview", "existing_tags", "stratum",
                            "manual_taste", "manual_occasion", "manual_cuisine"])
            for recipe_id, title, ingredients_json, tags_json, stratum in samples:
                ingredients = json.loads(ingredients_json) if ingredients_json else []
                tags = json.loads(tags_json) if tags_json else []
                writer.writerow([
                    recipe_id,
                    title,
                    ", ".join(ingredients[:8]),
                    ", ".join(tags[:10]),
                    stratum,
                    "",  # manual_taste
                    "",  # manual_occasion
                    "",  # manual_cuisine
                ])
        print(f"\nExported to {csv_path} for manual review")
        conn.close()
        return

    # Run LLM classification
    print("\n2. Running LLM classification...")
    start_time = time.time()

    results = []
    async with httpx.AsyncClient() as client:
        for i, (recipe_id, title, ingredients_json, tags_json, stratum) in enumerate(samples):
            ingredients = json.loads(ingredients_json) if ingredients_json else []

            result = await classify_recipe(client, recipe_id, title, ingredients)
            result["stratum"] = stratum
            results.append(result)

            if (i + 1) % 20 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                remaining = len(samples) - (i + 1)
                eta = remaining / rate if rate > 0 else 0
                print(f"   Processed {i + 1}/{len(samples)} ({rate:.1f}/sec, ETA: {eta/60:.1f}min)")

    elapsed = time.time() - start_time
    print(f"   Completed in {elapsed/60:.1f} minutes ({len(samples)/elapsed:.1f} recipes/sec)")

    # Calculate metrics
    print("\n3. Calculating accuracy metrics...")
    metrics = calculate_metrics(results, samples)

    # Print report
    print_metrics_report(metrics, len(samples))

    # Save results
    results_path = Path("data/validation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Show sample predictions
    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)

    for result in results[:5]:
        print(f"\nRecipe: {result['title'][:50]}")
        print(f"  Taste: {result['taste_tags']} ({result['taste_occasion_confidence']})")
        print(f"  Occasion: {result['occasion_tags']}")
        print(f"  Cuisine: {result['cuisine']} ({result['cuisine_confidence']})")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate LLM classification accuracy")
    parser.add_argument("--quick", action="store_true", help="Quick validation (100 recipes)")
    parser.add_argument("--export-csv", action="store_true", help="Export samples for manual review")
    args = parser.parse_args()

    asyncio.run(run_validation(quick=args.quick, export_csv=args.export_csv))

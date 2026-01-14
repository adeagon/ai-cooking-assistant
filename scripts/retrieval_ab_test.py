"""A/B test for retrieval quality with and without tag filtering.

Measures whether LLM-classified tags improve search result relevance
by comparing semantic-only search vs semantic + tag filtering.

Usage:
    python scripts/retrieval_ab_test.py
"""

import json
import sqlite3
from pathlib import Path

from src.app.settings import settings
from src.retrieval.retriever import RecipeRetriever

# ============================================================================
# Test Queries
# ============================================================================

# Each query has expected tags that results should have
TEST_QUERIES = [
    # Taste-based queries
    {
        "query": "light summer salad dinner",
        "expected_tags": ["light"],
        "category": "taste",
    },
    {
        "query": "rich creamy pasta comfort food",
        "expected_tags": ["rich"],
        "category": "taste",
    },
    {
        "query": "spicy hot wings appetizer",
        "expected_tags": ["spicy"],
        "category": "taste",
    },
    {
        "query": "mild family friendly chicken",
        "expected_tags": ["mild"],
        "category": "taste",
    },
    {
        "query": "sweet chocolate dessert cake",
        "expected_tags": ["sweet"],
        "category": "taste",
    },

    # Occasion-based queries
    {
        "query": "quick easy weeknight dinner",
        "expected_tags": ["weeknight"],
        "category": "occasion",
    },
    {
        "query": "impressive dinner party main course",
        "expected_tags": ["dinner-party"],
        "category": "occasion",
    },
    {
        "query": "kid friendly lunch picky eaters",
        "expected_tags": ["kid-friendly"],
        "category": "occasion",
    },
    {
        "query": "budget friendly cheap meal",
        "expected_tags": ["inexpensive"],
        "category": "occasion",
    },
    {
        "query": "holiday christmas dinner special",
        "expected_tags": ["holiday-event"],
        "category": "occasion",
    },

    # Cuisine-based queries
    {
        "query": "authentic italian pasta marinara",
        "expected_cuisine": "italian",
        "category": "cuisine",
    },
    {
        "query": "spicy mexican tacos salsa",
        "expected_cuisine": "mexican",
        "category": "cuisine",
    },
    {
        "query": "chinese stir fry soy sauce",
        "expected_cuisine": "chinese",
        "category": "cuisine",
    },
    {
        "query": "indian curry tikka masala",
        "expected_cuisine": "indian",
        "category": "cuisine",
    },
    {
        "query": "thai coconut curry basil",
        "expected_cuisine": "thai",
        "category": "cuisine",
    },

    # Combined queries
    {
        "query": "light italian salad vegetarian",
        "expected_tags": ["light"],
        "expected_cuisine": "italian",
        "category": "combined",
    },
    {
        "query": "spicy korean weeknight quick",
        "expected_tags": ["spicy", "weeknight"],
        "expected_cuisine": "korean",
        "category": "combined",
    },
    {
        "query": "rich french dessert cream",
        "expected_tags": ["rich", "sweet"],
        "expected_cuisine": "french",
        "category": "combined",
    },
    {
        "query": "mild kid friendly japanese",
        "expected_tags": ["mild", "kid-friendly"],
        "expected_cuisine": "japanese",
        "category": "combined",
    },
    {
        "query": "comfort food american dinner party",
        "expected_tags": ["comfort-food", "dinner-party"],
        "expected_cuisine": "american",
        "category": "combined",
    },
]


def get_recipe_tags(recipe_id: str, cursor) -> tuple[list[str], str]:
    """Get tags and cuisine for a recipe from the database."""
    cursor.execute(
        "SELECT tags FROM recipes WHERE recipe_id = ?",
        (recipe_id,)
    )
    row = cursor.fetchone()
    if not row or not row[0]:
        return [], ""

    tags = json.loads(row[0]) if row[0] else []
    tags_lower = [t.lower() for t in tags]

    # Extract cuisine from tags
    cuisine_tags = {
        "italian", "mexican", "chinese", "indian", "thai", "japanese",
        "french", "greek", "mediterranean", "korean", "vietnamese",
        "american", "southern-united-states", "asian", "european",
        "middle-eastern", "spanish", "german", "british", "irish",
        "cajun", "creole", "caribbean", "african", "brazilian",
    }
    cuisine = ""
    for tag in tags_lower:
        if tag in cuisine_tags:
            cuisine = tag
            break

    return tags_lower, cuisine


def check_result_relevance(
    recipe_id: str,
    cursor,
    expected_tags: list[str] | None,
    expected_cuisine: str | None,
) -> tuple[bool, dict]:
    """Check if a result matches expected tags/cuisine."""
    tags, cuisine = get_recipe_tags(recipe_id, cursor)

    tag_match = True
    cuisine_match = True

    if expected_tags:
        tag_match = any(t in tags for t in expected_tags)

    if expected_cuisine:
        cuisine_match = cuisine == expected_cuisine

    return tag_match and cuisine_match, {"tags": tags, "cuisine": cuisine}


def calculate_precision_at_k(relevant_flags: list[bool], k: int) -> float:
    """Calculate Precision@k."""
    if k > len(relevant_flags):
        k = len(relevant_flags)
    if k == 0:
        return 0.0
    return sum(relevant_flags[:k]) / k


def calculate_mrr(relevant_flags: list[bool]) -> float:
    """Calculate Mean Reciprocal Rank."""
    for i, is_relevant in enumerate(relevant_flags):
        if is_relevant:
            return 1.0 / (i + 1)
    return 0.0


def run_ab_test():
    """Run A/B test comparing search with and without tags."""
    print("=" * 70)
    print("RETRIEVAL A/B TEST")
    print("=" * 70)

    # Initialize retriever
    print("\nLoading retriever...")
    retriever = RecipeRetriever(
        chroma_dir=settings.chroma_persist_dir,
        embedding_model=settings.embedding_model,
    )

    # Connect to SQLite for tag lookup
    db_path = Path("data/sqlite/recipes.db")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    results = {
        "semantic_only": {"precision_at_5": [], "mrr": []},
        "with_tags": {"precision_at_5": [], "mrr": []},
    }

    category_results = {
        "taste": {"semantic": [], "with_tags": []},
        "occasion": {"semantic": [], "with_tags": []},
        "cuisine": {"semantic": [], "with_tags": []},
        "combined": {"semantic": [], "with_tags": []},
    }

    print(f"\nRunning {len(TEST_QUERIES)} test queries...")
    print("-" * 70)

    for i, test in enumerate(TEST_QUERIES):
        query = test["query"]
        expected_tags = test.get("expected_tags", [])
        expected_cuisine = test.get("expected_cuisine")
        category = test["category"]

        print(f"\n{i+1}. Query: \"{query}\"")
        print(f"   Expected: tags={expected_tags}, cuisine={expected_cuisine}")

        # A: Semantic-only search
        semantic_results = retriever.search(query, k=10)
        semantic_relevant = []
        for r in semantic_results:
            is_relevant, _ = check_result_relevance(
                r.recipe_id, cursor, expected_tags, expected_cuisine
            )
            semantic_relevant.append(is_relevant)

        semantic_p5 = calculate_precision_at_k(semantic_relevant, 5)
        semantic_mrr = calculate_mrr(semantic_relevant)

        # B: Search with tag/cuisine filter (if available)
        # Note: We can only filter on is_vegetarian, is_vegan, cuisine, and minutes
        # For this test, we'll use cuisine filter when expected_cuisine is set
        dietary = None
        cuisine_filter = expected_cuisine if expected_cuisine else None

        if cuisine_filter:
            filtered_results = retriever.search_with_constraints(
                query, k=10, cuisine=cuisine_filter
            )
        else:
            # No filter available for taste/occasion tags yet
            filtered_results = semantic_results

        filtered_relevant = []
        for r in filtered_results:
            is_relevant, _ = check_result_relevance(
                r.recipe_id, cursor, expected_tags, expected_cuisine
            )
            filtered_relevant.append(is_relevant)

        filtered_p5 = calculate_precision_at_k(filtered_relevant, 5)
        filtered_mrr = calculate_mrr(filtered_relevant)

        # Record results
        results["semantic_only"]["precision_at_5"].append(semantic_p5)
        results["semantic_only"]["mrr"].append(semantic_mrr)
        results["with_tags"]["precision_at_5"].append(filtered_p5)
        results["with_tags"]["mrr"].append(filtered_mrr)

        category_results[category]["semantic"].append(semantic_p5)
        category_results[category]["with_tags"].append(filtered_p5)

        print(f"   Semantic P@5: {semantic_p5:.2f}, MRR: {semantic_mrr:.2f}")
        print(f"   Filtered P@5: {filtered_p5:.2f}, MRR: {filtered_mrr:.2f}")
        if filtered_p5 > semantic_p5:
            print(f"   >>> Improvement: +{filtered_p5 - semantic_p5:.2f}")
        elif filtered_p5 < semantic_p5:
            print(f"   >>> Regression: {filtered_p5 - semantic_p5:.2f}")

    conn.close()

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    avg_semantic_p5 = sum(results["semantic_only"]["precision_at_5"]) / len(results["semantic_only"]["precision_at_5"])
    avg_filtered_p5 = sum(results["with_tags"]["precision_at_5"]) / len(results["with_tags"]["precision_at_5"])
    avg_semantic_mrr = sum(results["semantic_only"]["mrr"]) / len(results["semantic_only"]["mrr"])
    avg_filtered_mrr = sum(results["with_tags"]["mrr"]) / len(results["with_tags"]["mrr"])

    print(f"\n{'Metric':<25} {'Semantic Only':<15} {'With Filters':<15} {'Delta':<10}")
    print("-" * 65)
    print(f"{'Precision@5 (avg)':<25} {avg_semantic_p5:<15.2%} {avg_filtered_p5:<15.2%} {avg_filtered_p5 - avg_semantic_p5:+.2%}")
    print(f"{'MRR (avg)':<25} {avg_semantic_mrr:<15.2f} {avg_filtered_mrr:<15.2f} {avg_filtered_mrr - avg_semantic_mrr:+.2f}")

    print("\n" + "-" * 65)
    print("By Category:")
    for category in ["taste", "occasion", "cuisine", "combined"]:
        if category_results[category]["semantic"]:
            sem_avg = sum(category_results[category]["semantic"]) / len(category_results[category]["semantic"])
            flt_avg = sum(category_results[category]["with_tags"]) / len(category_results[category]["with_tags"])
            print(f"  {category.capitalize():<20} Semantic: {sem_avg:.2%}  Filtered: {flt_avg:.2%}  Delta: {flt_avg - sem_avg:+.2%}")

    # Go/no-go assessment
    print("\n" + "=" * 70)
    print("GO/NO-GO ASSESSMENT")
    print("=" * 70)

    improvement = avg_filtered_p5 - avg_semantic_p5
    improvement_pct = (avg_filtered_p5 - avg_semantic_p5) / avg_semantic_p5 * 100 if avg_semantic_p5 > 0 else 0

    print(f"\nPrecision@5 improvement: {improvement:+.2%} ({improvement_pct:+.1f}% relative)")
    print(f"Target: >= 10% relative improvement")

    if improvement_pct >= 15:
        print("\n>>> RESULT: STRONG IMPROVEMENT - Tags significantly improve retrieval")
    elif improvement_pct >= 10:
        print("\n>>> RESULT: MODERATE IMPROVEMENT - Tags provide measurable benefit")
    elif improvement_pct >= 5:
        print("\n>>> RESULT: MARGINAL IMPROVEMENT - Tags provide small benefit")
    elif improvement_pct >= 0:
        print("\n>>> RESULT: NO IMPROVEMENT - Tags don't help (but don't hurt)")
    else:
        print("\n>>> RESULT: REGRESSION - Tags are hurting retrieval quality")

    print("\nNOTE: This test primarily measures CUISINE filtering.")
    print("      Taste/occasion tags are not yet filterable in ChromaDB.")
    print("      Full benefit requires adding taste/occasion to metadata.")


if __name__ == "__main__":
    run_ab_test()

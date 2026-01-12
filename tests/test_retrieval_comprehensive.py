"""Comprehensive tests for recipe retrieval and search quality."""

import pytest
from pathlib import Path
from src.retrieval.retriever import RecipeRetriever
from src.app.settings import settings


@pytest.mark.skipif(
    not Path("data/chroma").exists(),
    reason="Vector store not built (run 'ingest embed' first)"
)
class TestSearchQuality:
    """Comprehensive search quality tests."""

    @pytest.fixture
    def retriever(self):
        """Create a retriever instance."""
        return RecipeRetriever(
            chroma_dir=settings.chroma_persist_dir,
            embedding_model=settings.embedding_model
        )

    def test_ingredient_based_search(self, retriever):
        """Test searches based on specific ingredients."""
        queries = [
            ("chicken breast", "chicken"),
            ("salmon fish", "salmon"),
            ("beef steak", "beef"),
            ("tofu vegetarian", "tofu"),
        ]

        for query, expected_ingredient in queries:
            results = retriever.search(query, k=10)
            assert len(results) > 0, f"No results for '{query}'"

            # Check top 5 results contain the expected ingredient
            top_titles = " ".join([r.title.lower() for r in results[:5]])
            assert expected_ingredient in top_titles or any(
                expected_ingredient in r.title.lower() for r in results[:10]
            ), f"Expected '{expected_ingredient}' in results for '{query}'"

    def test_cuisine_based_search(self, retriever):
        """Test searches based on cuisine types."""
        cuisines = ["italian pasta", "mexican tacos", "chinese stir fry", "indian curry"]

        for query in cuisines:
            results = retriever.search(query, k=10)
            assert len(results) > 0, f"No results for '{query}'"
            print(f"\n{query}: {results[0].title} (score: {results[0].score:.3f})")

    def test_cooking_method_search(self, retriever):
        """Test searches based on cooking methods."""
        methods = [
            "grilled chicken",
            "baked salmon",
            "fried rice",
            "slow cooker beef",
        ]

        for query in methods:
            results = retriever.search(query, k=10)
            assert len(results) > 0, f"No results for '{query}'"
            assert results[0].score > 0.7, f"Low relevance for '{query}'"

    def test_dietary_restriction_search(self, retriever):
        """Test searches with dietary restrictions."""
        queries = [
            "vegetarian pasta",
            "vegan soup",
            "gluten free bread",
            "low carb dinner",
        ]

        for query in queries:
            results = retriever.search(query, k=10)
            assert len(results) > 0, f"No results for '{query}'"
            print(f"\n{query}: {results[0].title}")

    def test_time_based_search(self, retriever):
        """Test searches with time constraints."""
        queries = [
            "quick 15 minute dinner",
            "fast breakfast",
            "quick lunch",
        ]

        for query in queries:
            results = retriever.search(query, k=10)
            assert len(results) > 0, f"No results for '{query}'"

            # Check if any results are actually quick
            quick_recipes = [r for r in results if r.minutes and r.minutes <= 30]
            assert len(quick_recipes) > 0, f"No quick recipes for '{query}'"

    def test_multi_ingredient_search(self, retriever):
        """Test searches with multiple ingredients."""
        queries = [
            "chicken tomato garlic",
            "beef onion mushroom",
            "pasta cheese basil",
            "salmon lemon dill",
        ]

        for query in queries:
            results = retriever.search(query, k=10)
            assert len(results) > 0, f"No results for '{query}'"
            assert results[0].score > 0.75, f"Low relevance for multi-ingredient query '{query}'"

    def test_complex_queries(self, retriever):
        """Test complex natural language queries."""
        queries = [
            "healthy chicken dinner under 30 minutes",
            "spicy vegetarian mexican food",
            "easy chocolate dessert for beginners",
            "comfort food pasta with cheese",
        ]

        for query in queries:
            results = retriever.search(query, k=10)
            assert len(results) > 0, f"No results for '{query}'"
            print(f"\n{query}:")
            for i, r in enumerate(results[:3], 1):
                print(f"  {i}. {r.title} (score: {r.score:.3f})")

    def test_score_distribution(self, retriever):
        """Test that scores are properly distributed."""
        results = retriever.search("chicken tomato spicy", k=30)

        scores = [r.score for r in results]

        # Scores should be between 0 and 1
        assert all(0 <= s <= 1 for s in scores), "Scores outside valid range"

        # Scores should be in descending order
        assert scores == sorted(scores, reverse=True), "Scores not in descending order"

        # Top score should be reasonably high for a good query
        assert scores[0] > 0.8, f"Top score too low: {scores[0]}"

        # Should have some variety in scores
        score_range = scores[0] - scores[-1]
        assert score_range > 0.05, "Scores too similar - poor discrimination"

        print(f"\nScore range: {scores[0]:.3f} to {scores[-1]:.3f} (range: {score_range:.3f})")

    def test_empty_query_handling(self, retriever):
        """Test handling of edge case queries."""
        # Very short query
        results = retriever.search("a", k=10)
        assert len(results) > 0

        # Single word
        results = retriever.search("chicken", k=10)
        assert len(results) > 0

    def test_performance_with_different_k(self, retriever):
        """Test search performance with different k values."""
        import time

        k_values = [10, 30, 50, 100]
        query = "chicken tomato spicy"

        for k in k_values:
            start = time.time()
            results = retriever.search(query, k=k)
            elapsed_ms = (time.time() - start) * 1000

            assert len(results) == min(k, 88399)  # Total recipes in dataset

            # Even with k=100, should be reasonably fast
            if k <= 50:
                assert elapsed_ms < 300, f"Search with k={k} took {elapsed_ms:.0f}ms"

            print(f"\nk={k}: {elapsed_ms:.0f}ms")

    def test_rating_filter_effectiveness(self, retriever):
        """Test that rating filters work correctly."""
        query = "chocolate cake"

        # Without filter
        results_all = retriever.search(query, k=20)

        # With high rating filter
        results_filtered = retriever.search_with_filters(
            query=query,
            k=20,
            min_rating=4.5
        )

        # All filtered results should meet criteria
        for result in results_filtered:
            if result.rating_avg is not None:
                assert result.rating_avg >= 4.5, f"Recipe {result.title} has rating {result.rating_avg}"

        # Should get some results
        assert len(results_filtered) > 0, "No results with rating filter"

        print(f"\nWithout filter: {len(results_all)} results")
        print(f"With rating >= 4.5: {len(results_filtered)} results")

    def test_time_filter_effectiveness(self, retriever):
        """Test that time filters work correctly."""
        query = "dinner recipe"

        # Quick recipes only (30 minutes or less)
        results_quick = retriever.search_with_filters(
            query=query,
            k=20,
            max_minutes=30
        )

        # All filtered results should meet criteria
        for result in results_quick:
            if result.minutes is not None:
                assert result.minutes <= 30, f"Recipe {result.title} takes {result.minutes}m"

        assert len(results_quick) > 0, "No results with time filter"

        print(f"\nRecipes under 30 minutes: {len(results_quick)} results")
        print("Sample quick recipes:")
        for r in results_quick[:5]:
            print(f"  - {r.title} ({r.minutes}m)")

    def test_combined_filters(self, retriever):
        """Test using multiple filters together."""
        query = "pasta dinner"

        results = retriever.search_with_filters(
            query=query,
            k=20,
            min_rating=4.0,
            max_minutes=45
        )

        # Check all results meet both criteria
        for result in results:
            if result.rating_avg is not None:
                assert result.rating_avg >= 4.0
            if result.minutes is not None:
                assert result.minutes <= 45

        assert len(results) > 0, "No results with combined filters"

        print(f"\nPasta recipes (rating >= 4.0, time <= 45m): {len(results)} results")

    def test_variety_in_results(self, retriever):
        """Test that results show variety, not just duplicates."""
        results = retriever.search("chicken dinner", k=20)

        titles = [r.title.lower() for r in results]
        unique_titles = set(titles)

        # Should have mostly unique titles
        assert len(unique_titles) >= len(titles) * 0.9, "Too many duplicate titles"

        # Titles should not be too similar
        first_words = [t.split()[0] for t in titles if t]
        unique_first_words = set(first_words)
        assert len(unique_first_words) > 5, "Results lack diversity"

    def test_specific_recipe_types(self, retriever):
        """Test specific recipe type searches."""
        recipe_types = [
            ("soup", ["soup", "stew", "chowder", "bisque"]),
            ("salad", ["salad", "slaw"]),
            ("cake", ["cake", "torte"]),
            ("cookie", ["cookie", "cookies", "biscuit"]),
        ]

        for query, expected_terms in recipe_types:
            results = retriever.search(query, k=10)
            assert len(results) > 0, f"No results for '{query}'"

            # Check that top results match expected terms
            top_titles_lower = " ".join([r.title.lower() for r in results[:10]])
            matches = any(term in top_titles_lower for term in expected_terms)
            assert matches, f"Results for '{query}' don't match expected terms: {expected_terms}"

    def test_seasonal_ingredients(self, retriever):
        """Test searches with seasonal ingredients."""
        seasonal = [
            "pumpkin spice",
            "summer tomato",
            "spring asparagus",
            "winter squash",
        ]

        for query in seasonal:
            results = retriever.search(query, k=10)
            assert len(results) > 0, f"No results for '{query}'"
            print(f"\n{query}: {results[0].title}")

    def test_consistency_across_runs(self, retriever):
        """Test that search results are consistent across multiple runs."""
        query = "chicken tomato spicy"

        results1 = retriever.search(query, k=10)
        results2 = retriever.search(query, k=10)

        # Should get identical results
        assert len(results1) == len(results2)

        for r1, r2 in zip(results1, results2):
            assert r1.recipe_id == r2.recipe_id, "Results not consistent across runs"
            assert abs(r1.score - r2.score) < 0.001, "Scores not consistent"

    def test_query_variations(self, retriever):
        """Test that similar queries return similar results."""
        base_query = "spicy chicken tacos"
        variations = [
            "spicy chicken taco",
            "chicken tacos spicy",
            "hot chicken tacos",
        ]

        base_results = retriever.search(base_query, k=5)
        base_ids = [r.recipe_id for r in base_results]

        for variant in variations:
            variant_results = retriever.search(variant, k=10)
            variant_ids = [r.recipe_id for r in variant_results]

            # Should have some overlap with base results
            overlap = set(base_ids) & set(variant_ids)
            overlap_ratio = len(overlap) / len(base_ids)

            print(f"\n'{base_query}' vs '{variant}': {overlap_ratio:.1%} overlap")
            assert overlap_ratio > 0.3, f"Too little overlap for similar queries"

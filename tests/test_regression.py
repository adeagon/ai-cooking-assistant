"""Regression tests to ensure Phase 3 changes don't break earlier functionality."""

import pytest
from pathlib import Path
from src.domain.models import Recipe, RetrievalResult


class TestPhase1Regression:
    """Regression tests for Phase 1 (Data Ingestion) functionality."""

    def test_recipe_model_structure(self):
        """Test that Recipe model still works as expected."""
        recipe = Recipe(
            recipe_id="123",
            title="Test Recipe",
            ingredients=["1 cup flour", "2 eggs"],
            ingredients_normalized=["flour", "eggs"],
            instructions=["Mix ingredients", "Bake at 350F"],
            tags=["baking", "easy"],
            rating_avg=4.5,
            rating_count=100,
            minutes=30,
            n_steps=2,
            n_ingredients=2,
            source="foodcom"
        )

        assert recipe.recipe_id == "123"
        assert recipe.title == "Test Recipe"
        assert len(recipe.ingredients) == 2
        assert len(recipe.ingredients_normalized) == 2
        assert len(recipe.instructions) == 2
        assert len(recipe.tags) == 2
        assert recipe.rating_avg == 4.5
        assert recipe.rating_count == 100
        assert recipe.minutes == 30

    def test_ingredient_normalization(self):
        """Test that ingredient normalization still works."""
        from src.ingest.normalize import normalize_ingredient
        from src.domain.models import NormalizedIngredient

        # Test that it returns NormalizedIngredient objects
        result = normalize_ingredient("1 cup flour")
        assert isinstance(result, NormalizedIngredient)
        assert result.raw == "1 cup flour"
        assert "flour" in result.name.lower()

        result2 = normalize_ingredient("2 tablespoons olive oil")
        assert isinstance(result2, NormalizedIngredient)
        assert "oil" in result2.name.lower()

        result3 = normalize_ingredient("Salt and pepper to taste")
        assert isinstance(result3, NormalizedIngredient)
        assert "salt" in result3.name.lower() or "pepper" in result3.name.lower()

    def test_extract_key_ingredients(self):
        """Test key ingredient extraction."""
        from src.ingest.normalize import extract_key_ingredients

        ingredients = [
            "1 cup all-purpose flour",
            "2 large eggs",
            "1/2 cup sugar",
            "1 teaspoon vanilla extract"
        ]

        key_ingredients = extract_key_ingredients(ingredients)

        # Returns list of normalized names
        assert isinstance(key_ingredients, list)
        assert len(key_ingredients) > 0

        # Check that ingredients are present (may be full strings if parsing fails)
        ingredients_str = " ".join(key_ingredients).lower()
        assert "flour" in ingredients_str
        assert "egg" in ingredients_str
        assert "sugar" in ingredients_str

    def test_quality_filters(self):
        """Test that quality filters still work."""
        from src.ingest.filters import apply_quality_filters
        from src.domain.models import RatingStats

        # Should pass filters
        good_recipe = {
            "id": "123",
            "name": "Good Recipe",
            "minutes": 45,
            "n_steps": 5,
            "n_ingredients": 8,
            "ingredients": ["flour", "eggs", "sugar"],
            "steps": ["Mix ingredients", "Bake", "Cool"]
        }
        good_rating = RatingStats(rating_avg=4.5, rating_count=50)
        assert apply_quality_filters(good_recipe, good_rating) is True

        # Should fail - low rating
        bad_recipe = {
            "id": "456",
            "name": "Bad Recipe",
            "minutes": 45,
            "n_steps": 5,
            "n_ingredients": 8,
            "ingredients": ["flour", "eggs"],
            "steps": ["Mix", "Bake"]
        }
        bad_rating = RatingStats(rating_avg=2.0, rating_count=50)
        assert apply_quality_filters(bad_recipe, bad_rating) is False

        # Should fail - low rating count
        low_count_recipe = {
            "id": "789",
            "name": "Unknown Recipe",
            "minutes": 45,
            "n_steps": 5,
            "n_ingredients": 8,
            "ingredients": ["flour"],
            "steps": ["Mix"]
        }
        low_count_rating = RatingStats(rating_avg=4.5, rating_count=1)
        assert apply_quality_filters(low_count_recipe, low_count_rating) is False

    @pytest.mark.skipif(
        not Path("data/sqlite/recipes.db").exists(),
        reason="Database not built"
    )
    def test_database_access(self):
        """Test that database access still works."""
        from src.ingest.build_db import get_recipe_by_id
        from src.app.settings import settings

        # Get database stats
        from src.ingest.build_db import get_stats
        stats = get_stats(settings.sqlite_db_path)

        assert stats["total_recipes"] > 0
        print(f"\nDatabase has {stats['total_recipes']} recipes")


class TestPhase2Regression:
    """Regression tests for Phase 2 (Embeddings + Vector Store) functionality."""

    def test_embedding_text_format(self):
        """Test that embedding text generation still works."""
        from src.ingest.build_embeddings import build_embedding_text

        recipe = Recipe(
            recipe_id="123",
            title="Chicken Soup",
            ingredients_normalized=["chicken", "carrot", "celery"],
            tags=["soup", "comfort-food", "easy"]
        )

        text = build_embedding_text(recipe)

        assert "Chicken Soup" in text
        assert "chicken" in text
        assert "carrot" in text
        assert "soup" in text

    @pytest.mark.skipif(
        not Path("data/chroma").exists(),
        reason="Vector store not built"
    )
    def test_vector_retrieval_still_works(self):
        """Test that basic vector retrieval (without reranking) still works."""
        from src.retrieval.retriever import RecipeRetriever
        from src.app.settings import settings

        retriever = RecipeRetriever(
            chroma_dir=settings.chroma_persist_dir,
            embedding_model=settings.embedding_model
        )

        # Test basic search
        results = retriever.search("chicken soup", k=10)

        assert len(results) == 10
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert all(r.recipe_id for r in results)
        assert all(r.title for r in results)
        assert all(0 <= r.score <= 1 for r in results)

        # Scores should be sorted descending
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.skipif(
        not Path("data/chroma").exists(),
        reason="Vector store not built"
    )
    def test_vector_retrieval_with_filters(self):
        """Test that filtered retrieval still works."""
        from src.retrieval.retriever import RecipeRetriever
        from src.app.settings import settings

        retriever = RecipeRetriever(
            chroma_dir=settings.chroma_persist_dir,
            embedding_model=settings.embedding_model
        )

        # Test rating filter
        results = retriever.search_with_filters(
            "chicken",
            k=10,
            min_rating=4.5
        )

        assert len(results) <= 10
        # All results should have high ratings
        for r in results:
            if r.rating_avg is not None:
                assert r.rating_avg >= 4.5

        # Test time filter
        results = retriever.search_with_filters(
            "pasta",
            k=10,
            max_minutes=30
        )

        assert len(results) <= 10
        # All results should be quick
        for r in results:
            if r.minutes is not None:
                assert r.minutes <= 30

    @pytest.mark.skipif(
        not Path("data/chroma").exists(),
        reason="Vector store not built"
    )
    def test_retrieval_performance_regression(self):
        """Test that retrieval performance hasn't regressed."""
        import time
        from src.retrieval.retriever import RecipeRetriever
        from src.app.settings import settings

        retriever = RecipeRetriever(
            chroma_dir=settings.chroma_persist_dir,
            embedding_model=settings.embedding_model
        )

        # Warm up
        retriever.search("chicken", k=10)

        # Measure performance
        start = time.time()
        results = retriever.search("pasta tomato basil", k=30)
        elapsed_ms = (time.time() - start) * 1000

        assert len(results) == 30
        # Should still be fast (target: <200ms)
        assert elapsed_ms < 500, f"Retrieval took {elapsed_ms:.0f}ms (target <200ms, allow 500ms)"

        print(f"\nRetrieval performance: {elapsed_ms:.0f}ms for k=30")


class TestPhase3BackwardsCompatibility:
    """Test that Phase 3 additions don't break existing code."""

    def test_retrieval_result_backwards_compatible(self):
        """Test that RetrievalResult works with Phase 3 code."""
        # Old code should still work
        result = RetrievalResult(
            recipe_id="123",
            title="Test Recipe",
            score=0.95,
            rating_avg=4.5,
            rating_count=100,
            minutes=30
        )

        # All fields accessible
        assert result.recipe_id == "123"
        assert result.title == "Test Recipe"
        assert result.score == 0.95
        assert result.rating_avg == 4.5
        assert result.rating_count == 100
        assert result.minutes == 30

    def test_retriever_import_still_works(self):
        """Test that old import paths still work."""
        # Direct import
        from src.retrieval.retriever import RecipeRetriever
        assert RecipeRetriever is not None

        # Package import (Phase 3 added this)
        from src.retrieval import RecipeRetriever as PackageRetriever
        assert PackageRetriever is not None

        # Should be the same class
        assert RecipeRetriever is PackageRetriever

    @pytest.mark.skipif(
        not Path("data/chroma").exists(),
        reason="Vector store not built"
    )
    def test_search_without_reranking_unchanged(self):
        """Test that search without --rerank flag produces same results."""
        from src.retrieval.retriever import RecipeRetriever
        from src.app.settings import settings

        retriever = RecipeRetriever(
            chroma_dir=settings.chroma_persist_dir,
            embedding_model=settings.embedding_model
        )

        # Search multiple times - should be deterministic
        results1 = retriever.search("chicken soup", k=20)
        results2 = retriever.search("chicken soup", k=20)

        assert len(results1) == len(results2) == 20

        # Results should be identical (same order, same scores)
        for r1, r2 in zip(results1, results2):
            assert r1.recipe_id == r2.recipe_id
            assert r1.title == r2.title
            assert abs(r1.score - r2.score) < 0.0001  # Floating point tolerance


class TestEndToEndRegression:
    """End-to-end regression tests for complete workflows."""

    @pytest.mark.skipif(
        not (Path("data/chroma").exists() and Path("data/sqlite/recipes.db").exists()),
        reason="Full pipeline not built"
    )
    def test_complete_search_workflow(self):
        """Test complete search workflow (Phase 1 + 2 + 3)."""
        from src.retrieval.retriever import RecipeRetriever
        from src.retrieval.rerank import RecipeReranker
        from src.retrieval.recipe_cards import RecipeCardBuilder
        from src.app.settings import settings

        # Step 1: Vector retrieval (Phase 2)
        retriever = RecipeRetriever(
            chroma_dir=settings.chroma_persist_dir,
            embedding_model=settings.embedding_model
        )
        vector_results = retriever.search("chicken tomato spicy", k=100)
        assert len(vector_results) == 100

        # Step 2: Reranking (Phase 3)
        reranker = RecipeReranker(model_name=settings.reranker_model)
        reranked_results = reranker.rerank("chicken tomato spicy", vector_results, top_k=20)
        assert len(reranked_results) == 20

        # Step 3: Card building (Phase 3)
        builder = RecipeCardBuilder(db_path=settings.sqlite_db_path)
        cards = builder.build_cards(reranked_results[:6], "chicken tomato spicy")
        assert len(cards) == 6

        # Verify all components worked together
        for card in cards:
            assert card.recipe_id
            assert card.title
            assert card.one_sentence_summary
            assert card.why_match
            assert isinstance(card.key_ingredients, list)

        print(f"\nEnd-to-end test: {len(vector_results)} -> {len(reranked_results)} -> {len(cards)} cards")

    @pytest.mark.skipif(
        not Path("data/chroma").exists(),
        reason="Vector store not built"
    )
    def test_diverse_query_regression(self):
        """Test that diverse query types still work."""
        from src.retrieval.retriever import RecipeRetriever
        from src.app.settings import settings

        retriever = RecipeRetriever(
            chroma_dir=settings.chroma_persist_dir,
            embedding_model=settings.embedding_model
        )

        test_queries = [
            "quick vegetarian pasta",
            "chocolate dessert",
            "healthy chicken salad",
            "spicy asian noodles",
            "comfort food soup",
            "italian dinner"
        ]

        for query in test_queries:
            results = retriever.search(query, k=10)
            assert len(results) == 10, f"Query '{query}' failed"
            assert all(r.score > 0 for r in results), f"Query '{query}' has invalid scores"

        print(f"\nTested {len(test_queries)} diverse queries - all passed")

"""Tests for recipe retrieval and embeddings."""

import pytest
from pathlib import Path
from src.domain.models import Recipe, RetrievalResult
from src.ingest.build_embeddings import build_embedding_text


class TestEmbeddingText:
    """Test embedding text generation."""

    def test_build_embedding_text_full(self):
        """Test building embedding text with all fields."""
        recipe = Recipe(
            recipe_id="123",
            title="Spicy Chicken Tacos",
            ingredients=["1 lb chicken breast", "2 tomatoes", "1 onion"],
            ingredients_normalized=["chicken", "tomato", "onion"],
            instructions=["Cook chicken", "Add vegetables"],
            tags=["mexican", "dinner", "spicy"],
            rating_avg=4.5,
            rating_count=100,
            minutes=30
        )

        text = build_embedding_text(recipe)

        assert "Spicy Chicken Tacos" in text
        assert "mexican" in text
        assert "dinner" in text
        assert "spicy" in text
        assert "chicken" in text
        assert "tomato" in text
        assert "onion" in text

    def test_build_embedding_text_no_tags(self):
        """Test building embedding text without tags."""
        recipe = Recipe(
            recipe_id="123",
            title="Simple Recipe",
            ingredients_normalized=["flour", "water"],
            tags=[]
        )

        text = build_embedding_text(recipe)

        assert "Simple Recipe" in text
        assert "flour" in text
        assert "water" in text
        # Should not have empty tags section
        assert text.count(".") <= 2

    def test_build_embedding_text_no_ingredients(self):
        """Test building embedding text without ingredients."""
        recipe = Recipe(
            recipe_id="123",
            title="Test Recipe",
            ingredients_normalized=[],
            tags=["test"]
        )

        text = build_embedding_text(recipe)

        assert "Test Recipe" in text
        assert "test" in text


@pytest.mark.skipif(
    not Path("data/chroma").exists(),
    reason="Vector store not built (run 'ingest embed' first)"
)
class TestRetriever:
    """Integration tests for recipe retrieval (requires vector store)."""

    @pytest.fixture
    def retriever(self):
        """Create a retriever instance."""
        from src.retrieval.retriever import RecipeRetriever
        from src.app.settings import settings

        return RecipeRetriever(
            chroma_dir=settings.chroma_persist_dir,
            embedding_model=settings.embedding_model
        )

    def test_retriever_returns_results(self, retriever):
        """Test that retriever returns results for a basic query."""
        results = retriever.search("chicken", k=10)

        assert len(results) > 0
        assert all(isinstance(r, RetrievalResult) for r in results)
        assert all(r.recipe_id for r in results)
        assert all(r.title for r in results)
        assert all(0 <= r.score <= 1 for r in results)

    def test_retriever_relevance(self, retriever):
        """Test that 'chicken tomato' returns relevant chicken recipes."""
        results = retriever.search("chicken tomato", k=10)

        assert len(results) > 0

        # Check that top results likely contain chicken-related recipes
        top_results_text = " ".join([r.title.lower() for r in results[:5]])
        assert "chicken" in top_results_text or "poultry" in top_results_text

    def test_retriever_performance(self, retriever):
        """Test that search completes within performance target (<200ms)."""
        import time

        start = time.time()
        results = retriever.search("spicy chicken", k=30)
        elapsed_ms = (time.time() - start) * 1000

        assert len(results) > 0
        assert elapsed_ms < 200, f"Search took {elapsed_ms:.0f}ms (target <200ms)"

    def test_retriever_with_rating_filter(self, retriever):
        """Test retrieval with minimum rating filter."""
        results = retriever.search_with_filters(
            query="pasta",
            k=10,
            min_rating=4.0
        )

        # If results are returned, they should all have rating >= 4.0
        for result in results:
            if result.rating_avg is not None:
                assert result.rating_avg >= 4.0

    def test_retriever_with_time_filter(self, retriever):
        """Test retrieval with maximum time filter."""
        results = retriever.search_with_filters(
            query="quick dinner",
            k=10,
            max_minutes=30
        )

        # If results are returned, they should all have time <= 30 minutes
        for result in results:
            if result.minutes is not None:
                assert result.minutes <= 30

    def test_retriever_spicy_query(self, retriever):
        """Test the exit criteria query: 'chicken tomato spicy'."""
        import time

        start = time.time()
        results = retriever.search("chicken tomato spicy", k=10)
        elapsed_ms = (time.time() - start) * 1000

        # Should return results
        assert len(results) > 0

        # Should complete quickly
        assert elapsed_ms < 200, f"Search took {elapsed_ms:.0f}ms (target <200ms)"

        # Top results should be relevant
        print("\nTop 5 results for 'chicken tomato spicy':")
        for i, result in enumerate(results[:5], 1):
            print(f"{i}. {result.title} (score: {result.score:.3f})")

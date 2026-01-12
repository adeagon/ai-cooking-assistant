"""Tests for cross-encoder reranking."""

import pytest
from pathlib import Path
from src.domain.models import RetrievalResult


class TestRerankerUnit:
    """Unit tests for RecipeReranker (no external dependencies)."""

    def test_rerank_empty_candidates(self):
        """Test reranking with empty candidate list."""
        from src.retrieval.rerank import RecipeReranker
        reranker = RecipeReranker()
        results = reranker.rerank("chicken", [], top_k=10)
        assert results == []

    def test_rerank_preserves_metadata(self):
        """Test that reranking preserves recipe metadata."""
        # Create mock candidates
        candidates = [
            RetrievalResult(
                recipe_id="123",
                title="Chicken Soup",
                score=0.8,
                rating_avg=4.5,
                rating_count=100,
                minutes=30
            )
        ]

        from src.retrieval.rerank import RecipeReranker
        reranker = RecipeReranker()
        results = reranker.rerank("chicken soup", candidates, top_k=1)

        assert len(results) == 1
        assert results[0].recipe_id == "123"
        assert results[0].rating_avg == 4.5
        assert results[0].rating_count == 100
        assert results[0].minutes == 30

    def test_rerank_updates_scores(self):
        """Test that scores are updated after reranking."""
        candidates = [
            RetrievalResult(
                recipe_id="1",
                title="Chicken Soup",
                score=0.9,
                rating_avg=4.0,
                rating_count=10,
                minutes=30
            ),
            RetrievalResult(
                recipe_id="2",
                title="Beef Stew",
                score=0.85,
                rating_avg=4.0,
                rating_count=10,
                minutes=60
            ),
        ]

        from src.retrieval.rerank import RecipeReranker
        reranker = RecipeReranker()
        results = reranker.rerank("chicken", candidates, top_k=2)

        # Scores should be different from original vector scores
        assert results[0].score != 0.9 or results[1].score != 0.85

    def test_rerank_respects_top_k(self):
        """Test that top_k limits output size."""
        candidates = [
            RetrievalResult(
                recipe_id=str(i),
                title=f"Recipe {i}",
                score=0.9 - i * 0.01,
                rating_avg=4.0,
                rating_count=10,
                minutes=30
            )
            for i in range(20)
        ]

        from src.retrieval.rerank import RecipeReranker
        reranker = RecipeReranker()
        results = reranker.rerank("recipe", candidates, top_k=5)

        assert len(results) == 5

    def test_rerank_fewer_candidates_than_top_k(self):
        """Test reranking when candidates < top_k."""
        candidates = [
            RetrievalResult(
                recipe_id="1",
                title="Chicken Soup",
                score=0.9,
                rating_avg=4.0,
                rating_count=10,
                minutes=30
            )
        ]

        from src.retrieval.rerank import RecipeReranker
        reranker = RecipeReranker()
        results = reranker.rerank("chicken", candidates, top_k=10)

        assert len(results) == 1

    def test_build_pairs(self):
        """Test query-document pair building."""
        candidates = [
            RetrievalResult(
                recipe_id="1",
                title="Chicken Soup",
                score=0.9,
                rating_avg=4.0,
                rating_count=10,
                minutes=30
            ),
            RetrievalResult(
                recipe_id="2",
                title="Beef Stew",
                score=0.85,
                rating_avg=4.0,
                rating_count=10,
                minutes=60
            ),
        ]

        from src.retrieval.rerank import RecipeReranker
        reranker = RecipeReranker()
        pairs = reranker._build_pairs("chicken soup", candidates)

        assert len(pairs) == 2
        assert pairs[0] == ("chicken soup", "Chicken Soup")
        assert pairs[1] == ("chicken soup", "Beef Stew")


@pytest.mark.skipif(
    not Path("data/chroma").exists(),
    reason="Vector store not built (run 'ingest embed' first)"
)
class TestRerankerIntegration:
    """Integration tests for reranking with real data."""

    @pytest.fixture
    def retriever(self):
        from src.retrieval.retriever import RecipeRetriever
        from src.app.settings import settings
        return RecipeRetriever(
            chroma_dir=settings.chroma_persist_dir,
            embedding_model=settings.embedding_model
        )

    @pytest.fixture
    def reranker(self):
        from src.retrieval.rerank import RecipeReranker
        from src.app.settings import settings
        return RecipeReranker(model_name=settings.reranker_model)

    def test_rerank_improves_relevance(self, retriever, reranker):
        """Test that reranking improves result relevance."""
        query = "spicy chicken tacos"

        # Get initial results
        candidates = retriever.search(query, k=100)

        # Rerank
        reranked = reranker.rerank(query, candidates, top_k=20)

        assert len(reranked) == 20

        # Top result should have high relevance
        assert reranked[0].score > 0  # Cross-encoder gives positive scores for good matches

        # Check that results are sorted by score
        scores = [r.score for r in reranked]
        assert scores == sorted(scores, reverse=True)

    def test_rerank_performance(self, retriever, reranker):
        """Test reranking performance (should complete in reasonable time)."""
        import time

        candidates = retriever.search("chicken tomato spicy", k=100)

        start = time.time()
        reranked = reranker.rerank("chicken tomato spicy", candidates, top_k=20)
        elapsed_ms = (time.time() - start) * 1000

        assert len(reranked) == 20
        # Reranking 100 items should complete in under 1000ms
        assert elapsed_ms < 1000, f"Reranking took {elapsed_ms:.0f}ms (target <1000ms)"

        print(f"\nReranking 100 -> 20 took {elapsed_ms:.0f}ms")

    def test_rerank_diverse_queries(self, retriever, reranker):
        """Test reranking with diverse query types."""
        queries = [
            "quick vegetarian pasta",
            "chocolate dessert",
            "healthy chicken salad",
            "spicy asian noodles"
        ]

        for query in queries:
            candidates = retriever.search(query, k=50)
            reranked = reranker.rerank(query, candidates, top_k=10)

            assert len(reranked) == 10
            # Scores should be sorted
            scores = [r.score for r in reranked]
            assert scores == sorted(scores, reverse=True)

            print(f"\n{query}: top score = {reranked[0].score:.3f}, top title = {reranked[0].title}")

    def test_rerank_preserves_all_metadata(self, retriever, reranker):
        """Test that all metadata is preserved through reranking."""
        candidates = retriever.search("chicken soup", k=30)

        reranked = reranker.rerank("chicken soup", candidates, top_k=10)

        for result in reranked:
            # All fields should be present
            assert result.recipe_id
            assert result.title
            assert isinstance(result.score, float)
            # rating_avg, rating_count, minutes can be None

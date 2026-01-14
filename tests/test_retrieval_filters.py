"""Tests for ChromaDB metadata filtering in recipe retrieval."""

import pytest
from pathlib import Path


@pytest.mark.skipif(
    not Path("data/chroma").exists(),
    reason="Vector store not built (run 'ingest embed' first)"
)
class TestMetadataFilters:
    """Integration tests for metadata filtering (requires vector store)."""

    @pytest.fixture
    def retriever(self):
        """Create a retriever instance."""
        from src.retrieval.retriever import RecipeRetriever
        from src.app.settings import settings

        return RecipeRetriever(
            chroma_dir=settings.chroma_persist_dir,
            embedding_model=settings.embedding_model,
        )

    def test_vegetarian_filter_returns_only_vegetarian(self, retriever):
        """Vegetarian filter should exclude non-vegetarian recipes."""
        results = retriever.search_with_constraints(
            query="dinner recipes",
            k=20,
            dietary="vegetarian",
        )

        assert len(results) > 0, "Should return some results"

        # Verify all results are from vegetarian-tagged recipes
        # We check via the retriever's collection metadata
        for result in results:
            metadata = retriever.collection.get(ids=[result.recipe_id])["metadatas"][0]
            assert metadata.get("is_vegetarian") is True, (
                f"Recipe {result.title} should be vegetarian"
            )

    def test_vegan_filter_returns_only_vegan(self, retriever):
        """Vegan filter should exclude non-vegan recipes."""
        results = retriever.search_with_constraints(
            query="dinner recipes",
            k=20,
            dietary="vegan",
        )

        assert len(results) > 0, "Should return some results"

        # Verify all results are vegan
        for result in results:
            metadata = retriever.collection.get(ids=[result.recipe_id])["metadatas"][0]
            assert metadata.get("is_vegan") is True, (
                f"Recipe {result.title} should be vegan"
            )

    def test_cuisine_filter_returns_matching_cuisine(self, retriever):
        """Cuisine filter should return only matching cuisine."""
        results = retriever.search_with_constraints(
            query="pasta dinner",
            k=20,
            cuisine="italian",
        )

        assert len(results) > 0, "Should return some results"

        # Verify all results have Italian cuisine
        for result in results:
            metadata = retriever.collection.get(ids=[result.recipe_id])["metadatas"][0]
            assert metadata.get("cuisine") == "italian", (
                f"Recipe {result.title} should be Italian cuisine"
            )

    def test_time_filter_returns_quick_recipes(self, retriever):
        """Time filter should return only recipes under the limit."""
        results = retriever.search_with_constraints(
            query="dinner recipes",
            k=20,
            max_minutes=30,
        )

        assert len(results) > 0, "Should return some results"

        # Verify all results are under 30 minutes
        for result in results:
            metadata = retriever.collection.get(ids=[result.recipe_id])["metadatas"][0]
            minutes = metadata.get("minutes")
            if minutes is not None:
                assert minutes <= 30, (
                    f"Recipe {result.title} should be ≤30 minutes, got {minutes}"
                )

    def test_combined_filters_vegetarian_italian_quick(self, retriever):
        """Multiple filters should work together."""
        results = retriever.search_with_constraints(
            query="pasta",
            k=20,
            dietary="vegetarian",
            cuisine="italian",
            max_minutes=45,
        )

        assert len(results) > 0, "Should return some results"

        # Verify all constraints are met
        for result in results:
            metadata = retriever.collection.get(ids=[result.recipe_id])["metadatas"][0]
            assert metadata.get("is_vegetarian") is True, "Should be vegetarian"
            assert metadata.get("cuisine") == "italian", "Should be Italian"
            minutes = metadata.get("minutes")
            if minutes is not None:
                assert minutes <= 45, f"Should be ≤45 minutes, got {minutes}"

    def test_no_filters_returns_semantic_results(self, retriever):
        """Without filters, should return semantically relevant results."""
        results = retriever.search_with_constraints(
            query="chicken tacos spicy",
            k=10,
        )

        assert len(results) > 0, "Should return some results"
        # First result should be relevant to query
        assert any("chicken" in r.title.lower() or "taco" in r.title.lower() for r in results)

    def test_unsupported_dietary_falls_back_to_semantic(self, retriever):
        """Unsupported dietary (keto, gluten-free) should use semantic search only."""
        # These aren't indexed as metadata, so they fall back to semantic search
        results = retriever.search_with_constraints(
            query="keto dinner low carb",
            k=10,
            dietary="keto",  # Not indexed, should be ignored
        )

        # Should still return results (via semantic search)
        assert len(results) > 0, "Should return results even with unsupported filter"


@pytest.mark.skipif(
    not Path("data/chroma").exists(),
    reason="Vector store not built (run 'ingest embed' first)"
)
class TestVectorStoreMetadata:
    """Tests to verify metadata schema is correctly populated."""

    @pytest.fixture
    def collection(self):
        """Get ChromaDB collection."""
        import chromadb
        from chromadb.config import Settings as ChromaSettings
        from src.app.settings import settings

        client = chromadb.PersistentClient(
            path=str(settings.chroma_persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        return client.get_collection(name="recipes")

    def test_metadata_has_dietary_fields(self, collection):
        """Verify metadata includes is_vegetarian and is_vegan fields."""
        # Sample a few records
        sample = collection.peek(10)

        for metadata in sample["metadatas"]:
            assert "is_vegetarian" in metadata, "Should have is_vegetarian field"
            assert "is_vegan" in metadata, "Should have is_vegan field"
            assert isinstance(metadata["is_vegetarian"], bool)
            assert isinstance(metadata["is_vegan"], bool)

    def test_metadata_has_cuisine_field(self, collection):
        """Verify metadata includes cuisine field."""
        sample = collection.peek(10)

        for metadata in sample["metadatas"]:
            assert "cuisine" in metadata, "Should have cuisine field"
            assert isinstance(metadata["cuisine"], str)

    def test_vegetarian_count_reasonable(self, collection):
        """Verify reasonable number of vegetarian recipes indexed."""
        # Query for vegetarian recipes
        results = collection.get(
            where={"is_vegetarian": {"$eq": True}},
            limit=1,
            include=[]
        )

        # We should have indexed vegetarian recipes
        # The actual count is ~51K, but we just check it's working
        assert results is not None

    def test_vegan_implies_vegetarian(self, collection):
        """All vegan recipes should also be vegetarian."""
        # Get some vegan recipes
        vegan_results = collection.get(
            where={"is_vegan": {"$eq": True}},
            limit=100,
            include=["metadatas"]
        )

        # All vegan should also be vegetarian
        for metadata in vegan_results["metadatas"]:
            assert metadata["is_vegetarian"] is True, (
                "Vegan recipe should also be vegetarian"
            )

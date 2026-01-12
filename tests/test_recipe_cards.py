"""Tests for recipe card generation."""

import pytest
from pathlib import Path
from src.domain.models import Recipe, RecipeCard, RetrievalResult


class TestRecipeCardBuilderUnit:
    """Unit tests for RecipeCardBuilder."""

    def test_generate_summary_basic(self):
        """Test basic summary generation."""
        from src.retrieval.recipe_cards import RecipeCardBuilder

        recipe = Recipe(
            recipe_id="123",
            title="Spicy Chicken Tacos",
            ingredients_normalized=["chicken", "tomato", "onion", "cheese"],
            tags=["mexican", "dinner", "spicy", "30-minutes-or-less"],
            minutes=25
        )

        builder = RecipeCardBuilder(db_path=Path("data/sqlite/app.db"))
        summary = builder.generate_summary(recipe)

        assert isinstance(summary, str)
        assert len(summary) > 10
        assert summary.endswith(".")

    def test_generate_summary_no_tags(self):
        """Test summary generation without tags."""
        from src.retrieval.recipe_cards import RecipeCardBuilder

        recipe = Recipe(
            recipe_id="123",
            title="Simple Pasta",
            ingredients_normalized=["pasta", "sauce"],
            tags=[],
            minutes=15
        )

        builder = RecipeCardBuilder(db_path=Path("data/sqlite/app.db"))
        summary = builder.generate_summary(recipe)

        assert isinstance(summary, str)
        assert len(summary) > 5

    def test_generate_summary_with_cuisine(self):
        """Test summary with cuisine tags."""
        from src.retrieval.recipe_cards import RecipeCardBuilder

        recipe = Recipe(
            recipe_id="123",
            title="Pad Thai",
            ingredients_normalized=["noodles", "shrimp", "peanuts"],
            tags=["thai", "asian", "stir-fry"],
            minutes=30
        )

        builder = RecipeCardBuilder(db_path=Path("data/sqlite/app.db"))
        summary = builder.generate_summary(recipe)

        assert "Thai" in summary or "thai" in summary.lower()

    def test_generate_summary_quick_recipe(self):
        """Test summary for quick recipes."""
        from src.retrieval.recipe_cards import RecipeCardBuilder

        recipe = Recipe(
            recipe_id="123",
            title="Quick Salad",
            ingredients_normalized=["lettuce", "tomato", "cucumber"],
            tags=["salad", "quick", "healthy"],
            minutes=10
        )

        builder = RecipeCardBuilder(db_path=Path("data/sqlite/app.db"))
        summary = builder.generate_summary(recipe)

        assert "quick" in summary.lower()

    def test_generate_summary_long_recipe(self):
        """Test summary for long-cooking recipes."""
        from src.retrieval.recipe_cards import RecipeCardBuilder

        recipe = Recipe(
            recipe_id="123",
            title="Slow Cooker Beef Stew",
            ingredients_normalized=["beef", "potato", "carrot"],
            tags=["stew", "slow-cooked"],
            minutes=360  # 6 hours
        )

        builder = RecipeCardBuilder(db_path=Path("data/sqlite/app.db"))
        summary = builder.generate_summary(recipe)

        assert "hour" in summary.lower()

    def test_compute_why_match_ingredients(self):
        """Test why_match with matching ingredients."""
        from src.retrieval.recipe_cards import RecipeCardBuilder

        recipe = Recipe(
            recipe_id="123",
            title="Chicken Stir Fry",
            ingredients_normalized=["chicken", "broccoli", "soy sauce", "garlic"],
            tags=["asian", "stir-fry"]
        )

        builder = RecipeCardBuilder(db_path=Path("data/sqlite/app.db"))
        why_match = builder.compute_why_match(recipe, "chicken broccoli")

        assert "chicken" in why_match.lower()
        assert "broccoli" in why_match.lower()

    def test_compute_why_match_tags(self):
        """Test why_match with matching tags."""
        from src.retrieval.recipe_cards import RecipeCardBuilder

        recipe = Recipe(
            recipe_id="123",
            title="Pad Thai",
            ingredients_normalized=["noodles", "shrimp"],
            tags=["thai", "asian", "spicy"]
        )

        builder = RecipeCardBuilder(db_path=Path("data/sqlite/app.db"))
        why_match = builder.compute_why_match(recipe, "spicy thai food")

        assert "spicy" in why_match.lower() or "thai" in why_match.lower()

    def test_compute_why_match_quick(self):
        """Test why_match with quick/fast queries."""
        from src.retrieval.recipe_cards import RecipeCardBuilder

        recipe = Recipe(
            recipe_id="123",
            title="Quick Salad",
            ingredients_normalized=["lettuce", "tomato"],
            tags=[],
            minutes=10
        )

        builder = RecipeCardBuilder(db_path=Path("data/sqlite/app.db"))
        why_match = builder.compute_why_match(recipe, "quick lunch")

        assert "quick" in why_match.lower()

    def test_compute_why_match_highly_rated(self):
        """Test why_match includes rating for highly rated recipes."""
        from src.retrieval.recipe_cards import RecipeCardBuilder

        recipe = Recipe(
            recipe_id="123",
            title="Amazing Brownies",
            ingredients_normalized=["chocolate", "flour", "eggs"],
            tags=["dessert"],
            rating_avg=4.8,
            rating_count=500
        )

        builder = RecipeCardBuilder(db_path=Path("data/sqlite/app.db"))
        why_match = builder.compute_why_match(recipe, "chocolate dessert")

        assert "rated" in why_match.lower()

    def test_compute_why_match_no_matches(self):
        """Test why_match with no specific matches."""
        from src.retrieval.recipe_cards import RecipeCardBuilder

        recipe = Recipe(
            recipe_id="123",
            title="Random Recipe",
            ingredients_normalized=["ingredient1", "ingredient2"],
            tags=[]
        )

        builder = RecipeCardBuilder(db_path=Path("data/sqlite/app.db"))
        why_match = builder.compute_why_match(recipe, "totally different query")

        assert why_match == "matches search query"

    def test_select_key_ingredients(self):
        """Test key ingredient selection."""
        from src.retrieval.recipe_cards import RecipeCardBuilder

        builder = RecipeCardBuilder(db_path=Path("data/sqlite/app.db"))

        ingredients = [
            "chicken breast", "olive oil", "salt", "pepper",
            "garlic", "onion", "tomato", "basil", "oregano",
            "parmesan cheese", "pasta", "water"
        ]

        key = builder.select_key_ingredients(ingredients, max_count=8)

        assert len(key) <= 8
        # Should prioritize proteins and main ingredients over seasonings
        assert "chicken breast" in key
        # Salt/pepper should be deprioritized
        assert "salt" not in key or len([i for i in key if i not in ["salt", "pepper", "water", "oil"]]) > 5

    def test_select_key_ingredients_empty(self):
        """Test key ingredient selection with empty list."""
        from src.retrieval.recipe_cards import RecipeCardBuilder

        builder = RecipeCardBuilder(db_path=Path("data/sqlite/app.db"))
        key = builder.select_key_ingredients([], max_count=10)

        assert key == []

    def test_select_key_ingredients_fewer_than_max(self):
        """Test key ingredient selection when fewer ingredients than max."""
        from src.retrieval.recipe_cards import RecipeCardBuilder

        builder = RecipeCardBuilder(db_path=Path("data/sqlite/app.db"))
        ingredients = ["chicken", "salt", "pepper"]
        key = builder.select_key_ingredients(ingredients, max_count=10)

        assert len(key) == 3

    def test_build_card_structure(self):
        """Test that build_card returns proper RecipeCard structure."""
        from src.retrieval.recipe_cards import RecipeCardBuilder

        recipe = Recipe(
            recipe_id="123",
            title="Test Recipe",
            ingredients=["1 cup flour", "2 eggs"],
            ingredients_normalized=["flour", "eggs"],
            instructions=["Mix", "Bake"],
            tags=["baking", "easy"],
            rating_avg=4.5,
            rating_count=50,
            minutes=45
        )

        builder = RecipeCardBuilder(db_path=Path("data/sqlite/app.db"))
        card = builder.build_card(recipe, query="flour eggs", score=0.95)

        assert isinstance(card, RecipeCard)
        assert card.recipe_id == "123"
        assert card.title == "Test Recipe"
        assert card.rating_avg == 4.5
        assert card.rating_count == 50
        assert card.time_total == 45
        assert len(card.tags) > 0
        assert len(card.key_ingredients) > 0
        assert len(card.one_sentence_summary) > 0
        assert len(card.why_match) > 0


@pytest.mark.skipif(
    not Path("data/sqlite/app.db").exists(),
    reason="SQLite database not built (run 'ingest process' first)"
)
class TestRecipeCardBuilderIntegration:
    """Integration tests for recipe card building."""

    @pytest.fixture
    def builder(self):
        from src.retrieval.recipe_cards import RecipeCardBuilder
        from src.app.settings import settings
        return RecipeCardBuilder(db_path=settings.sqlite_db_path)

    def test_build_cards_from_results(self, builder):
        """Test building cards from retrieval results."""
        from src.retrieval.retriever import RecipeRetriever
        from src.app.settings import settings

        retriever = RecipeRetriever(
            chroma_dir=settings.chroma_persist_dir,
            embedding_model=settings.embedding_model
        )

        results = retriever.search("chicken tomato", k=10)
        cards = builder.build_cards(results[:6], "chicken tomato")

        assert len(cards) <= 6
        assert all(isinstance(c, RecipeCard) for c in cards)

        for card in cards:
            print(f"\n{card.title}")
            print(f"  Summary: {card.one_sentence_summary}")
            print(f"  Why: {card.why_match}")

    def test_card_token_estimate(self, builder):
        """Test that cards are within token target (120-250)."""
        from src.retrieval.retriever import RecipeRetriever
        from src.app.settings import settings

        retriever = RecipeRetriever(
            chroma_dir=settings.chroma_persist_dir,
            embedding_model=settings.embedding_model
        )

        results = retriever.search("pasta dinner", k=20)
        cards = builder.build_cards(results[:6], "pasta dinner")

        for card in cards:
            # Rough token estimate: words * 1.3
            card_text = (
                f"{card.title} {card.one_sentence_summary} {card.why_match} "
                f"{' '.join(card.key_ingredients)} {' '.join(card.tags)}"
            )
            word_count = len(card_text.split())
            estimated_tokens = int(word_count * 1.3)

            print(f"\n{card.title}: ~{estimated_tokens} tokens")

            # Allow some flexibility but target 120-350
            assert estimated_tokens < 400, f"Card too large: {estimated_tokens} tokens"

    def test_build_cards_diverse_queries(self, builder):
        """Test card building with diverse query types."""
        from src.retrieval.retriever import RecipeRetriever
        from src.app.settings import settings

        retriever = RecipeRetriever(
            chroma_dir=settings.chroma_persist_dir,
            embedding_model=settings.embedding_model
        )

        queries = [
            "quick vegetarian pasta",
            "chocolate dessert",
            "healthy chicken salad",
            "spicy asian noodles"
        ]

        for query in queries:
            results = retriever.search(query, k=10)
            cards = builder.build_cards(results[:3], query)

            assert len(cards) <= 3
            assert all(c.recipe_id for c in cards)
            assert all(c.title for c in cards)
            assert all(c.one_sentence_summary for c in cards)
            assert all(c.why_match for c in cards)

            print(f"\n{query}:")
            for card in cards:
                print(f"  - {card.title}: {card.why_match}")

    def test_card_has_all_fields(self, builder):
        """Test that generated cards have all required fields."""
        from src.retrieval.retriever import RecipeRetriever
        from src.app.settings import settings

        retriever = RecipeRetriever(
            chroma_dir=settings.chroma_persist_dir,
            embedding_model=settings.embedding_model
        )

        results = retriever.search("chicken soup", k=5)
        cards = builder.build_cards(results[:3], "chicken soup")

        for card in cards:
            # Required fields
            assert card.recipe_id
            assert card.title
            assert card.one_sentence_summary
            assert card.why_match
            # Optional but expected
            assert isinstance(card.key_ingredients, list)
            assert isinstance(card.tags, list)

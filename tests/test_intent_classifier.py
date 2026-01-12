"""Tests for intent classification."""

import pytest
from src.chains.intent_classifier import check_quick_intent, build_recommendations_context
from src.domain.models import RecipeCard, IntentClassification


class TestQuickIntentCheck:
    """Test quick pattern matching for stateless commands."""

    def test_history_match(self):
        """Test history command variations."""
        assert check_quick_intent("history") == "history"
        assert check_quick_intent("show history") == "history"
        assert check_quick_intent("what have i cooked") == "history"

    def test_box_match(self):
        """Test box command variations."""
        assert check_quick_intent("box") == "box"
        assert check_quick_intent("saved recipes") == "box"
        assert check_quick_intent("my bookmarks") == "box"

    def test_new_match(self):
        """Test new session command variations."""
        assert check_quick_intent("new") == "new"
        assert check_quick_intent("start over") == "new"
        assert check_quick_intent("reset") == "new"

    def test_prefs_match(self):
        """Test preferences command variations."""
        assert check_quick_intent("prefs") == "prefs"
        assert check_quick_intent("preferences") == "prefs"
        assert check_quick_intent("settings") == "prefs"

    def test_no_match(self):
        """Test that non-matching input returns None."""
        assert check_quick_intent("I loved that recipe") is None
        assert check_quick_intent("save that") is None
        assert check_quick_intent("quick chicken dinner") is None


class TestBuildRecommendationsContext:
    """Test building context about recent recommendations."""

    def test_empty_cards(self):
        """Test context with no recent recommendations."""
        context = build_recommendations_context([])
        assert "None (no recent recipe suggestions)" in context

    def test_single_card(self):
        """Test context with one recommendation."""
        cards = [
            RecipeCard(
                recipe_id="123",
                title="Chicken Tacos",
                tags=["mexican"],
                key_ingredients=["chicken", "tortillas"]
            )
        ]
        context = build_recommendations_context(cards)
        assert "1. Chicken Tacos" in context

    def test_multiple_cards(self):
        """Test context with multiple recommendations."""
        cards = [
            RecipeCard(
                recipe_id="123",
                title="Chicken Tacos",
                tags=["mexican"],
                key_ingredients=["chicken", "tortillas"]
            ),
            RecipeCard(
                recipe_id="456",
                title="Pasta Carbonara",
                tags=["italian"],
                key_ingredients=["pasta", "eggs", "bacon"]
            )
        ]
        context = build_recommendations_context(cards)
        assert "1. Chicken Tacos" in context
        assert "2. Pasta Carbonara" in context


class TestIntentClassification:
    """Test intent classification model."""

    def test_conversation_intent(self):
        """Test conversation intent creation."""
        result = IntentClassification(
            intent="conversation",
            confidence="high",
            reasoning="User is asking a recipe query"
        )
        assert result.intent == "conversation"
        assert result.confidence == "high"
        assert result.recipe_reference is None
        assert result.rating_value is None

    def test_like_intent_with_reference(self):
        """Test like intent with recipe reference."""
        result = IntentClassification(
            intent="like",
            confidence="high",
            recipe_reference="first one",
            reasoning="Clear like action on specific recipe"
        )
        assert result.intent == "like"
        assert result.recipe_reference == "first one"

    def test_rate_intent_with_value(self):
        """Test rate intent with rating value."""
        result = IntentClassification(
            intent="rate",
            confidence="high",
            recipe_reference="2",
            rating_value=4,
            reasoning="Clear rating action with value"
        )
        assert result.intent == "rate"
        assert result.rating_value == 4
        assert result.recipe_reference == "2"

    def test_save_intent(self):
        """Test save intent."""
        result = IntentClassification(
            intent="save",
            confidence="medium",
            recipe_reference="that",
            reasoning="Save action with vague reference"
        )
        assert result.intent == "save"
        assert result.confidence == "medium"

    def test_stateless_intent(self):
        """Test stateless command intent."""
        result = IntentClassification(
            intent="history",
            confidence="high",
            reasoning="Quick pattern match for history command"
        )
        assert result.intent == "history"
        assert result.recipe_reference is None


# LLM integration tests would go here with @pytest.mark.llm decorator
# These require Ollama to be running and are more expensive

@pytest.mark.llm
class TestIntentClassificationLLM:
    """LLM integration tests for intent classification.

    These tests require Ollama to be running with the configured model.
    Run with: pytest tests/test_intent_classifier.py -v -s -m llm
    """

    @pytest.fixture
    def llm(self):
        """Create LLM client for testing."""
        from langchain_ollama import ChatOllama
        from src.app.settings import settings

        return ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=0.0  # Deterministic for testing
        )

    @pytest.fixture
    def sample_cards(self):
        """Sample recipe cards for context."""
        return [
            RecipeCard(
                recipe_id="123",
                title="Chicken Tacos",
                tags=["mexican"],
                key_ingredients=["chicken", "tortillas", "salsa"]
            ),
            RecipeCard(
                recipe_id="456",
                title="Pasta Carbonara",
                tags=["italian"],
                key_ingredients=["pasta", "eggs", "bacon"]
            )
        ]

    def test_clear_like_action(self, llm, sample_cards):
        """Test clear like action is detected."""
        from src.chains.intent_classifier import classify_intent

        result = classify_intent("I loved the first one", sample_cards, llm)

        assert result.intent == "like"
        assert result.confidence in ["high", "medium"]
        assert result.recipe_reference is not None

    def test_preference_not_like(self, llm, sample_cards):
        """Test that preference statement is not detected as like."""
        from src.chains.intent_classifier import classify_intent

        result = classify_intent("I love Italian food", sample_cards, llm)

        assert result.intent == "conversation"
        # Should be conservative - not an action on a specific recipe

    def test_rate_extracts_value(self, llm, sample_cards):
        """Test that rating value is extracted."""
        from src.chains.intent_classifier import classify_intent

        result = classify_intent("give it 4 stars", sample_cards, llm)

        assert result.intent == "rate"
        assert result.rating_value == 4

    def test_ambiguous_defaults_conversation(self, llm, sample_cards):
        """Test that ambiguous input defaults to conversation."""
        from src.chains.intent_classifier import classify_intent

        result = classify_intent("that was nice", sample_cards, llm)

        # Conservative - no clear action verb
        assert result.intent == "conversation" or result.confidence == "low"

    def test_show_recipe(self, llm, sample_cards):
        """Test show recipe intent."""
        from src.chains.intent_classifier import classify_intent

        result = classify_intent("show me the pasta recipe", sample_cards, llm)

        assert result.intent == "show"
        assert "pasta" in result.recipe_reference.lower()

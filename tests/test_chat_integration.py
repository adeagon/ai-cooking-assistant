"""Integration tests for chat chains components."""

import pytest

from src.chains.chat_chain import should_clarify
from src.chains.extractors import ConstraintExtractorChain
from src.chains.prompts import format_preferences, format_session_context
from src.domain.models import Constraints, PreferenceProfile, SessionState


class TestConstraintExtractorChain:
    """Tests for ConstraintExtractorChain integration."""

    def test_constraint_extractor_chain_invoke(self):
        """Test that ConstraintExtractorChain can be invoked."""
        input_data = {"user_input": "I have chicken and tomatoes"}

        result = ConstraintExtractorChain.invoke(input_data)

        assert "constraints" in result
        assert isinstance(result["constraints"], Constraints)
        assert "chicken" in result["constraints"].ingredients

    def test_constraint_extractor_chain_preserves_input(self):
        """Test that extractor chain preserves original input."""
        input_data = {"user_input": "quick pasta", "extra_field": "value"}

        result = ConstraintExtractorChain.invoke(input_data)

        assert "user_input" in result
        assert "extra_field" in result
        assert result["extra_field"] == "value"


class TestShouldClarifyIntegration:
    """Integration tests for should_clarify gate."""

    def test_clarify_flow_comprehensive(self):
        """Test the clarify logic with various inputs."""
        # Case 1: Empty constraints - should clarify
        input_data = {
            "constraints": Constraints(),
            "session": SessionState(),
        }
        assert should_clarify(input_data) is True

        # Case 2: With ingredients - should not clarify
        input_data = {
            "constraints": Constraints(ingredients=["chicken"]),
            "session": SessionState(),
        }
        assert should_clarify(input_data) is False

        # Case 3: With session ingredients - should not clarify
        input_data = {
            "constraints": Constraints(),
            "session": SessionState(ingredients_on_hand=["tomatoes"]),
        }
        assert should_clarify(input_data) is False

        # Case 4: With goals - should not clarify
        input_data = {
            "constraints": Constraints(goals=["healthy"]),
            "session": SessionState(),
        }
        assert should_clarify(input_data) is False

        # Case 5: With dietary restriction - should not clarify
        input_data = {
            "constraints": Constraints(dietary="vegetarian"),
            "session": SessionState(),
        }
        assert should_clarify(input_data) is False

        # Case 6: With cuisine - should not clarify
        input_data = {
            "constraints": Constraints(cuisine="italian"),
            "session": SessionState(),
        }
        assert should_clarify(input_data) is False


class TestPromptIntegration:
    """Integration tests for prompt formatting."""

    def test_format_preferences_complete(self):
        """Test formatting a complete profile."""
        profile = PreferenceProfile(
            spice_level="hot",
            diet="vegetarian",
            avoid_ingredients=["meat", "fish"],
            preferred_cuisines=["italian", "indian"],
            time_limit_default_minutes=45,
        )

        formatted = format_preferences(profile)

        assert "hot" in formatted
        assert "vegetarian" in formatted
        assert "meat" in formatted
        assert "italian" in formatted
        assert "45 minutes" in formatted

    def test_format_session_context_complete(self):
        """Test formatting a complete session."""
        session = SessionState(
            ingredients_on_hand=["chicken", "tomatoes", "garlic"],
            avoid_tonight=["fish"],
            goals=["quick", "healthy"],
            time_limit_minutes=30,
            servings=4,
        )
        rolling_summary = "ingredients: chicken; time: 30 min"

        formatted = format_session_context(session, rolling_summary)

        assert "chicken" in formatted
        assert "fish" in formatted
        assert "quick" in formatted
        assert "30 minutes" in formatted
        assert "4" in formatted
        assert "PREVIOUS DISCUSSION" in formatted


class TestConstraintExtractionFlow:
    """Test full constraint extraction flow."""

    def test_extract_and_gate(self):
        """Test extracting constraints and checking clarify gate."""
        # Input with constraints
        input_data = {"user_input": "I have chicken, want something quick"}

        # Extract constraints
        result = ConstraintExtractorChain.invoke(input_data)

        # Add empty session
        result["session"] = SessionState()

        # Check gate
        needs_clarification = should_clarify(result)

        # Should not need clarification (has ingredients)
        assert needs_clarification is False
        assert "chicken" in result["constraints"].ingredients

    def test_extract_and_gate_vague_input(self):
        """Test extracting from vague input."""
        # Vague input
        input_data = {"user_input": "What should I cook?"}

        # Extract constraints
        result = ConstraintExtractorChain.invoke(input_data)

        # Add empty session
        result["session"] = SessionState()

        # Check gate
        needs_clarification = should_clarify(result)

        # Should need clarification (no constraints)
        assert needs_clarification is True


@pytest.mark.integration
class TestEndToEndFlow:
    """End-to-end integration tests (requires database and models)."""

    @pytest.mark.skip(reason="Requires full database, vector store, and Ollama")
    def test_full_chat_flow(self):
        """Test complete chat flow from input to output.

        This would test:
        1. User input -> constraint extraction
        2. Retrieval from vector store
        3. Reranking
        4. Recipe card building
        5. LLM response generation
        6. Memory updates

        Skip for now as it requires full infrastructure.
        """
        pass

    @pytest.mark.skip(reason="Requires database and vector store")
    def test_retrieval_pipeline(self):
        """Test retrieval -> rerank -> cards pipeline.

        Skip for now as it requires database and embeddings.
        """
        pass

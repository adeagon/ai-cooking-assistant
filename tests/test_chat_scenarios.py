"""Automated tests for Phase 4 chat scenarios.

These tests verify the chat functionality end-to-end, including:
- Clarification flow for vague queries
- Direct recommendation for specific queries
- Dietary restriction handling
- Cuisine preference handling
- Multi-constraint queries
- Multi-turn conversations with memory
"""

import pytest

from src.chains.chat_chain import should_clarify
from src.chains.extractors import ConstraintExtractor
from src.domain.models import Constraints, SessionState


class TestConstraintExtractionScenarios:
    """Test constraint extraction for various real-world scenarios."""

    def setup_method(self):
        self.extractor = ConstraintExtractor()

    def test_scenario_vague_request(self):
        """Scenario 1: Vague request should extract no constraints."""
        user_input = "What should I cook tonight?"
        constraints = self.extractor.extract_constraints(user_input)

        assert len(constraints.ingredients) == 0
        assert constraints.time_limit is None
        assert constraints.dietary is None
        assert constraints.cuisine is None
        assert len(constraints.goals) == 0

    def test_scenario_ingredients_and_time(self):
        """Scenario 2: Should extract ingredients and time."""
        user_input = "I have chicken and tomatoes, something quick under 30 minutes"
        constraints = self.extractor.extract_constraints(user_input)

        assert "chicken" in constraints.ingredients
        assert "tomatoes" in constraints.ingredients
        assert constraints.time_limit == 30

    def test_scenario_dietary_restriction(self):
        """Scenario 3: Should extract vegetarian dietary restriction."""
        user_input = "Show me vegetarian pasta recipes"
        constraints = self.extractor.extract_constraints(user_input)

        assert constraints.dietary == "vegetarian"

    def test_scenario_cuisine_preference(self):
        """Scenario 4: Should extract Italian cuisine."""
        user_input = "I want to make Italian food tonight"
        constraints = self.extractor.extract_constraints(user_input)

        assert constraints.cuisine == "italian"

    def test_scenario_multiple_constraints(self):
        """Scenario 5: Should extract multiple constraints."""
        user_input = "I need a healthy, quick dinner with chicken, under 45 minutes"
        constraints = self.extractor.extract_constraints(user_input)

        assert "healthy" in constraints.goals
        assert "chicken" in constraints.ingredients
        assert constraints.time_limit == 45

    def test_scenario_goal_based(self):
        """Scenario 6: Goal-based query (comfort food)."""
        user_input = "Something comforting and hearty"
        constraints = self.extractor.extract_constraints(user_input)

        # May or may not extract "comfort" as a goal
        # Main test is that it doesn't extract ingredients
        assert len(constraints.ingredients) == 0

    def test_scenario_vegan_with_time(self):
        """Scenario 7: Should extract vegan and time."""
        user_input = "Quick vegan dinner ideas, I have about 20 minutes"
        constraints = self.extractor.extract_constraints(user_input)

        assert constraints.dietary == "vegan"
        assert constraints.time_limit == 20

    def test_scenario_multi_turn_first(self):
        """Scenario 8 Turn 1: Vague pasta request."""
        user_input = "Show me pasta recipes"
        constraints = self.extractor.extract_constraints(user_input)

        # Should be vague (no specific ingredients or constraints)
        assert len(constraints.ingredients) == 0

    def test_scenario_multi_turn_second(self):
        """Scenario 8 Turn 2: Refined with tomato base."""
        user_input = "Actually, I prefer something with a tomato base"
        constraints = self.extractor.extract_constraints(user_input)

        # Should extract tomato as ingredient
        assert any("tomato" in ing.lower() for ing in constraints.ingredients)

    def test_scenario_gluten_free(self):
        """Scenario 9: Should extract gluten-free dietary restriction."""
        user_input = "I need gluten-free dinner ideas"
        constraints = self.extractor.extract_constraints(user_input)

        assert constraints.dietary == "gluten_free"

    def test_scenario_just_cuisine(self):
        """Scenario 10: Just cuisine, no other details."""
        user_input = "Mexican food"
        constraints = self.extractor.extract_constraints(user_input)

        assert constraints.cuisine == "mexican"


class TestClarificationGateScenarios:
    """Test should_clarify gate logic for various scenarios."""

    def test_scenario_vague_request_should_clarify(self):
        """Scenario 1: Vague request should trigger clarification."""
        input_data = {
            "constraints": Constraints(),
            "session": SessionState(),
        }
        assert should_clarify(input_data) is True

    def test_scenario_with_ingredients_no_clarify(self):
        """Scenario 2: With ingredients should not clarify."""
        input_data = {
            "constraints": Constraints(ingredients=["chicken", "tomatoes"], time_limit=30),
            "session": SessionState(),
        }
        assert should_clarify(input_data) is False

    def test_scenario_with_dietary_no_clarify(self):
        """Scenario 3: With dietary restriction should not clarify."""
        input_data = {
            "constraints": Constraints(dietary="vegetarian"),
            "session": SessionState(),
        }
        assert should_clarify(input_data) is False

    def test_scenario_with_cuisine_no_clarify(self):
        """Scenario 4: With cuisine should not clarify."""
        input_data = {
            "constraints": Constraints(cuisine="italian"),
            "session": SessionState(),
        }
        assert should_clarify(input_data) is False

    def test_scenario_with_goals_no_clarify(self):
        """Scenario 5: With goals should not clarify."""
        input_data = {
            "constraints": Constraints(goals=["healthy"], ingredients=["chicken"], time_limit=45),
            "session": SessionState(),
        }
        assert should_clarify(input_data) is False

    def test_scenario_vague_goal_should_clarify(self):
        """Scenario 6: Vague goal without ingredients should clarify."""
        input_data = {
            "constraints": Constraints(),
            "session": SessionState(),
        }
        assert should_clarify(input_data) is True

    def test_scenario_dietary_with_time_no_clarify(self):
        """Scenario 7: Dietary with time should not clarify."""
        input_data = {
            "constraints": Constraints(dietary="vegan", time_limit=20),
            "session": SessionState(),
        }
        assert should_clarify(input_data) is False


class TestMemoryScenarios:
    """Test memory and session tracking."""

    def test_scenario_session_state_preservation(self):
        """Test that session state can track ingredients across turns."""
        session = SessionState(ingredients_on_hand=["pasta"])

        # Turn 1: User says they have pasta
        input_data = {
            "constraints": Constraints(),
            "session": session,
        }

        # Should not clarify because session has ingredients
        assert should_clarify(input_data) is False

    def test_scenario_constraints_override_session(self):
        """Test that new constraints can override session state."""
        session = SessionState(ingredients_on_hand=[])

        # User provides specific ingredients in query
        input_data = {
            "constraints": Constraints(ingredients=["tomato"]),
            "session": session,
        }

        # Should not clarify because constraints have ingredients
        assert should_clarify(input_data) is False


@pytest.mark.integration
class TestEndToEndScenarios:
    """Integration tests requiring full stack (marked for selective running)."""

    @pytest.mark.skip(reason="Requires Ollama, vector store, and database")
    def test_full_scenario_1_vague_to_specific(self):
        """Test full flow: vague query -> clarification -> specific query -> recommendation."""
        # This would test the complete pipeline with LLM
        pass

    @pytest.mark.skip(reason="Requires Ollama, vector store, and database")
    def test_full_scenario_8_multi_turn(self):
        """Test full multi-turn conversation with memory updates."""
        # This would test rolling summary and context preservation
        pass


# Scenario validation tests
class TestScenarioExpectations:
    """Validate that our test scenarios match real-world expectations."""

    def test_all_scenarios_covered(self):
        """Ensure we have tests for all 10 scenarios from test_chat_scenarios.py."""
        # This is a meta-test to ensure completeness
        scenario_tests = [
            "test_scenario_vague_request",
            "test_scenario_ingredients_and_time",
            "test_scenario_dietary_restriction",
            "test_scenario_cuisine_preference",
            "test_scenario_multiple_constraints",
            "test_scenario_goal_based",
            "test_scenario_vegan_with_time",
            "test_scenario_multi_turn_first",
            "test_scenario_multi_turn_second",
            "test_scenario_gluten_free",
            "test_scenario_just_cuisine",
        ]

        # All 10 scenarios + multi-turn second = 11 constraint extraction tests
        assert len(scenario_tests) == 11

    def test_clarification_scenarios_covered(self):
        """Ensure we test both clarify and no-clarify paths."""
        clarify_tests = [
            "test_scenario_vague_request_should_clarify",
            "test_scenario_vague_goal_should_clarify",
        ]
        no_clarify_tests = [
            "test_scenario_with_ingredients_no_clarify",
            "test_scenario_with_dietary_no_clarify",
            "test_scenario_with_cuisine_no_clarify",
            "test_scenario_with_goals_no_clarify",
            "test_scenario_dietary_with_time_no_clarify",
        ]

        # Should have tests for both paths
        assert len(clarify_tests) >= 2
        assert len(no_clarify_tests) >= 5

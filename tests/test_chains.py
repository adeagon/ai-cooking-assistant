"""Unit tests for LCEL chains and extractors."""

import pytest

from src.chains.extractors import ConstraintExtractor
from src.chains.chat_chain import should_clarify
from src.chains.prompts import format_preferences, format_session_context
from src.domain.models import Constraints, PreferenceProfile, SessionState


class TestConstraintExtractor:
    """Tests for ConstraintExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create ConstraintExtractor instance."""
        return ConstraintExtractor()

    def test_extract_ingredients_simple(self, extractor):
        """Test extracting ingredients from simple input."""
        constraints = extractor.extract_constraints("I have chicken and tomatoes")

        assert "chicken" in constraints.ingredients
        assert "tomatoes" in constraints.ingredients

    def test_extract_ingredients_with_using(self, extractor):
        """Test extracting ingredients with 'using' keyword."""
        constraints = extractor.extract_constraints("using garlic, onions and peppers")

        assert "garlic" in constraints.ingredients
        assert "onions" in constraints.ingredients
        assert "peppers" in constraints.ingredients

    def test_extract_time_limit_minutes(self, extractor):
        """Test extracting time limit in minutes."""
        constraints = extractor.extract_constraints("under 30 minutes")

        assert constraints.time_limit == 30

    def test_extract_time_limit_hours(self, extractor):
        """Test extracting time limit in hours."""
        constraints = extractor.extract_constraints("less than 2 hours")

        assert constraints.time_limit == 120  # 2 hours * 60

    def test_extract_quick_keyword(self, extractor):
        """Test that 'quick' sets default time limit."""
        constraints = extractor.extract_constraints("something quick")

        assert constraints.time_limit == 30  # Default for quick

    def test_extract_dietary_vegetarian(self, extractor):
        """Test extracting vegetarian dietary restriction."""
        constraints = extractor.extract_constraints("I'm vegetarian")

        assert constraints.dietary == "vegetarian"

    def test_extract_dietary_vegan(self, extractor):
        """Test extracting vegan dietary restriction."""
        constraints = extractor.extract_constraints("vegan recipes please")

        assert constraints.dietary == "vegan"

    def test_extract_dietary_keto(self, extractor):
        """Test extracting keto dietary restriction."""
        constraints = extractor.extract_constraints("looking for ketogenic meals")

        assert constraints.dietary == "keto"

    def test_extract_cuisine_italian(self, extractor):
        """Test extracting Italian cuisine."""
        constraints = extractor.extract_constraints("I want italian food")

        assert constraints.cuisine == "italian"

    def test_extract_cuisine_mexican(self, extractor):
        """Test extracting Mexican cuisine."""
        constraints = extractor.extract_constraints("something mexican")

        assert constraints.cuisine == "mexican"

    def test_extract_goal_healthy(self, extractor):
        """Test extracting healthy goal."""
        constraints = extractor.extract_constraints("I want something healthy")

        assert "healthy" in constraints.goals

    def test_extract_goal_spicy(self, extractor):
        """Test extracting spicy goal."""
        constraints = extractor.extract_constraints("make it spicy and hot")

        assert "spicy" in constraints.goals

    def test_extract_combined_constraints(self, extractor):
        """Test extracting multiple constraint types at once."""
        constraints = extractor.extract_constraints(
            "I have chicken and tomatoes, want something quick and healthy, italian style"
        )

        assert "chicken" in constraints.ingredients
        assert "tomatoes" in constraints.ingredients
        assert constraints.time_limit == 30  # quick
        assert "healthy" in constraints.goals
        assert constraints.cuisine == "italian"

    def test_extract_empty_input(self, extractor):
        """Test extracting from empty input."""
        constraints = extractor.extract_constraints("")

        assert constraints.ingredients == []
        assert constraints.time_limit is None
        assert constraints.dietary is None
        assert constraints.cuisine is None
        assert constraints.goals == []

    def test_extract_dish_tikka_masala(self, extractor):
        """Test extracting tikka masala as Indian cuisine."""
        constraints = extractor.extract_constraints("I want chicken tikka masala")

        assert constraints.cuisine == "indian"
        assert constraints.dish_name == "tikka masala"

    def test_extract_dish_pad_thai(self, extractor):
        """Test extracting pad thai as Thai cuisine."""
        constraints = extractor.extract_constraints("make me pad thai")

        assert constraints.cuisine == "thai"
        assert constraints.dish_name == "pad thai"

    def test_extract_dish_carbonara(self, extractor):
        """Test extracting carbonara as Italian cuisine."""
        constraints = extractor.extract_constraints("I'd like pasta carbonara")

        assert constraints.cuisine == "italian"
        assert constraints.dish_name == "carbonara"

    def test_extract_dish_ramen(self, extractor):
        """Test extracting ramen as Japanese cuisine."""
        constraints = extractor.extract_constraints("chicken ramen sounds good")

        assert constraints.cuisine == "japanese"
        assert constraints.dish_name == "ramen"

    def test_extract_dish_butter_chicken(self, extractor):
        """Test extracting butter chicken as Indian cuisine."""
        constraints = extractor.extract_constraints("butter chicken for dinner")

        assert constraints.cuisine == "indian"
        assert constraints.dish_name == "butter chicken"

    def test_explicit_cuisine_with_dish(self, extractor):
        """Test that explicit cuisine takes precedence."""
        constraints = extractor.extract_constraints("indian curry recipe")

        assert constraints.cuisine == "indian"
        # curry is in the dish list but "indian" is matched first by pattern

    def test_extract_cuisine_asian(self, extractor):
        """Test extracting generic 'asian' cuisine (loaded from DB)."""
        constraints = extractor.extract_constraints("I want asian food")

        assert constraints.cuisine == "asian"

    def test_extract_cuisine_korean(self, extractor):
        """Test extracting korean cuisine."""
        constraints = extractor.extract_constraints("korean style dishes")

        assert constraints.cuisine == "korean"

    def test_extract_cuisine_middle_eastern(self, extractor):
        """Test extracting hyphenated cuisine (middle-eastern)."""
        constraints = extractor.extract_constraints("middle eastern cuisine")

        assert constraints.cuisine == "middle-eastern"

    def test_extract_cuisine_greek(self, extractor):
        """Test extracting greek cuisine."""
        constraints = extractor.extract_constraints("something greek")

        assert constraints.cuisine == "greek"

    def test_extract_goal_savory(self, extractor):
        """Test extracting savory goal (from DB)."""
        constraints = extractor.extract_constraints("something savory")

        assert "savory" in constraints.goals

    def test_extract_goal_sweet(self, extractor):
        """Test extracting sweet goal."""
        constraints = extractor.extract_constraints("I want something sweet")

        assert "sweet" in constraints.goals

    def test_extract_goal_fallback_light(self, extractor):
        """Test that 'light' falls back to 'low-calorie'."""
        constraints = extractor.extract_constraints("something light")

        assert "low-calorie" in constraints.goals

    def test_extract_goal_fallback_cheap(self, extractor):
        """Test that 'cheap' falls back to 'inexpensive'."""
        constraints = extractor.extract_constraints("something cheap")

        assert "inexpensive" in constraints.goals

    def test_extract_goal_fallback_hearty(self, extractor):
        """Test that 'hearty' falls back to 'comfort-food'."""
        constraints = extractor.extract_constraints("something hearty")

        assert "comfort-food" in constraints.goals

    def test_extract_combined_asian_savory(self, extractor):
        """Test extracting both asian cuisine and savory goal together."""
        constraints = extractor.extract_constraints("savory, asian")

        assert constraints.cuisine == "asian"
        assert "savory" in constraints.goals

    def test_extract_avoid_no_casseroles(self, extractor):
        """Test extracting avoid constraint with 'no' keyword."""
        constraints = extractor.extract_constraints("mexican recipes but no casseroles")

        assert "casseroles" in constraints.avoid

    def test_extract_avoid_without_cheese(self, extractor):
        """Test extracting avoid constraint with 'without' keyword."""
        constraints = extractor.extract_constraints("pasta without cheese")

        assert "cheese" in constraints.avoid

    def test_extract_avoid_but_not(self, extractor):
        """Test extracting avoid constraint with 'but not' phrase."""
        constraints = extractor.extract_constraints("show me some salads but not soups")

        assert "soups" in constraints.avoid

    def test_extract_avoid_multiple(self, extractor):
        """Test extracting multiple avoid constraints."""
        constraints = extractor.extract_constraints("no casseroles, avoid soups")

        assert "casseroles" in constraints.avoid
        assert "soups" in constraints.avoid

    def test_extract_avoid_with_other_constraints(self, extractor):
        """Test extracting avoid alongside other constraints."""
        constraints = extractor.extract_constraints(
            "quick italian dinner, no casseroles"
        )

        assert constraints.cuisine == "italian"
        assert constraints.time_limit == 30  # quick
        assert "casseroles" in constraints.avoid


class TestShouldClarify:
    """Tests for should_clarify gate function."""

    def test_clarify_when_no_constraints(self):
        """Test that clarification is needed when no constraints."""
        input_data = {
            "constraints": Constraints(),
            "session": SessionState(),
        }

        assert should_clarify(input_data) is True

    def test_no_clarify_with_ingredients(self):
        """Test that clarification is not needed with ingredients."""
        input_data = {
            "constraints": Constraints(ingredients=["chicken"]),
            "session": SessionState(),
        }

        assert should_clarify(input_data) is False

    def test_no_clarify_with_session_ingredients(self):
        """Test that session ingredients satisfy requirement."""
        input_data = {
            "constraints": Constraints(),
            "session": SessionState(ingredients_on_hand=["chicken"]),
        }

        assert should_clarify(input_data) is False

    def test_no_clarify_with_goals(self):
        """Test that goals satisfy requirement."""
        input_data = {
            "constraints": Constraints(goals=["healthy"]),
            "session": SessionState(),
        }

        assert should_clarify(input_data) is False

    def test_no_clarify_with_dietary(self):
        """Test that dietary restriction satisfies requirement."""
        input_data = {
            "constraints": Constraints(dietary="vegetarian"),
            "session": SessionState(),
        }

        assert should_clarify(input_data) is False

    def test_no_clarify_with_cuisine(self):
        """Test that cuisine satisfies requirement."""
        input_data = {
            "constraints": Constraints(cuisine="italian"),
            "session": SessionState(),
        }

        assert should_clarify(input_data) is False

    def test_clarify_with_dish_name_alone(self):
        """Test that dish_name ALONE requires clarification (to ask about meat, style, etc.)."""
        input_data = {
            "constraints": Constraints(dish_name="tikka masala"),
            "session": SessionState(),
        }

        # dish_name alone should trigger clarification to ask about meat type, style, etc.
        assert should_clarify(input_data) is True

    def test_no_clarify_with_dish_name_and_dietary(self):
        """Test that dish_name + dietary constraint satisfies requirement."""
        input_data = {
            "constraints": Constraints(dish_name="tikka masala", dietary="vegetarian"),
            "session": SessionState(),
        }

        # dish_name + dietary is sufficient - no clarification needed
        assert should_clarify(input_data) is False

    def test_no_clarify_with_dish_name_and_time(self):
        """Test that dish_name + time constraint satisfies requirement."""
        input_data = {
            "constraints": Constraints(dish_name="tikka masala", time_limit=30),
            "session": SessionState(),
        }

        # dish_name + time is sufficient - no clarification needed
        assert should_clarify(input_data) is False


class TestPromptFormatters:
    """Tests for prompt formatting functions."""

    def test_format_preferences_defaults(self):
        """Test formatting default preferences."""
        profile = PreferenceProfile()

        formatted = format_preferences(profile)

        assert "USER PREFERENCES:" in formatted
        assert "Spice level: medium" in formatted
        assert "Diet: none" in formatted

    def test_format_preferences_with_avoid(self):
        """Test formatting preferences with avoid list."""
        profile = PreferenceProfile(
            avoid_ingredients=["fish", "shellfish", "peanuts"]
        )

        formatted = format_preferences(profile)

        assert "Avoid:" in formatted
        assert "fish" in formatted

    def test_format_preferences_with_cuisines(self):
        """Test formatting preferences with preferred cuisines."""
        profile = PreferenceProfile(
            preferred_cuisines=["italian", "mexican"]
        )

        formatted = format_preferences(profile)

        assert "Preferred cuisines:" in formatted
        assert "italian" in formatted

    def test_format_session_context_empty(self):
        """Test formatting empty session context."""
        session = SessionState()

        formatted = format_session_context(session)

        assert formatted == ""

    def test_format_session_context_with_ingredients(self):
        """Test formatting session with ingredients."""
        session = SessionState(ingredients_on_hand=["chicken", "tomatoes"])

        formatted = format_session_context(session)

        assert "CURRENT SESSION:" in formatted
        assert "Ingredients on hand:" in formatted
        assert "chicken" in formatted

    def test_format_session_context_with_goals(self):
        """Test formatting session with goals."""
        session = SessionState(goals=["quick", "healthy"])

        formatted = format_session_context(session)

        assert "Goals:" in formatted
        assert "quick" in formatted

    def test_format_session_context_with_summary(self):
        """Test formatting session with rolling summary."""
        session = SessionState(ingredients_on_hand=["chicken"])
        rolling_summary = "ingredients: chicken; time: 30 min"

        formatted = format_session_context(session, rolling_summary)

        assert "PREVIOUS DISCUSSION:" in formatted
        assert "ingredients: chicken" in formatted
        assert "CURRENT SESSION:" in formatted


class TestPromptRules:
    """Tests for prompt template rules."""

    def test_clarification_prompt_requires_english(self):
        """Verify clarification prompt requires English-only responses."""
        from src.chains.prompts import CLARIFICATION_PROMPT

        # Get the system message template
        system_template = CLARIFICATION_PROMPT.messages[0].prompt.template

        assert "English" in system_template

    def test_recommendation_prompt_requires_english(self):
        """Verify recommendation prompt requires English-only responses."""
        from src.chains.prompts import RECOMMENDATION_PROMPT

        # Get the system message template
        system_template = RECOMMENDATION_PROMPT.messages[0].prompt.template

        assert "English" in system_template


class TestEmptyResponseValidation:
    """Tests for empty response validation."""

    def test_validate_response_with_valid_response(self):
        """Test that valid responses pass through unchanged."""
        from src.chains.chat_chain import _validate_response

        result = {"response": "Here are some recipes", "cards": []}
        validated = _validate_response(result)

        assert validated["response"] == "Here are some recipes"
        assert validated["cards"] == []

    def test_validate_response_with_empty_string(self):
        """Test that empty responses get fallback message."""
        from src.chains.chat_chain import _validate_response, EMPTY_RESPONSE_FALLBACK

        result = {"response": "", "cards": []}
        validated = _validate_response(result)

        assert validated["response"] == EMPTY_RESPONSE_FALLBACK
        assert validated["cards"] == []

    def test_validate_response_with_whitespace_only(self):
        """Test that whitespace-only responses get fallback message."""
        from src.chains.chat_chain import _validate_response, EMPTY_RESPONSE_FALLBACK

        result = {"response": "   \n\t  ", "cards": []}
        validated = _validate_response(result)

        assert validated["response"] == EMPTY_RESPONSE_FALLBACK

    def test_validate_response_preserves_cards(self):
        """Test that cards are preserved even with empty response."""
        from src.chains.chat_chain import _validate_response

        cards = [{"title": "Test Recipe"}]
        result = {"response": "", "cards": cards}
        validated = _validate_response(result)

        assert validated["cards"] == cards

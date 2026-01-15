"""Golden tests for meal plan constraint extraction."""

import pytest

from src.domain.models import (
    DietaryRestriction,
    ExtractionSource,
    IngredientCategory,
    PreferenceProfile,
)
from src.planning.constraint_extractor import MealPlanConstraintExtractor


@pytest.fixture
def extractor():
    """Create a constraint extractor without database (basic mode)."""
    return MealPlanConstraintExtractor(db_path=None)


class TestTimeExtraction:
    """Test time constraint extraction."""

    @pytest.mark.parametrize(
        "input,expected_time",
        [
            ("plan dinners under 30 minutes", 30),
            ("meals in 45 min", 45),
            ("quick dinners", 30),
            ("fast recipes", 30),
            ("no more than 30 minutes", 30),
            ("less than 45 min", 45),
            ("30 minute meals", 30),
        ],
    )
    def test_time_extraction(self, extractor, input, expected_time):
        """Time constraints are extracted correctly."""
        result = extractor.extract(input)
        assert result.max_prep_time == expected_time

    def test_no_time_returns_none(self, extractor):
        """No time mentioned returns None."""
        result = extractor.extract("plan some dinners for the week")
        assert result.max_prep_time is None


class TestDaysExtraction:
    """Test days extraction."""

    @pytest.mark.parametrize(
        "input,expected_days",
        [
            ("plan 5 days of dinners", 5),
            ("3 nights of meals", 3),
            ("plan dinners for the week", 7),
            ("weeknight dinners", 5),
            ("monday through friday", 5),
            ("mon-fri dinners", 5),
        ],
    )
    def test_days_extraction(self, extractor, input, expected_days):
        """Day counts are extracted correctly."""
        result = extractor.extract(input)
        assert result.days == expected_days

    def test_default_days_is_5(self, extractor):
        """Default is 5 days when not specified."""
        result = extractor.extract("plan some dinners")
        assert result.days == 5  # Default


class TestDietaryExtraction:
    """Test dietary restriction extraction."""

    @pytest.mark.parametrize(
        "input,expected_dietary",
        [
            ("vegetarian meals", DietaryRestriction.VEGETARIAN),
            ("vegan dinners", DietaryRestriction.VEGAN),
            ("pescatarian week", DietaryRestriction.PESCATARIAN),
            ("keto meals", DietaryRestriction.KETO),
            ("gluten-free dinners", DietaryRestriction.GLUTEN_FREE),
            ("gluten free recipes", DietaryRestriction.GLUTEN_FREE),
            ("dairy-free meals", DietaryRestriction.DAIRY_FREE),
        ],
    )
    def test_dietary_extraction(self, extractor, input, expected_dietary):
        """Dietary restrictions are extracted correctly."""
        result = extractor.extract(input)
        assert result.dietary == expected_dietary

    def test_no_dietary_returns_none(self, extractor):
        """No dietary mentioned uses default NONE."""
        result = extractor.extract("plan some dinners")
        assert result.dietary == DietaryRestriction.NONE

    def test_profile_dietary_fallback(self, extractor):
        """Falls back to profile dietary when not in input."""
        profile = PreferenceProfile(diet="vegetarian")
        result = extractor.extract("plan dinners", profile)
        assert result.dietary == DietaryRestriction.VEGETARIAN
        assert result.extraction_sources["dietary"].source == ExtractionSource.USER_PROFILE

    def test_input_overrides_profile(self, extractor):
        """Input dietary overrides profile."""
        profile = PreferenceProfile(diet="vegetarian")
        result = extractor.extract("plan vegan dinners", profile)
        assert result.dietary == DietaryRestriction.VEGAN
        assert result.extraction_sources["dietary"].source == ExtractionSource.RULE


class TestMealTypeExtraction:
    """Test meal type extraction."""

    @pytest.mark.parametrize(
        "input,expected_types",
        [
            ("plan dinners", ["dinner"]),
            ("plan breakfast", ["breakfast"]),
            ("plan lunch", ["lunch"]),
            ("breakfast and lunch", ["breakfast", "lunch"]),
            ("breakfast lunch and dinner", ["breakfast", "lunch", "dinner"]),
        ],
    )
    def test_meal_type_extraction(self, extractor, input, expected_types):
        """Meal types are extracted correctly."""
        result = extractor.extract(input)
        assert result.meal_types == expected_types

    def test_default_is_dinner(self, extractor):
        """Default meal type is dinner when not specified."""
        result = extractor.extract("plan some meals")
        assert result.meal_types == ["dinner"]  # Default


class TestExclusionExtraction:
    """Test exclusion extraction."""

    def test_category_exclusion(self, extractor):
        """Category exclusions are extracted correctly."""
        result = extractor.extract("plan dinners, no dairy")
        assert IngredientCategory.DAIRY in result.excluded_categories

    def test_multiple_categories(self, extractor):
        """Multiple category exclusions work."""
        result = extractor.extract("no meat and no dairy")
        assert IngredientCategory.MEAT in result.excluded_categories
        assert IngredientCategory.DAIRY in result.excluded_categories

    def test_dish_type_exclusion(self, extractor):
        """Dish type exclusions become tags."""
        result = extractor.extract("no casseroles")
        assert "casserole" in result.excluded_tags

    def test_ingredient_exclusion(self, extractor):
        """Ingredient exclusions are captured."""
        result = extractor.extract("no cilantro")
        assert "cilantro" in result.excluded_ingredients

    def test_without_pattern(self, extractor):
        """'without' pattern works."""
        result = extractor.extract("dinners without seafood")
        assert IngredientCategory.SEAFOOD in result.excluded_categories

    def test_avoid_pattern(self, extractor):
        """'avoid' pattern works."""
        result = extractor.extract("avoid pasta dishes")
        assert "pasta" in result.excluded_tags

    def test_but_not_pattern(self, extractor):
        """'but not' pattern works."""
        result = extractor.extract("chicken dinners but not soup")
        assert "soup" in result.excluded_tags


class TestFalsePositiveRegressions:
    """Regression tests for false positive cases."""

    def test_no_more_than_not_exclusion(self, extractor):
        """'no more than 30 minutes' is NOT an exclusion for 'more'."""
        result = extractor.extract("no more than 30 minutes")

        # "more" should NOT be in excluded_ingredients
        assert "more" not in result.excluded_ingredients
        assert "than" not in result.excluded_ingredients

        # Time should be extracted
        assert result.max_prep_time == 30

    def test_no_less_than_not_exclusion(self, extractor):
        """'no less than' doesn't create false positives."""
        result = extractor.extract("less than 45 minutes")

        assert "less" not in result.excluded_ingredients
        assert "than" not in result.excluded_ingredients


class TestSourceTracking:
    """Test extraction source tracking."""

    def test_tracks_rule_source(self, extractor):
        """Rule-based extractions are tracked."""
        result = extractor.extract("5 days of vegetarian dinners under 30 minutes")

        assert result.extraction_sources["days"].source == ExtractionSource.RULE
        assert result.extraction_sources["dietary"].source == ExtractionSource.RULE
        assert result.extraction_sources["max_prep_time"].source == ExtractionSource.RULE

    def test_tracks_profile_source(self, extractor):
        """Profile-based extractions are tracked."""
        profile = PreferenceProfile(diet="vegetarian")
        result = extractor.extract("plan dinners", profile)

        assert result.extraction_sources["dietary"].source == ExtractionSource.USER_PROFILE


class TestStartDateExtraction:
    """Test start date extraction."""

    def test_tomorrow(self, extractor):
        """'tomorrow' is recognized."""
        result = extractor.extract("start tomorrow")
        assert result.start_date is not None

    def test_next_week(self, extractor):
        """'next week' is recognized."""
        result = extractor.extract("plan for next week")
        assert result.start_date is not None

    def test_weekday_names(self, extractor):
        """Weekday names are recognized."""
        result = extractor.extract("starting monday")
        assert result.start_date is not None
        assert result.start_date.weekday() == 0  # Monday

    def test_no_date_returns_none(self, extractor):
        """No date mentioned returns None (uses today)."""
        result = extractor.extract("plan some dinners")
        assert result.start_date is None  # Will default to today when used


class TestComplexInputs:
    """Test complex, real-world input patterns."""

    def test_full_request(self, extractor):
        """Full meal planning request."""
        result = extractor.extract(
            "plan 5 vegetarian dinners for next week, no casseroles, under 30 minutes"
        )

        assert result.days == 5
        assert result.dietary == DietaryRestriction.VEGETARIAN
        assert "casserole" in result.excluded_tags
        assert result.max_prep_time == 30

    def test_casual_request(self, extractor):
        """Casual meal planning request."""
        result = extractor.extract("help me plan some quick weeknight dinners")

        assert result.days == 5  # weeknight
        assert result.max_prep_time == 30  # quick

    def test_allergy_focused(self, extractor):
        """Request focused on allergies."""
        result = extractor.extract("plan meals, no dairy, no nuts, no eggs")

        assert IngredientCategory.DAIRY in result.excluded_categories
        assert IngredientCategory.NUTS in result.excluded_categories
        assert IngredientCategory.EGGS in result.excluded_categories


class TestEdgeCases:
    """Test edge cases."""

    def test_empty_input(self, extractor):
        """Empty input returns defaults."""
        result = extractor.extract("")

        assert result.days == 5  # Default
        assert result.dietary == DietaryRestriction.NONE
        assert result.meal_types == ["dinner"]

    def test_gibberish_input(self, extractor):
        """Gibberish input returns defaults."""
        result = extractor.extract("asdf qwerty xyz 123")

        assert result.days == 5  # Default

    def test_conflicting_dietary(self, extractor):
        """First matching dietary wins."""
        result = extractor.extract("vegetarian vegan meals")

        # vegetarian appears first
        assert result.dietary == DietaryRestriction.VEGETARIAN

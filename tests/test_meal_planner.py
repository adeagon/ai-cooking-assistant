"""Tests for the meal planner algorithm."""

from datetime import date

import pytest

from src.domain.models import (
    DietaryRestriction,
    IngredientCategory,
    MealPlanConstraints,
    PreferenceProfile,
    Recipe,
)
from src.planning.meal_planner import MealPlanner, RecipeFeatures


@pytest.fixture
def planner():
    """Create a MealPlanner instance."""
    return MealPlanner()


@pytest.fixture
def sample_recipes():
    """Create sample recipes for testing."""
    return [
        Recipe(
            recipe_id="r1",
            title="Chicken Stir Fry",
            ingredients=["chicken", "soy sauce", "garlic", "ginger", "broccoli"],
            ingredients_normalized=["chicken", "soy sauce", "garlic", "ginger", "broccoli"],
            tags=["asian", "chicken", "quick"],
            rating_avg=4.5,
            minutes=30,
        ),
        Recipe(
            recipe_id="r2",
            title="Garlic Chicken Pasta",
            ingredients=["chicken", "garlic", "pasta", "parmesan", "olive oil"],
            ingredients_normalized=["chicken", "garlic", "pasta", "parmesan", "olive oil"],
            tags=["italian", "chicken", "pasta"],
            rating_avg=4.3,
            minutes=40,
        ),
        Recipe(
            recipe_id="r3",
            title="Beef Tacos",
            ingredients=["ground beef", "tortilla", "cheese", "lettuce", "tomato"],
            ingredients_normalized=["ground beef", "tortilla", "cheese", "lettuce", "tomato"],
            tags=["mexican", "beef", "quick"],
            rating_avg=4.2,
            minutes=25,
        ),
        Recipe(
            recipe_id="r4",
            title="Salmon with Vegetables",
            ingredients=["salmon", "broccoli", "garlic", "lemon", "olive oil"],
            ingredients_normalized=["salmon", "broccoli", "garlic", "lemon", "olive oil"],
            tags=["seafood", "healthy"],
            rating_avg=4.6,
            minutes=35,
        ),
        Recipe(
            recipe_id="r5",
            title="Vegetarian Pasta Primavera",
            ingredients=["pasta", "tomato", "garlic", "basil", "olive oil"],
            ingredients_normalized=["pasta", "tomato", "garlic", "basil", "olive oil"],
            tags=["italian", "vegetarian", "pasta"],
            rating_avg=4.1,
            minutes=30,
        ),
        Recipe(
            recipe_id="r6",
            title="Thai Chicken Curry",
            ingredients=["chicken", "coconut milk", "curry paste", "garlic", "vegetables"],
            ingredients_normalized=["chicken", "coconut milk", "curry paste", "garlic", "vegetables"],
            tags=["thai", "chicken", "curry"],
            rating_avg=4.4,
            minutes=45,
        ),
        Recipe(
            recipe_id="r7",
            title="Beef Stew",
            ingredients=["beef", "potato", "carrot", "onion", "beef broth"],
            ingredients_normalized=["beef", "potato", "carrot", "onion", "beef broth"],
            tags=["american", "beef", "comfort-food"],
            rating_avg=4.3,
            minutes=120,
        ),
    ]


@pytest.fixture
def default_constraints():
    """Create default constraints for testing."""
    return MealPlanConstraints(
        days=5,
        start_date=date(2024, 1, 1),
        meal_types=["dinner"],
        dietary=DietaryRestriction.NONE,
        max_prep_time=None,
        prefer_recipe_box=True,
        ingredient_overlap_weight=0.3,
        max_same_protein=2,
        max_same_cuisine=2,
    )


class TestBasicPlanGeneration:
    """Test basic plan generation."""

    def test_generates_correct_number_of_meals(self, planner, sample_recipes, default_constraints):
        """Plan has correct number of meals."""
        meals, metrics = planner.generate_plan(
            sample_recipes, default_constraints
        )

        assert len(meals) == 5  # 5 days * 1 meal type

    def test_meals_have_required_fields(self, planner, sample_recipes, default_constraints):
        """Each meal has required fields."""
        meals, _ = planner.generate_plan(
            sample_recipes, default_constraints
        )

        for meal in meals:
            assert meal.recipe_id is not None
            assert meal.title is not None
            assert meal.day is not None
            assert meal.meal_type == "dinner"

    def test_no_duplicate_recipes(self, planner, sample_recipes, default_constraints):
        """Plan doesn't have duplicate recipes."""
        meals, _ = planner.generate_plan(
            sample_recipes, default_constraints
        )

        recipe_ids = [m.recipe_id for m in meals]
        assert len(recipe_ids) == len(set(recipe_ids))

    def test_empty_recipes_returns_empty_plan(self, planner, default_constraints):
        """Empty recipe list returns empty plan."""
        meals, metrics = planner.generate_plan([], default_constraints)

        assert len(meals) == 0
        assert metrics.unique_ingredients == 0


class TestDeterminism:
    """Test that plan generation is deterministic."""

    def test_same_input_same_output(self, planner, sample_recipes, default_constraints):
        """Same inputs produce same outputs."""
        meals1, metrics1 = planner.generate_plan(
            sample_recipes, default_constraints
        )
        meals2, metrics2 = planner.generate_plan(
            sample_recipes, default_constraints
        )
        meals3, metrics3 = planner.generate_plan(
            sample_recipes, default_constraints
        )

        ids1 = [m.recipe_id for m in meals1]
        ids2 = [m.recipe_id for m in meals2]
        ids3 = [m.recipe_id for m in meals3]

        assert ids1 == ids2 == ids3, "Beam search must be deterministic"

    def test_order_preserved(self, planner, sample_recipes, default_constraints):
        """Selection order is preserved."""
        meals, _ = planner.generate_plan(
            sample_recipes, default_constraints
        )

        # Meals should be in day order
        for i in range(1, len(meals)):
            assert meals[i].day >= meals[i - 1].day


class TestDiversityConstraints:
    """Test diversity constraints (max same protein/cuisine)."""

    def test_max_same_protein_respected(self, planner, sample_recipes, default_constraints):
        """Doesn't select more than max_same_protein of same protein."""
        default_constraints.max_same_protein = 2

        meals, metrics = planner.generate_plan(
            sample_recipes, default_constraints
        )

        # Count proteins
        proteins = list(metrics.protein_distribution.values())
        for count in proteins:
            assert count <= 2, f"Protein count {count} exceeds max_same_protein=2"

    def test_max_same_cuisine_respected(self, planner, sample_recipes, default_constraints):
        """Doesn't select more than max_same_cuisine of same cuisine."""
        default_constraints.max_same_cuisine = 2

        meals, metrics = planner.generate_plan(
            sample_recipes, default_constraints
        )

        # Count cuisines
        cuisines = list(metrics.cuisine_distribution.values())
        for count in cuisines:
            assert count <= 2, f"Cuisine count {count} exceeds max_same_cuisine=2"


class TestTimeConstraints:
    """Test time filtering."""

    def test_max_prep_time_filters_recipes(self, planner, sample_recipes, default_constraints):
        """Max prep time filters out slow recipes."""
        default_constraints.max_prep_time = 35

        meals, _ = planner.generate_plan(
            sample_recipes, default_constraints
        )

        # Should filter out r6 (45 min) and r7 (120 min)
        recipe_ids = {m.recipe_id for m in meals}
        assert "r6" not in recipe_ids  # Thai Chicken Curry - 45 min
        assert "r7" not in recipe_ids  # Beef Stew - 120 min


class TestDietaryConstraints:
    """Test dietary filtering."""

    def test_vegetarian_filter(self, planner, sample_recipes, default_constraints):
        """Vegetarian filter only selects vegetarian recipes."""
        default_constraints.dietary = DietaryRestriction.VEGETARIAN
        default_constraints.days = 1  # Only 1 day since only 1 vegetarian recipe

        meals, _ = planner.generate_plan(
            sample_recipes, default_constraints
        )

        # Only r5 (Vegetarian Pasta Primavera) should be selected
        if meals:
            assert meals[0].recipe_id == "r5"


class TestExclusionConstraints:
    """Test exclusion filtering."""

    def test_excluded_tags_filters_recipes(self, planner, sample_recipes, default_constraints):
        """Excluded tags filter out matching recipes."""
        default_constraints.excluded_tags = ["pasta"]

        meals, _ = planner.generate_plan(
            sample_recipes, default_constraints
        )

        # Should filter out r2 (Garlic Chicken Pasta) and r5 (Pasta Primavera)
        recipe_ids = {m.recipe_id for m in meals}
        assert "r2" not in recipe_ids
        assert "r5" not in recipe_ids

    def test_excluded_ingredients_filters_recipes(self, planner, sample_recipes, default_constraints):
        """Excluded ingredients filter out matching recipes."""
        default_constraints.excluded_ingredients = ["beef"]

        meals, _ = planner.generate_plan(
            sample_recipes, default_constraints
        )

        # Should filter out r3 (Beef Tacos) and r7 (Beef Stew)
        recipe_ids = {m.recipe_id for m in meals}
        assert "r3" not in recipe_ids
        assert "r7" not in recipe_ids

    def test_excluded_categories_filters_recipes(self, planner, sample_recipes, default_constraints):
        """Excluded categories filter out matching recipes."""
        default_constraints.excluded_categories = [IngredientCategory.SEAFOOD]

        meals, _ = planner.generate_plan(
            sample_recipes, default_constraints
        )

        # Should filter out r4 (Salmon with Vegetables)
        recipe_ids = {m.recipe_id for m in meals}
        assert "r4" not in recipe_ids


class TestRecipeBoxPreference:
    """Test Recipe Box preference scoring."""

    def test_box_recipes_get_priority(self, planner, sample_recipes, default_constraints):
        """Box recipes are preferred when prefer_recipe_box is True."""
        default_constraints.prefer_recipe_box = True
        box_ids = {"r1", "r2"}  # Make first 2 recipes "in box"

        meals, metrics = planner.generate_plan(
            sample_recipes, default_constraints, box_recipe_ids=box_ids
        )

        # Box recipes should appear (though not guaranteed first due to overlap scoring)
        recipe_ids = {m.recipe_id for m in meals}
        assert "r1" in recipe_ids or "r2" in recipe_ids

        # Check metrics
        assert metrics.box_recipe_count > 0


class TestMetrics:
    """Test plan metrics calculation."""

    def test_metrics_calculated(self, planner, sample_recipes, default_constraints):
        """Metrics are calculated for the plan."""
        meals, metrics = planner.generate_plan(
            sample_recipes, default_constraints
        )

        assert metrics.unique_ingredients > 0
        assert metrics.total_ingredient_uses >= metrics.unique_ingredients
        assert 0 <= metrics.overlap_ratio <= 1
        assert metrics.unique_per_meal > 0

    def test_top_shared_ingredients(self, planner, sample_recipes, default_constraints):
        """Top shared ingredients are computed."""
        meals, metrics = planner.generate_plan(
            sample_recipes, default_constraints
        )

        # Should have some shared ingredients (garlic appears in many)
        assert len(metrics.top_shared_ingredients) > 0

    def test_protein_distribution(self, planner, sample_recipes, default_constraints):
        """Protein distribution is tracked."""
        meals, metrics = planner.generate_plan(
            sample_recipes, default_constraints
        )

        # Should have protein counts
        assert isinstance(metrics.protein_distribution, dict)

    def test_cuisine_distribution(self, planner, sample_recipes, default_constraints):
        """Cuisine distribution is tracked."""
        meals, metrics = planner.generate_plan(
            sample_recipes, default_constraints
        )

        # Should have cuisine counts
        assert isinstance(metrics.cuisine_distribution, dict)


class TestIngredientOverlap:
    """Test ingredient overlap optimization."""

    def test_overlap_scoring_prefers_shared_ingredients(self, planner, default_constraints):
        """Overlap scoring favors recipes with shared ingredients."""
        # Create recipes with deliberate overlap
        recipes = [
            Recipe(
                recipe_id="a",
                title="Recipe A",
                ingredients=["garlic", "onion", "tomato"],
                ingredients_normalized=["garlic", "onion", "tomato"],
                tags=[],
                rating_avg=4.0,
            ),
            Recipe(
                recipe_id="b",
                title="Recipe B - shares garlic and onion",
                ingredients=["garlic", "onion", "bell pepper"],
                ingredients_normalized=["garlic", "onion", "bell pepper"],
                tags=[],
                rating_avg=4.0,
            ),
            Recipe(
                recipe_id="c",
                title="Recipe C - no overlap",
                ingredients=["salmon", "lemon", "dill"],
                ingredients_normalized=["salmon", "lemon", "dill"],
                tags=[],
                rating_avg=4.0,
            ),
        ]

        default_constraints.days = 2
        default_constraints.ingredient_overlap_weight = 0.5  # High overlap weight

        meals, metrics = planner.generate_plan(recipes, default_constraints)

        # With high overlap weight, A and B should be preferred together
        recipe_ids = {m.recipe_id for m in meals}
        assert "a" in recipe_ids
        assert "b" in recipe_ids


class TestMealTypes:
    """Test multiple meal types."""

    def test_multiple_meal_types(self, planner, sample_recipes, default_constraints):
        """Can plan for multiple meal types."""
        default_constraints.days = 2
        default_constraints.meal_types = ["breakfast", "lunch", "dinner"]

        meals, _ = planner.generate_plan(
            sample_recipes, default_constraints
        )

        # Should have 2 * 3 = 6 meals
        assert len(meals) == 6

        # Should have all meal types
        meal_types = {m.meal_type for m in meals}
        assert "breakfast" in meal_types
        assert "lunch" in meal_types
        assert "dinner" in meal_types


class TestSuggestSwaps:
    """Test swap suggestions."""

    def test_suggest_swaps_returns_alternatives(self, planner, sample_recipes, default_constraints):
        """Can suggest swaps for a position."""
        meals, _ = planner.generate_plan(
            sample_recipes, default_constraints
        )

        # Build features for current plan
        plan_features = []
        features_map = {}
        for recipe in sample_recipes:
            features = planner._compute_features(
                [recipe], default_constraints, set()
            )[0]
            features_map[recipe.recipe_id] = features

        plan_features = [features_map[m.recipe_id] for m in meals]
        all_candidates = [features_map[r.recipe_id] for r in sample_recipes]

        # Get swap suggestions for position 0
        swaps = planner.suggest_swaps(
            plan_features, all_candidates, default_constraints, position=0, k=3
        )

        # Should get some suggestions
        assert len(swaps) > 0

        # Each suggestion is a tuple (RecipeFeatures, score_delta)
        for candidate, delta in swaps:
            assert isinstance(candidate, RecipeFeatures)
            assert isinstance(delta, float)

    def test_suggest_swaps_respects_constraints(self, planner, sample_recipes, default_constraints):
        """Swap suggestions respect diversity constraints."""
        default_constraints.max_same_protein = 1  # Strict constraint

        meals, _ = planner.generate_plan(
            sample_recipes, default_constraints
        )

        features_map = {}
        for recipe in sample_recipes:
            features = planner._compute_features(
                [recipe], default_constraints, set()
            )[0]
            features_map[recipe.recipe_id] = features

        plan_features = [features_map[m.recipe_id] for m in meals]
        all_candidates = [features_map[r.recipe_id] for r in sample_recipes]

        # Get suggestions - should not suggest recipes that would violate constraints
        swaps = planner.suggest_swaps(
            plan_features, all_candidates, default_constraints, position=0, k=5
        )

        # All suggestions should be valid swaps
        for candidate, _ in swaps:
            assert candidate.recipe_id not in {m.recipe_id for m in meals}


class TestEdgeCases:
    """Test edge cases."""

    def test_fewer_recipes_than_meals(self, planner, default_constraints):
        """Handles fewer recipes than meals needed."""
        recipes = [
            Recipe(
                recipe_id="only_one",
                title="Only Recipe",
                ingredients=["stuff"],
                ingredients_normalized=["stuff"],
                tags=[],
                rating_avg=4.0,
            )
        ]

        default_constraints.days = 5  # Need 5 meals

        meals, _ = planner.generate_plan(recipes, default_constraints)

        # Should only have 1 meal (can't duplicate)
        assert len(meals) == 1

    def test_no_valid_candidates_after_filtering(self, planner, sample_recipes, default_constraints):
        """Handles case where all recipes are filtered out."""
        default_constraints.max_prep_time = 1  # Impossible constraint

        meals, metrics = planner.generate_plan(
            sample_recipes, default_constraints
        )

        assert len(meals) == 0
        assert metrics.unique_ingredients == 0

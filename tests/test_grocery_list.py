"""Tests for grocery list generation."""

from datetime import date, timedelta

import pytest

from src.domain.models import MealPlan, PlannedMeal, Recipe
from src.planning.grocery_list import GroceryListGenerator


@pytest.fixture
def generator():
    """Create a GroceryListGenerator instance."""
    return GroceryListGenerator()


@pytest.fixture
def sample_recipes():
    """Create sample recipes for testing."""
    return {
        "r1": Recipe(
            recipe_id="r1",
            title="Chicken Stir Fry",
            ingredients=["2 chicken breasts", "3 tablespoons soy sauce", "4 cloves garlic"],
            ingredients_normalized=["chicken breast", "soy sauce", "garlic"],
            tags=["asian", "quick"],
        ),
        "r2": Recipe(
            recipe_id="r2",
            title="Garlic Pasta",
            ingredients=["1 lb pasta", "6 cloves garlic", "olive oil", "parmesan cheese"],
            ingredients_normalized=["pasta", "garlic", "olive oil", "parmesan"],
            tags=["italian"],
        ),
        "r3": Recipe(
            recipe_id="r3",
            title="Salmon with Broccoli",
            ingredients=["1 salmon fillet", "2 cups broccoli", "lemon", "garlic"],
            ingredients_normalized=["salmon", "broccoli", "lemon", "garlic"],
            tags=["healthy"],
        ),
    }


@pytest.fixture
def sample_plan():
    """Create a sample meal plan."""
    start = date.today()
    return MealPlan(
        id=1,
        start_date=start,
        end_date=start + timedelta(days=2),
        meal_types=["dinner"],
        meals=[
            PlannedMeal(
                day=start,
                meal_type="dinner",
                recipe_id="r1",
                title="Chicken Stir Fry",
                position=0,
                source="box",
            ),
            PlannedMeal(
                day=start + timedelta(days=1),
                meal_type="dinner",
                recipe_id="r2",
                title="Garlic Pasta",
                position=0,
                source="discovery",
            ),
            PlannedMeal(
                day=start + timedelta(days=2),
                meal_type="dinner",
                recipe_id="r3",
                title="Salmon with Broccoli",
                position=0,
                source="discovery",
            ),
        ],
    )


class TestBasicGeneration:
    """Test basic grocery list generation."""

    def test_generates_list(self, generator, sample_plan, sample_recipes):
        """Can generate a grocery list from a plan."""
        grocery_list = generator.generate(sample_plan, sample_recipes)

        assert grocery_list is not None
        assert len(grocery_list.items) > 0

    def test_sets_plan_id(self, generator, sample_plan, sample_recipes):
        """Grocery list includes plan_id."""
        grocery_list = generator.generate(sample_plan, sample_recipes)

        assert grocery_list.plan_id == 1

    def test_sets_generated_at(self, generator, sample_plan, sample_recipes):
        """Grocery list includes generation timestamp."""
        grocery_list = generator.generate(sample_plan, sample_recipes)

        assert grocery_list.generated_at is not None


class TestIngredientAggregation:
    """Test ingredient aggregation across recipes."""

    def test_aggregates_duplicate_ingredients(self, generator, sample_plan, sample_recipes):
        """Duplicate ingredients across recipes are aggregated."""
        grocery_list = generator.generate(sample_plan, sample_recipes)

        # Garlic appears in all 3 recipes
        garlic_items = [item for item in grocery_list.items if "garlic" in item.normalized]
        assert len(garlic_items) == 1  # Should be consolidated

        garlic = garlic_items[0]
        assert len(garlic.recipes) == 3  # Used in all 3 recipes

    def test_tracks_recipe_usage(self, generator, sample_plan, sample_recipes):
        """Items track which recipes use them."""
        grocery_list = generator.generate(sample_plan, sample_recipes)

        pasta_items = [item for item in grocery_list.items if "pasta" in item.normalized]
        assert len(pasta_items) == 1

        pasta = pasta_items[0]
        assert "Garlic Pasta" in pasta.recipes


class TestPantryStapleExclusion:
    """Test exclusion of pantry staples."""

    def test_excludes_stop_ingredients_by_default(self, generator, sample_plan, sample_recipes):
        """Stop ingredients are excluded by default."""
        grocery_list = generator.generate(sample_plan, sample_recipes, exclude_pantry_staples=True)

        # Olive oil and soy sauce are stop ingredients
        normalized_items = {item.normalized for item in grocery_list.items}

        assert "olive_oil" not in normalized_items
        assert "soy_sauce" not in normalized_items

    def test_includes_stop_ingredients_when_disabled(self, generator, sample_plan, sample_recipes):
        """Stop ingredients are included when exclusion is disabled."""
        grocery_list = generator.generate(sample_plan, sample_recipes, exclude_pantry_staples=False)

        # Should include olive oil now
        normalized_items = {item.normalized for item in grocery_list.items}

        # Either raw or normalized form should be present
        has_olive_oil = any("olive" in n for n in normalized_items)
        assert has_olive_oil


class TestCategorization:
    """Test grocery item categorization."""

    def test_categorizes_produce(self, generator, sample_plan, sample_recipes):
        """Produce items are categorized correctly."""
        grocery_list = generator.generate(sample_plan, sample_recipes)

        broccoli_items = [item for item in grocery_list.items if "broccoli" in item.normalized]
        assert len(broccoli_items) == 1
        assert broccoli_items[0].category == "produce"

    def test_categorizes_protein(self, generator, sample_plan, sample_recipes):
        """Protein items are categorized correctly."""
        grocery_list = generator.generate(sample_plan, sample_recipes)

        chicken_items = [item for item in grocery_list.items if "chicken" in item.normalized]
        assert len(chicken_items) == 1
        assert chicken_items[0].category == "protein"

        salmon_items = [item for item in grocery_list.items if "salmon" in item.normalized]
        assert len(salmon_items) == 1
        assert salmon_items[0].category == "protein"

    def test_categorizes_dairy(self, generator, sample_plan, sample_recipes):
        """Dairy items are categorized correctly."""
        grocery_list = generator.generate(sample_plan, sample_recipes)

        cheese_items = [item for item in grocery_list.items if "parmesan" in item.normalized]
        assert len(cheese_items) == 1
        assert cheese_items[0].category == "dairy"

    def test_categorizes_pantry(self, generator, sample_plan, sample_recipes):
        """Pantry items are categorized correctly."""
        grocery_list = generator.generate(sample_plan, sample_recipes)

        pasta_items = [item for item in grocery_list.items if "pasta" in item.normalized]
        assert len(pasta_items) == 1
        assert pasta_items[0].category == "pantry"


class TestSorting:
    """Test grocery list sorting."""

    def test_sorted_by_category_then_alphabetically(self, generator, sample_plan, sample_recipes):
        """Items are sorted by category then alphabetically."""
        grocery_list = generator.generate(sample_plan, sample_recipes)

        # Check items are grouped by category
        current_category = None
        for item in grocery_list.items:
            if current_category is None:
                current_category = item.category
            elif item.category != current_category:
                # Category changed - verify it's in the expected order
                category_order = ["produce", "protein", "dairy", "pantry", "spices", "other"]
                if current_category in category_order and item.category in category_order:
                    assert category_order.index(current_category) <= category_order.index(item.category)
                current_category = item.category


class TestFormatting:
    """Test grocery list formatting."""

    def test_format_with_recipes(self, generator, sample_plan, sample_recipes):
        """Can format with recipe references."""
        grocery_list = generator.generate(sample_plan, sample_recipes)

        formatted = generator.format_for_display(grocery_list, show_recipes=True)

        assert "Chicken Stir Fry" in formatted or "Garlic Pasta" in formatted

    def test_format_without_recipes(self, generator, sample_plan, sample_recipes):
        """Can format without recipe references."""
        grocery_list = generator.generate(sample_plan, sample_recipes)

        formatted = generator.format_for_display(grocery_list, show_recipes=False)

        # Should have ingredient names but maybe not full recipe names in parentheses
        assert "-" in formatted  # List format

    def test_format_grouped(self, generator, sample_plan, sample_recipes):
        """Can format with category grouping."""
        grocery_list = generator.generate(sample_plan, sample_recipes)

        formatted = generator.format_for_display(grocery_list, group_by_category=True)

        # Should have category headers
        assert "**Produce**" in formatted or "**Protein**" in formatted

    def test_format_empty_list(self, generator):
        """Handles empty list gracefully."""
        empty_plan = MealPlan(
            id=1,
            start_date=date.today(),
            end_date=date.today(),
            meals=[],
        )

        grocery_list = generator.generate(empty_plan, {})

        formatted = generator.format_for_display(grocery_list)

        assert "No items" in formatted


class TestSummary:
    """Test grocery list summary."""

    def test_get_summary(self, generator, sample_plan, sample_recipes):
        """Can get category summary."""
        grocery_list = generator.generate(sample_plan, sample_recipes)

        summary = generator.get_summary(grocery_list)

        assert isinstance(summary, dict)
        assert len(summary) > 0


class TestEdgeCases:
    """Test edge cases."""

    def test_missing_recipe(self, generator, sample_recipes):
        """Handles missing recipes gracefully."""
        plan = MealPlan(
            id=1,
            start_date=date.today(),
            end_date=date.today(),
            meals=[
                PlannedMeal(
                    day=date.today(),
                    meal_type="dinner",
                    recipe_id="nonexistent",
                    title="Missing Recipe",
                    position=0,
                    source="discovery",
                ),
            ],
        )

        # Should not raise, just skip the missing recipe
        grocery_list = generator.generate(plan, sample_recipes)

        assert grocery_list.items == []

    def test_empty_ingredients(self, generator):
        """Handles recipe with no ingredients."""
        plan = MealPlan(
            id=1,
            start_date=date.today(),
            end_date=date.today(),
            meals=[
                PlannedMeal(
                    day=date.today(),
                    meal_type="dinner",
                    recipe_id="empty",
                    title="Empty Recipe",
                    position=0,
                    source="discovery",
                ),
            ],
        )

        recipes = {
            "empty": Recipe(
                recipe_id="empty",
                title="Empty Recipe",
                ingredients=[],
                ingredients_normalized=[],
                tags=[],
            )
        }

        grocery_list = generator.generate(plan, recipes)

        assert grocery_list.items == []

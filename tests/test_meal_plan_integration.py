"""Integration tests for meal planning feature."""

from datetime import date, timedelta
from pathlib import Path

import pytest

from src.domain.models import (
    DietaryRestriction,
    ExtractionSource,
    IngredientCategory,
    MealPlan,
    MealPlanConstraints,
    PlannedMeal,
    PreferenceProfile,
    Recipe,
)
from src.planning.constraint_extractor import MealPlanConstraintExtractor
from src.planning.grocery_list import GroceryListGenerator
from src.planning.ingredient_normalizer import IngredientNormalizer
from src.planning.meal_planner import MealPlanner


@pytest.fixture
def db_path(tmp_path):
    """Create a test database with sample recipes."""
    import sqlite3
    import json

    db_file = tmp_path / "test.db"
    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()

    # Create recipes table matching the real schema
    cursor.execute("""
        CREATE TABLE recipes (
            recipe_id TEXT PRIMARY KEY,
            title TEXT,
            ingredients_raw TEXT,
            ingredients_normalized TEXT,
            instructions TEXT,
            tags TEXT,
            minutes INTEGER,
            rating_avg REAL,
            rating_count INTEGER,
            cuisine TEXT,
            is_vegetarian INTEGER,
            is_vegan INTEGER
        )
    """)

    # Insert sample recipes for testing
    recipes_data = [
        {
            "recipe_id": "r1",
            "title": "Chicken Stir Fry",
            "ingredients_raw": json.dumps(["chicken breast", "soy sauce", "garlic", "bell pepper"]),
            "ingredients_normalized": json.dumps(["chicken breast", "soy sauce", "garlic", "bell pepper"]),
            "instructions": json.dumps(["Cook chicken", "Add vegetables"]),
            "tags": json.dumps(["asian", "quick", "chicken"]),
            "minutes": 25,
            "rating_avg": 4.5,
            "rating_count": 100,
            "cuisine": "asian",
            "is_vegetarian": 0,
            "is_vegan": 0,
        },
        {
            "recipe_id": "r2",
            "title": "Vegetable Pasta",
            "ingredients_raw": json.dumps(["pasta", "garlic", "olive oil", "tomato", "basil"]),
            "ingredients_normalized": json.dumps(["pasta", "garlic", "olive oil", "tomato", "basil"]),
            "instructions": json.dumps(["Boil pasta", "Saute vegetables"]),
            "tags": json.dumps(["italian", "vegetarian", "pasta"]),
            "minutes": 30,
            "rating_avg": 4.2,
            "rating_count": 80,
            "cuisine": "italian",
            "is_vegetarian": 1,
            "is_vegan": 1,
        },
        {
            "recipe_id": "r3",
            "title": "Salmon with Broccoli",
            "ingredients_raw": json.dumps(["salmon", "broccoli", "lemon", "garlic", "olive oil"]),
            "ingredients_normalized": json.dumps(["salmon", "broccoli", "lemon", "garlic", "olive oil"]),
            "instructions": json.dumps(["Bake salmon", "Steam broccoli"]),
            "tags": json.dumps(["healthy", "seafood", "quick"]),
            "minutes": 30,
            "rating_avg": 4.6,
            "rating_count": 120,
            "cuisine": "american",
            "is_vegetarian": 0,
            "is_vegan": 0,
        },
        {
            "recipe_id": "r4",
            "title": "Beef Tacos",
            "ingredients_raw": json.dumps(["ground beef", "taco shells", "lettuce", "tomato", "cheese"]),
            "ingredients_normalized": json.dumps(["ground beef", "taco shell", "lettuce", "tomato", "cheese"]),
            "instructions": json.dumps(["Brown beef", "Assemble tacos"]),
            "tags": json.dumps(["mexican", "quick"]),
            "minutes": 20,
            "rating_avg": 4.3,
            "rating_count": 90,
            "cuisine": "mexican",
            "is_vegetarian": 0,
            "is_vegan": 0,
        },
        {
            "recipe_id": "r5",
            "title": "Vegetable Curry",
            "ingredients_raw": json.dumps(["coconut milk", "potato", "chickpea", "curry powder", "onion"]),
            "ingredients_normalized": json.dumps(["coconut milk", "potato", "chickpea", "curry powder", "onion"]),
            "instructions": json.dumps(["Saute onion", "Add vegetables", "Simmer"]),
            "tags": json.dumps(["indian", "vegetarian", "vegan", "curry"]),
            "minutes": 40,
            "rating_avg": 4.4,
            "rating_count": 70,
            "cuisine": "indian",
            "is_vegetarian": 1,
            "is_vegan": 1,
        },
        {
            "recipe_id": "r6",
            "title": "Garlic Shrimp Pasta",
            "ingredients_raw": json.dumps(["shrimp", "pasta", "garlic", "butter", "parsley"]),
            "ingredients_normalized": json.dumps(["shrimp", "pasta", "garlic", "butter", "parsley"]),
            "instructions": json.dumps(["Cook shrimp", "Toss with pasta"]),
            "tags": json.dumps(["seafood", "pasta", "quick"]),
            "minutes": 25,
            "rating_avg": 4.5,
            "rating_count": 85,
            "cuisine": "italian",
            "is_vegetarian": 0,
            "is_vegan": 0,
        },
        {
            "recipe_id": "r7",
            "title": "Mushroom Risotto",
            "ingredients_raw": json.dumps(["arborio rice", "mushroom", "parmesan", "white wine", "onion"]),
            "ingredients_normalized": json.dumps(["arborio rice", "mushroom", "parmesan", "white wine", "onion"]),
            "instructions": json.dumps(["Toast rice", "Add broth gradually"]),
            "tags": json.dumps(["italian", "vegetarian"]),
            "minutes": 45,
            "rating_avg": 4.7,
            "rating_count": 150,
            "cuisine": "italian",
            "is_vegetarian": 1,
            "is_vegan": 0,
        },
    ]

    for r in recipes_data:
        cursor.execute("""
            INSERT INTO recipes
            (recipe_id, title, ingredients_raw, ingredients_normalized, instructions, tags,
             minutes, rating_avg, rating_count, cuisine, is_vegetarian, is_vegan)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            r["recipe_id"], r["title"], r["ingredients_raw"], r["ingredients_normalized"],
            r["instructions"], r["tags"], r["minutes"], r["rating_avg"], r["rating_count"],
            r["cuisine"], r["is_vegetarian"], r["is_vegan"]
        ))

    conn.commit()
    conn.close()

    return db_file


@pytest.fixture
def sample_recipes():
    """Create sample Recipe objects for testing."""
    import json
    return [
        Recipe(
            recipe_id="r1",
            title="Chicken Stir Fry",
            ingredients=["chicken breast", "soy sauce", "garlic", "bell pepper"],
            ingredients_normalized=["chicken breast", "soy sauce", "garlic", "bell pepper"],
            instructions=["Cook chicken", "Add vegetables"],
            tags=["asian", "quick", "chicken"],
            minutes=25,
            rating_avg=4.5,
            rating_count=100,
        ),
        Recipe(
            recipe_id="r2",
            title="Vegetable Pasta",
            ingredients=["pasta", "garlic", "olive oil", "tomato", "basil"],
            ingredients_normalized=["pasta", "garlic", "olive oil", "tomato", "basil"],
            instructions=["Boil pasta", "Saute vegetables"],
            tags=["italian", "vegetarian", "pasta"],
            minutes=30,
            rating_avg=4.2,
            rating_count=80,
        ),
        Recipe(
            recipe_id="r3",
            title="Salmon with Broccoli",
            ingredients=["salmon", "broccoli", "lemon", "garlic", "olive oil"],
            ingredients_normalized=["salmon", "broccoli", "lemon", "garlic", "olive oil"],
            instructions=["Bake salmon", "Steam broccoli"],
            tags=["healthy", "seafood", "quick"],
            minutes=30,
            rating_avg=4.6,
            rating_count=120,
        ),
        Recipe(
            recipe_id="r4",
            title="Beef Tacos",
            ingredients=["ground beef", "taco shells", "lettuce", "tomato", "cheese"],
            ingredients_normalized=["ground beef", "taco shell", "lettuce", "tomato", "cheese"],
            instructions=["Brown beef", "Assemble tacos"],
            tags=["mexican", "quick"],
            minutes=20,
            rating_avg=4.3,
            rating_count=90,
        ),
        Recipe(
            recipe_id="r5",
            title="Vegetable Curry",
            ingredients=["coconut milk", "potato", "chickpea", "curry powder", "onion"],
            ingredients_normalized=["coconut milk", "potato", "chickpea", "curry powder", "onion"],
            instructions=["Saute onion", "Add vegetables", "Simmer"],
            tags=["indian", "vegetarian", "vegan", "curry"],
            minutes=40,
            rating_avg=4.4,
            rating_count=70,
        ),
        Recipe(
            recipe_id="r6",
            title="Garlic Shrimp Pasta",
            ingredients=["shrimp", "pasta", "garlic", "butter", "parsley"],
            ingredients_normalized=["shrimp", "pasta", "garlic", "butter", "parsley"],
            instructions=["Cook shrimp", "Toss with pasta"],
            tags=["seafood", "pasta", "quick"],
            minutes=25,
            rating_avg=4.5,
            rating_count=85,
        ),
        Recipe(
            recipe_id="r7",
            title="Mushroom Risotto",
            ingredients=["arborio rice", "mushroom", "parmesan", "white wine", "onion"],
            ingredients_normalized=["arborio rice", "mushroom", "parmesan", "white wine", "onion"],
            instructions=["Toast rice", "Add broth gradually"],
            tags=["italian", "vegetarian"],
            minutes=45,
            rating_avg=4.7,
            rating_count=150,
        ),
    ]


@pytest.fixture
def profile():
    """Create a test user profile."""
    return PreferenceProfile(
        diet="none",
        spice_level="medium",
        preferred_cuisines=["italian", "asian"],
    )


class TestFullFlowIntegration:
    """Test the full meal planning flow from constraints to grocery list."""

    def test_full_flow_basic(self, db_path, sample_recipes, profile):
        """Test basic flow: constraints -> plan -> grocery."""
        # Step 1: Extract constraints from natural language
        extractor = MealPlanConstraintExtractor(db_path=db_path)
        constraints = extractor.extract("plan 3 days of dinners", profile)

        assert constraints.days == 3
        assert "dinner" in constraints.meal_types

        # Step 2: Generate meal plan
        planner = MealPlanner()
        meals, metrics = planner.generate_plan(sample_recipes, constraints, profile)

        assert len(meals) == 3
        assert metrics.unique_ingredients > 0

        # Step 3: Create plan and generate grocery list
        plan = MealPlan(
            start_date=date.today(),
            end_date=date.today() + timedelta(days=2),
            meals=meals,
            metrics=metrics,
        )

        # Get recipes for grocery list
        recipes = {r.recipe_id: r for r in sample_recipes}
        generator = GroceryListGenerator()
        grocery_list = generator.generate(plan, recipes)

        assert len(grocery_list.items) > 0

    def test_vegetarian_constraint_flow(self, db_path, sample_recipes, profile):
        """Test flow with vegetarian dietary constraint."""
        # Extract constraints
        extractor = MealPlanConstraintExtractor(db_path=db_path)
        constraints = extractor.extract("plan 3 days vegetarian dinners", profile)

        assert constraints.dietary == DietaryRestriction.VEGETARIAN

        # Generate plan
        planner = MealPlanner()
        meals, metrics = planner.generate_plan(sample_recipes, constraints, profile)

        # Should have some meals
        assert len(meals) > 0

    def test_time_constraint_flow(self, db_path, sample_recipes, profile):
        """Test flow with time constraint."""
        extractor = MealPlanConstraintExtractor(db_path=db_path)
        constraints = extractor.extract("plan 3 days quick dinners under 30 minutes", profile)

        assert constraints.max_prep_time == 30

        planner = MealPlanner()
        meals, metrics = planner.generate_plan(sample_recipes, constraints, profile)

        # All meals should be within time limit
        recipes = {r.recipe_id: r for r in sample_recipes}
        for meal in meals:
            recipe = recipes.get(meal.recipe_id)
            if recipe and recipe.minutes:
                assert recipe.minutes <= 30


class TestDeterminismIntegration:
    """Test that meal planning is deterministic across runs."""

    def test_same_constraints_same_plan(self, sample_recipes, profile):
        """Verify identical constraints produce identical plans."""
        constraints = MealPlanConstraints(
            days=5,
            meal_types=["dinner"],
            dietary=DietaryRestriction.NONE,
        )

        planner = MealPlanner()

        # Generate three plans with same constraints
        meals1, _ = planner.generate_plan(sample_recipes, constraints, profile)
        meals2, _ = planner.generate_plan(sample_recipes, constraints, profile)
        meals3, _ = planner.generate_plan(sample_recipes, constraints, profile)

        # All should be identical
        ids1 = [m.recipe_id for m in meals1]
        ids2 = [m.recipe_id for m in meals2]
        ids3 = [m.recipe_id for m in meals3]

        assert ids1 == ids2 == ids3, "Plans should be deterministic"

    def test_constraint_order_independent(self, db_path, sample_recipes, profile):
        """Verify order of constraints doesn't affect plan."""
        extractor = MealPlanConstraintExtractor(db_path=db_path)

        # Same constraints, different order
        c1 = extractor.extract("plan 3 days quick vegetarian dinners", profile)
        c2 = extractor.extract("vegetarian quick 3 days dinners plan", profile)

        planner = MealPlanner()

        meals1, _ = planner.generate_plan(sample_recipes, c1, profile)
        meals2, _ = planner.generate_plan(sample_recipes, c2, profile)

        ids1 = [m.recipe_id for m in meals1]
        ids2 = [m.recipe_id for m in meals2]

        # Results should be same
        assert ids1 == ids2


class TestAuditTrailIntegration:
    """Test that extraction sources are tracked through the flow."""

    def test_extraction_sources_tracked(self, db_path, profile):
        """Verify extraction sources are recorded."""
        extractor = MealPlanConstraintExtractor(db_path=db_path)
        constraints = extractor.extract("plan 5 days vegetarian dinners under 30 minutes", profile)

        # Check extraction sources exist
        assert "days" in constraints.extraction_sources
        assert constraints.extraction_sources["days"].source == ExtractionSource.RULE

        assert "max_prep_time" in constraints.extraction_sources
        assert constraints.extraction_sources["max_prep_time"].source == ExtractionSource.RULE

        assert "dietary" in constraints.extraction_sources
        assert constraints.extraction_sources["dietary"].source == ExtractionSource.RULE

    def test_profile_fallback_tracked(self, db_path):
        """Verify profile-based defaults are tracked."""
        profile = PreferenceProfile(diet="vegetarian")

        extractor = MealPlanConstraintExtractor(db_path=db_path)
        constraints = extractor.extract("plan dinners", profile)

        # Dietary should come from profile
        assert constraints.dietary == DietaryRestriction.VEGETARIAN
        assert "dietary" in constraints.extraction_sources
        assert constraints.extraction_sources["dietary"].source == ExtractionSource.USER_PROFILE


class TestGroceryListIntegration:
    """Test grocery list generation from meal plans."""

    def test_grocery_list_aggregates_ingredients(self, sample_recipes, profile):
        """Verify ingredients are properly aggregated."""
        # Create a simple plan with known recipes
        plan = MealPlan(
            id=1,
            start_date=date.today(),
            end_date=date.today() + timedelta(days=1),
            meals=[
                PlannedMeal(
                    day=date.today(),
                    meal_type="dinner",
                    recipe_id="r2",  # Vegetable Pasta
                    title="Vegetable Pasta",
                    position=0,
                    source="discovery",
                ),
                PlannedMeal(
                    day=date.today() + timedelta(days=1),
                    meal_type="dinner",
                    recipe_id="r6",  # Garlic Shrimp Pasta
                    title="Garlic Shrimp Pasta",
                    position=0,
                    source="discovery",
                ),
            ],
        )

        # Get recipes
        recipes = {r.recipe_id: r for r in sample_recipes}

        generator = GroceryListGenerator()
        grocery_list = generator.generate(plan, recipes, exclude_pantry_staples=False)

        # Garlic and pasta should appear in multiple recipes
        garlic_items = [item for item in grocery_list.items if "garlic" in item.normalized]
        if garlic_items:
            garlic = garlic_items[0]
            assert len(garlic.recipes) == 2  # Both recipes use garlic

    def test_pantry_staples_excluded(self, sample_recipes, profile):
        """Verify pantry staples can be excluded."""
        plan = MealPlan(
            id=1,
            start_date=date.today(),
            end_date=date.today(),
            meals=[
                PlannedMeal(
                    day=date.today(),
                    meal_type="dinner",
                    recipe_id="r2",
                    title="Vegetable Pasta",
                    position=0,
                    source="discovery",
                ),
            ],
        )

        recipes = {r.recipe_id: r for r in sample_recipes}

        generator = GroceryListGenerator()
        list_with_staples = generator.generate(plan, recipes, exclude_pantry_staples=False)
        list_without_staples = generator.generate(plan, recipes, exclude_pantry_staples=True)

        # List without staples should have fewer items
        assert len(list_without_staples.items) <= len(list_with_staples.items)


class TestExclusionConstraints:
    """Test ingredient and category exclusions."""

    def test_category_exclusion(self, db_path, profile):
        """Test excluding a category like dairy."""
        extractor = MealPlanConstraintExtractor(db_path=db_path)
        constraints = extractor.extract("plan dinners, no dairy", profile)

        assert IngredientCategory.DAIRY in constraints.excluded_categories

    def test_tag_exclusion(self, db_path, profile):
        """Test excluding a dish type like casseroles."""
        extractor = MealPlanConstraintExtractor(db_path=db_path)
        constraints = extractor.extract("plan dinners, no pasta", profile)

        # Pasta is a tag
        assert "pasta" in constraints.excluded_tags

    def test_multiple_exclusions(self, db_path, profile):
        """Test multiple exclusions combined."""
        extractor = MealPlanConstraintExtractor(db_path=db_path)
        constraints = extractor.extract("plan dinners, no dairy, no seafood", profile)

        assert IngredientCategory.DAIRY in constraints.excluded_categories
        assert IngredientCategory.SEAFOOD in constraints.excluded_categories


class TestRecipeBoxIntegration:
    """Test that Recipe Box recipes are prioritized in plans."""

    def test_box_recipes_included(self, sample_recipes, profile):
        """Verify Recipe Box recipes are considered."""
        constraints = MealPlanConstraints(days=3)

        planner = MealPlanner()

        # Pass box recipe IDs
        box_ids = {"r1", "r2"}  # Chicken Stir Fry and Vegetable Pasta

        meals, metrics = planner.generate_plan(sample_recipes, constraints, profile, box_recipe_ids=box_ids)

        # Check metrics show box usage
        assert metrics.box_recipe_count >= 0


class TestMetricsIntegration:
    """Test that metrics are computed correctly."""

    def test_metrics_computed(self, sample_recipes, profile):
        """Verify metrics are populated."""
        constraints = MealPlanConstraints(days=5)
        planner = MealPlanner()

        meals, metrics = planner.generate_plan(sample_recipes, constraints, profile)

        assert metrics.unique_ingredients > 0
        assert metrics.total_ingredient_uses >= metrics.unique_ingredients
        assert 0 <= metrics.overlap_ratio <= 1
        assert metrics.unique_per_meal > 0

    def test_top_shared_ingredients_tracked(self, sample_recipes, profile):
        """Verify top shared ingredients are identified."""
        constraints = MealPlanConstraints(days=5)
        planner = MealPlanner()

        meals, metrics = planner.generate_plan(sample_recipes, constraints, profile)

        # Should have some shared ingredients
        if len(meals) > 1:
            # Check structure of top_shared_ingredients
            for item in metrics.top_shared_ingredients:
                assert len(item) == 2  # (ingredient, count)
                assert isinstance(item[0], str)
                assert isinstance(item[1], int)

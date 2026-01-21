"""Tests for meal plan storage."""

import sqlite3
from datetime import date, timedelta

import pytest

from src.domain.models import MealPlan, PlannedMeal, PlanMetrics
from src.memory.meal_plan_store import MealPlanStore


@pytest.fixture
def store(temp_db, test_user_id):
    """Create a MealPlanStore instance."""
    return MealPlanStore(temp_db, test_user_id)


@pytest.fixture
def sample_plan():
    """Create a sample meal plan for testing."""
    start = date.today()
    end = start + timedelta(days=4)

    return MealPlan(
        name="Week 1 Dinners",
        start_date=start,
        end_date=end,
        meal_types=["dinner"],
        status="draft",
        schema_version=1,
        constraints={"days": 5, "dietary": "none"},
        metrics=PlanMetrics(
            unique_ingredients=15,
            total_ingredient_uses=25,
            overlap_ratio=0.4,
            unique_per_meal=3.0,
            top_shared_ingredients=[("chicken", 2), ("garlic", 3)],
            protein_distribution={"chicken": 2, "beef": 1},
            cuisine_distribution={"italian": 2, "asian": 1},
            box_recipe_count=2,
            discovery_recipe_count=3,
        ),
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
                title="Pasta Primavera",
                position=0,
                source="discovery",
            ),
        ],
    )


class TestMealPlanStoreInit:
    """Test store initialization."""

    def test_creates_tables(self, temp_db, test_user_id):
        """Store creates necessary tables on init."""
        # Add extra recipes for meal plan tests (use column names for full schema)
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO recipes (recipe_id, title, tags) VALUES ('r1', 'Chicken Stir Fry', '[]')"
        )
        cursor.execute(
            "INSERT OR IGNORE INTO recipes (recipe_id, title, tags) VALUES ('r2', 'Pasta Primavera', '[]')"
        )
        cursor.execute(
            "INSERT OR IGNORE INTO recipes (recipe_id, title, tags) VALUES ('r3', 'Beef Tacos', '[]')"
        )
        conn.commit()
        conn.close()

        store = MealPlanStore(temp_db, test_user_id)

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        # Check meal_plans table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='meal_plans'"
        )
        assert cursor.fetchone() is not None

        # Check planned_meals table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='planned_meals'"
        )
        assert cursor.fetchone() is not None

        conn.close()

    def test_creates_indexes(self, temp_db, test_user_id):
        """Store creates indexes on init."""
        store = MealPlanStore(temp_db, test_user_id)

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_meal_plans_user'"
        )
        assert cursor.fetchone() is not None

        conn.close()


class TestCreatePlan:
    """Test plan creation."""

    def test_creates_plan(self, store, sample_plan):
        """Can create a meal plan."""
        plan_id = store.create_plan(sample_plan)

        assert plan_id is not None
        assert plan_id > 0

    def test_creates_plan_with_meals(self, store, sample_plan):
        """Plan creation includes meals."""
        plan_id = store.create_plan(sample_plan)

        plan = store.get_plan(plan_id)

        assert len(plan.meals) == 2
        assert plan.meals[0].recipe_id == "r1"
        assert plan.meals[1].recipe_id == "r2"

    def test_serializes_constraints(self, store, sample_plan):
        """Constraints are serialized as JSON."""
        plan_id = store.create_plan(sample_plan)

        plan = store.get_plan(plan_id)

        assert plan.constraints == {"days": 5, "dietary": "none"}

    def test_serializes_metrics(self, store, sample_plan):
        """Metrics are serialized as JSON."""
        plan_id = store.create_plan(sample_plan)

        plan = store.get_plan(plan_id)

        assert plan.metrics is not None
        assert plan.metrics.unique_ingredients == 15
        assert plan.metrics.overlap_ratio == 0.4


class TestGetPlan:
    """Test plan retrieval."""

    def test_get_existing_plan(self, store, sample_plan):
        """Can retrieve an existing plan."""
        plan_id = store.create_plan(sample_plan)

        plan = store.get_plan(plan_id)

        assert plan is not None
        assert plan.id == plan_id
        assert plan.name == "Week 1 Dinners"

    def test_get_nonexistent_plan(self, store):
        """Returns None for nonexistent plan."""
        plan = store.get_plan(9999)

        assert plan is None

    def test_meals_ordered_by_day(self, store, sample_plan):
        """Meals are returned ordered by day."""
        plan_id = store.create_plan(sample_plan)

        plan = store.get_plan(plan_id)

        assert plan.meals[0].day < plan.meals[1].day


class TestUpdatePlanStatus:
    """Test plan status updates."""

    def test_update_status(self, store, sample_plan):
        """Can update plan status."""
        plan_id = store.create_plan(sample_plan)

        result = store.update_plan_status(plan_id, "active")

        assert result is True

        plan = store.get_plan(plan_id)
        assert plan.status == "active"

    def test_update_nonexistent_plan(self, store):
        """Returns False for nonexistent plan."""
        result = store.update_plan_status(9999, "active")

        assert result is False


class TestUpdatePlanMetrics:
    """Test plan metrics updates."""

    def test_update_metrics(self, store, sample_plan):
        """Can update plan metrics."""
        plan_id = store.create_plan(sample_plan)

        new_metrics = PlanMetrics(
            unique_ingredients=20,
            total_ingredient_uses=30,
            overlap_ratio=0.33,
            unique_per_meal=4.0,
            top_shared_ingredients=[],
            protein_distribution={},
            cuisine_distribution={},
            box_recipe_count=0,
            discovery_recipe_count=5,
        )

        result = store.update_plan_metrics(plan_id, new_metrics)

        assert result is True

        plan = store.get_plan(plan_id)
        assert plan.metrics.unique_ingredients == 20


class TestDeletePlan:
    """Test plan deletion."""

    def test_delete_plan(self, store, sample_plan):
        """Can delete a plan."""
        plan_id = store.create_plan(sample_plan)

        result = store.delete_plan(plan_id)

        assert result is True
        assert store.get_plan(plan_id) is None

    def test_delete_removes_meals(self, store, sample_plan, temp_db):
        """Deleting plan removes associated meals."""
        plan_id = store.create_plan(sample_plan)

        store.delete_plan(plan_id)

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM planned_meals WHERE plan_id = ?", (plan_id,))
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 0

    def test_delete_nonexistent_plan(self, store):
        """Returns False for nonexistent plan."""
        result = store.delete_plan(9999)

        assert result is False


class TestGetPlansByStatus:
    """Test filtering plans by status."""

    def test_get_draft_plans(self, store, sample_plan):
        """Can get draft plans."""
        store.create_plan(sample_plan)

        plans = store.get_plans_by_status("draft")

        assert len(plans) == 1
        assert plans[0].status == "draft"

    def test_get_active_plans(self, store, sample_plan):
        """Can get active plans."""
        plan_id = store.create_plan(sample_plan)
        store.update_plan_status(plan_id, "active")

        plans = store.get_plans_by_status("active")

        assert len(plans) == 1
        assert plans[0].status == "active"

    def test_empty_result(self, store):
        """Returns empty list when no plans match."""
        plans = store.get_plans_by_status("completed")

        assert plans == []


class TestGetActivePlan:
    """Test getting the active plan."""

    def test_get_active_plan(self, store, sample_plan):
        """Can get the active plan."""
        plan_id = store.create_plan(sample_plan)
        store.update_plan_status(plan_id, "active")

        plan = store.get_active_plan()

        assert plan is not None
        assert plan.id == plan_id

    def test_no_active_plan(self, store):
        """Returns None when no active plan."""
        plan = store.get_active_plan()

        assert plan is None


class TestMealOperations:
    """Test adding and removing individual meals."""

    def test_add_meal_to_plan(self, store, sample_plan):
        """Can add a meal to an existing plan."""
        plan_id = store.create_plan(sample_plan)

        new_meal = PlannedMeal(
            plan_id=plan_id,
            day=sample_plan.start_date + timedelta(days=2),
            meal_type="dinner",
            recipe_id="r3",
            title="Beef Tacos",
            position=0,
            source="discovery",
        )

        meal_id = store.add_meal_to_plan(new_meal)

        assert meal_id > 0

        plan = store.get_plan(plan_id)
        assert len(plan.meals) == 3

    def test_add_meal_without_plan_id_raises(self, store):
        """Adding meal without plan_id raises ValueError."""
        meal = PlannedMeal(
            day=date.today(),
            meal_type="dinner",
            recipe_id="r3",
            title="Beef Tacos",
            position=0,
            source="discovery",
        )

        with pytest.raises(ValueError, match="plan_id must be set"):
            store.add_meal_to_plan(meal)

    def test_remove_meal_from_plan(self, store, sample_plan):
        """Can remove a meal from a plan."""
        plan_id = store.create_plan(sample_plan)

        plan = store.get_plan(plan_id)
        meal_id = plan.meals[0].id

        result = store.remove_meal_from_plan(meal_id)

        assert result is True

        plan = store.get_plan(plan_id)
        assert len(plan.meals) == 1

    def test_remove_nonexistent_meal(self, store):
        """Returns False for nonexistent meal."""
        result = store.remove_meal_from_plan(9999)

        assert result is False


class TestGetRecentPlans:
    """Test getting recent plans."""

    def test_get_recent_plans(self, store, sample_plan):
        """Can get recently updated plans."""
        store.create_plan(sample_plan)

        sample_plan.name = "Week 2 Dinners"
        store.create_plan(sample_plan)

        plans = store.get_recent_plans(limit=2)

        assert len(plans) == 2

    def test_respect_limit(self, store, sample_plan):
        """Respects limit parameter."""
        for i in range(5):
            sample_plan.name = f"Week {i} Dinners"
            store.create_plan(sample_plan)

        plans = store.get_recent_plans(limit=3)

        assert len(plans) == 3


class TestPlanCount:
    """Test plan counting."""

    def test_count_plans(self, store, sample_plan):
        """Can count total plans."""
        assert store.get_plan_count() == 0

        store.create_plan(sample_plan)
        assert store.get_plan_count() == 1

        store.create_plan(sample_plan)
        assert store.get_plan_count() == 2


class TestSchemaVersion:
    """Test schema version tracking."""

    def test_stores_schema_version(self, store, sample_plan):
        """Schema version is stored."""
        sample_plan.schema_version = 2
        plan_id = store.create_plan(sample_plan)

        plan = store.get_plan(plan_id)

        assert plan.schema_version == 2

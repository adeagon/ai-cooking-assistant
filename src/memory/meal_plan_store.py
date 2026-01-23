"""Meal plan storage using SQLite."""

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Literal

# Register datetime adapters for Python 3.12+ compatibility
from src.memory import _sqlite_compat  # noqa: F401

from src.app.logging_config import get_logger
from src.domain.models import MealPlan, PlannedMeal, PlanMetrics
from src.memory.base_store import BaseUserBoundStore

logger = get_logger(__name__)


class MealPlanStore(BaseUserBoundStore):
    """Manages persistent storage of meal plans in SQLite.

    Each store instance is bound to a specific user at instantiation.
    The user cannot be changed after initialization.
    """

    def _ensure_table(self) -> None:
        """Create meal plan tables and indexes if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Create meal_plans table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS meal_plans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                name TEXT,
                start_date DATE NOT NULL,
                end_date DATE NOT NULL,
                meal_types TEXT NOT NULL,
                status TEXT DEFAULT 'draft',
                schema_version INTEGER DEFAULT 1,
                constraints TEXT,
                metrics TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create planned_meals table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS planned_meals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plan_id INTEGER NOT NULL,
                day DATE NOT NULL,
                meal_type TEXT NOT NULL,
                recipe_id TEXT NOT NULL,
                title TEXT NOT NULL,
                position INTEGER DEFAULT 0,
                source TEXT DEFAULT 'discovery',
                notes TEXT,
                FOREIGN KEY (plan_id) REFERENCES meal_plans(id) ON DELETE CASCADE
            )
        """)

        # Create indexes for efficient queries
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_meal_plans_user
            ON meal_plans(user_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_meal_plans_status
            ON meal_plans(status)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_planned_meals_plan
            ON planned_meals(plan_id)
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_planned_meals_day
            ON planned_meals(day)
        """)

        conn.commit()
        conn.close()

        logger.info("Meal plan tables ensured", db_path=str(self.db_path))

    def create_plan(self, plan: MealPlan) -> int:
        """Create a new meal plan.

        Args:
            plan: MealPlan object to save

        Returns:
            ID of the created plan
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        now = datetime.now()

        # Serialize meal_types, constraints, and metrics as JSON
        meal_types_json = json.dumps(plan.meal_types)
        constraints_json = json.dumps(plan.constraints) if plan.constraints else None
        metrics_json = (
            json.dumps(plan.metrics.model_dump()) if plan.metrics else None
        )

        cursor.execute(
            """
            INSERT INTO meal_plans (
                user_id, name, start_date, end_date, meal_types,
                status, schema_version, constraints, metrics,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                self.user,  # Use bound user, not plan.user_id
                plan.name,
                plan.start_date.isoformat(),
                plan.end_date.isoformat(),
                meal_types_json,
                plan.status,
                plan.schema_version,
                constraints_json,
                metrics_json,
                now,
                now,
            ),
        )

        plan_id = cursor.lastrowid

        # Insert planned meals
        for meal in plan.meals:
            cursor.execute(
                """
                INSERT INTO planned_meals (
                    plan_id, day, meal_type, recipe_id, title, position, source, notes
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plan_id,
                    meal.day.isoformat(),
                    meal.meal_type,
                    meal.recipe_id,
                    meal.title,
                    meal.position,
                    meal.source,
                    None,  # notes
                ),
            )

        conn.commit()
        conn.close()

        logger.info("Created meal plan", plan_id=plan_id, name=plan.name)
        return plan_id

    def get_plan(self, plan_id: int) -> MealPlan | None:
        """Get a meal plan by ID.

        Args:
            plan_id: ID of the plan to retrieve

        Returns:
            MealPlan object or None if not found
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, user_id, name, start_date, end_date, meal_types,
                   status, schema_version, constraints, metrics,
                   created_at, updated_at
            FROM meal_plans
            WHERE id = ? AND user_id = ?
            """,
            (plan_id, self.user),
        )

        row = cursor.fetchone()

        if not row:
            conn.close()
            return None

        # Get planned meals for this plan
        cursor.execute(
            """
            SELECT id, plan_id, day, meal_type, recipe_id, title, position, source, notes
            FROM planned_meals
            WHERE plan_id = ?
            ORDER BY day, position
            """,
            (plan_id,),
        )

        meals = [
            PlannedMeal(
                id=meal_row["id"],
                plan_id=meal_row["plan_id"],
                day=date.fromisoformat(meal_row["day"]),
                meal_type=meal_row["meal_type"],
                recipe_id=meal_row["recipe_id"],
                title=meal_row["title"],
                position=meal_row["position"],
                source=meal_row["source"],
            )
            for meal_row in cursor.fetchall()
        ]

        conn.close()

        # Parse JSON fields
        meal_types = json.loads(row["meal_types"])
        constraints = json.loads(row["constraints"]) if row["constraints"] else None
        metrics = (
            PlanMetrics(**json.loads(row["metrics"])) if row["metrics"] else None
        )

        return MealPlan(
            id=row["id"],
            user_id=row["user_id"],
            name=row["name"],
            start_date=date.fromisoformat(row["start_date"]),
            end_date=date.fromisoformat(row["end_date"]),
            meal_types=meal_types,
            status=row["status"],
            schema_version=row["schema_version"],
            constraints=constraints,
            metrics=metrics,
            created_at=(
                datetime.fromisoformat(row["created_at"]) if row["created_at"] else None
            ),
            updated_at=(
                datetime.fromisoformat(row["updated_at"]) if row["updated_at"] else None
            ),
            meals=meals,
        )

    def get_plans_by_status(
        self,
        status: Literal["draft", "active", "completed", "archived"],
        limit: int = 10,
    ) -> list[MealPlan]:
        """Get meal plans by status for this user.

        Args:
            status: Plan status to filter by
            limit: Maximum number of plans to return

        Returns:
            List of MealPlan objects
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id FROM meal_plans
            WHERE status = ? AND user_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (status, self.user, limit),
        )

        plan_ids = [row["id"] for row in cursor.fetchall()]
        conn.close()

        # Fetch full plan details for each
        return [self.get_plan(pid) for pid in plan_ids if self.get_plan(pid)]

    def get_active_plan(self) -> MealPlan | None:
        """Get the currently active meal plan for this user.

        Returns:
            Active MealPlan or None if no active plan
        """
        plans = self.get_plans_by_status("active", limit=1)
        return plans[0] if plans else None

    def update_plan_status(
        self,
        plan_id: int,
        status: Literal["draft", "active", "completed", "archived"],
    ) -> bool:
        """Update the status of a meal plan.

        Args:
            plan_id: ID of the plan to update
            status: New status

        Returns:
            True if updated, False if plan not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE meal_plans
            SET status = ?, updated_at = ?
            WHERE id = ? AND user_id = ?
            """,
            (status, datetime.now(), plan_id, self.user),
        )

        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()

        if rows_affected > 0:
            logger.info("Updated plan status", plan_id=plan_id, status=status)
            return True
        else:
            logger.warning("Plan not found for status update", plan_id=plan_id)
            return False

    def update_plan_metrics(self, plan_id: int, metrics: PlanMetrics) -> bool:
        """Update the metrics for a meal plan.

        Args:
            plan_id: ID of the plan to update
            metrics: New metrics

        Returns:
            True if updated, False if plan not found
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        metrics_json = json.dumps(metrics.model_dump())

        cursor.execute(
            """
            UPDATE meal_plans
            SET metrics = ?, updated_at = ?
            WHERE id = ? AND user_id = ?
            """,
            (metrics_json, datetime.now(), plan_id, self.user),
        )

        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()

        if rows_affected > 0:
            logger.info("Updated plan metrics", plan_id=plan_id)
            return True
        else:
            logger.warning("Plan not found for metrics update", plan_id=plan_id)
            return False

    def delete_plan(self, plan_id: int) -> bool:
        """Delete a meal plan and its meals.

        Args:
            plan_id: ID of the plan to delete

        Returns:
            True if deleted, False if plan not found or not owned by user
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Verify ownership before deletion
        cursor.execute(
            "SELECT id FROM meal_plans WHERE id = ? AND user_id = ?",
            (plan_id, self.user),
        )
        if cursor.fetchone() is None:
            conn.close()
            logger.warning("Plan not found or not owned by user", plan_id=plan_id, user=self.user)
            return False

        # Delete meals first (foreign key constraint)
        cursor.execute(
            """
            DELETE FROM planned_meals
            WHERE plan_id = ?
            """,
            (plan_id,),
        )

        # Delete plan (ownership already verified)
        cursor.execute(
            """
            DELETE FROM meal_plans
            WHERE id = ? AND user_id = ?
            """,
            (plan_id, self.user),
        )

        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()

        if rows_affected > 0:
            logger.info("Deleted meal plan", plan_id=plan_id, user=self.user)
            return True
        else:
            logger.warning("Plan not found for deletion", plan_id=plan_id)
            return False

    def get_recent_plans(self, limit: int = 5) -> list[MealPlan]:
        """Get recently updated meal plans for this user.

        Args:
            limit: Maximum number of plans to return

        Returns:
            List of MealPlan objects, most recently updated first
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id FROM meal_plans
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (self.user, limit),
        )

        plan_ids = [row["id"] for row in cursor.fetchall()]
        conn.close()

        return [p for pid in plan_ids if (p := self.get_plan(pid)) is not None]

    def add_meal_to_plan(self, meal: PlannedMeal) -> int:
        """Add a single meal to an existing plan.

        Args:
            meal: PlannedMeal to add

        Returns:
            ID of the inserted meal

        Raises:
            ValueError: If plan_id is not set on the meal, or plan not owned by user
        """
        if meal.plan_id is None:
            raise ValueError("plan_id must be set on the meal")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Verify plan ownership before adding meal
        cursor.execute(
            "SELECT id FROM meal_plans WHERE id = ? AND user_id = ?",
            (meal.plan_id, self.user),
        )
        if cursor.fetchone() is None:
            conn.close()
            raise ValueError(f"Plan {meal.plan_id} not found or not owned by {self.user}")

        cursor.execute(
            """
            INSERT INTO planned_meals (
                plan_id, day, meal_type, recipe_id, title, position, source, notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                meal.plan_id,
                meal.day.isoformat(),
                meal.meal_type,
                meal.recipe_id,
                meal.title,
                meal.position,
                meal.source,
                None,
            ),
        )

        meal_id = cursor.lastrowid

        # Update plan's updated_at (ownership already verified)
        cursor.execute(
            """
            UPDATE meal_plans
            SET updated_at = ?
            WHERE id = ? AND user_id = ?
            """,
            (datetime.now(), meal.plan_id, self.user),
        )

        conn.commit()
        conn.close()

        logger.info("Added meal to plan", meal_id=meal_id, plan_id=meal.plan_id, user=self.user)
        return meal_id

    def remove_meal_from_plan(self, meal_id: int) -> bool:
        """Remove a single meal from a plan.

        Args:
            meal_id: ID of the meal to remove

        Returns:
            True if removed, False if not found or not owned by user
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get plan_id and verify ownership before deletion
        cursor.execute(
            """
            SELECT pm.plan_id
            FROM planned_meals pm
            JOIN meal_plans mp ON pm.plan_id = mp.id
            WHERE pm.id = ? AND mp.user_id = ?
            """,
            (meal_id, self.user),
        )

        row = cursor.fetchone()

        if not row:
            conn.close()
            logger.warning("Meal not found or plan not owned by user", meal_id=meal_id, user=self.user)
            return False

        plan_id = row[0]

        cursor.execute(
            """
            DELETE FROM planned_meals
            WHERE id = ?
            """,
            (meal_id,),
        )

        # Update plan's updated_at (ownership already verified)
        cursor.execute(
            """
            UPDATE meal_plans
            SET updated_at = ?
            WHERE id = ? AND user_id = ?
            """,
            (datetime.now(), plan_id, self.user),
        )

        conn.commit()
        conn.close()

        logger.info("Removed meal from plan", meal_id=meal_id, plan_id=plan_id, user=self.user)
        return True

    def get_plan_count(self) -> int:
        """Get total number of meal plans for this user.

        Returns:
            Count of meal plans
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT COUNT(*) FROM meal_plans WHERE user_id = ?", (self.user,))

        count = cursor.fetchone()[0]

        conn.close()
        return count

    def save_plan(self, plan: MealPlan) -> int:
        """Save a meal plan (alias for create_plan).

        Args:
            plan: MealPlan object to save

        Returns:
            ID of the created plan
        """
        return self.create_plan(plan)

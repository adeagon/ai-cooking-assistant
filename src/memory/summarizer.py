"""Rolling summary manager for session context."""

from src.app.logging_config import get_logger
from src.domain.models import Constraints

logger = get_logger(__name__)


class RollingSummarizer:
    """Manages rolling summary of session conversation.

    Uses template-based approach (not LLM) to keep track of 1-3 key constraints
    that have been discussed during the session.
    """

    MAX_POINTS = 3

    def update_summary(
        self, old_summary: str, new_constraints: Constraints, user_input: str
    ) -> str:
        """Update rolling summary with new turn information.

        Args:
            old_summary: Previous summary (may be empty)
            new_constraints: Constraints extracted from current turn
            user_input: User's input text

        Returns:
            Updated summary string (1-3 sentences)
        """
        # Extract key points from new constraints
        points = []

        if new_constraints.ingredients:
            ingredients_str = ", ".join(new_constraints.ingredients[:3])
            if len(new_constraints.ingredients) > 3:
                ingredients_str += "..."
            points.append(f"ingredients: {ingredients_str}")

        if new_constraints.time_limit:
            points.append(f"time: {new_constraints.time_limit} min")

        if new_constraints.dietary:
            points.append(f"diet: {new_constraints.dietary}")

        if new_constraints.cuisine:
            points.append(f"cuisine: {new_constraints.cuisine}")

        if new_constraints.dish_name:
            points.append(f"dish: {new_constraints.dish_name}")

        if new_constraints.goals:
            goals_str = ", ".join(new_constraints.goals[:2])
            points.append(f"goals: {goals_str}")

        # If no new constraints, return old summary
        if not points:
            logger.debug("No new constraints to add to summary")
            return old_summary

        # Parse old summary to extract existing points
        existing_points = []
        if old_summary:
            # Simple parsing: split by "; " or ". "
            existing_points = [p.strip() for p in old_summary.replace(". ", "; ").split("; ")]

        # Merge new points, avoiding duplicates
        all_points = existing_points + points

        # Deduplicate by keeping first occurrence
        seen = set()
        unique_points = []
        for point in all_points:
            # Get point category (e.g., "ingredients:", "time:")
            category = point.split(":")[0] if ":" in point else point
            if category not in seen:
                seen.add(category)
                unique_points.append(point)

        # Keep only last MAX_POINTS
        final_points = unique_points[-self.MAX_POINTS :]

        # Format as summary
        summary = "; ".join(final_points)

        logger.info("Updated rolling summary", point_count=len(final_points), summary=summary)

        return summary

    def clear_summary(self) -> str:
        """Clear summary (for new session).

        Returns:
            Empty string
        """
        logger.info("Cleared rolling summary")
        return ""

    def format_for_prompt(self, summary: str) -> str:
        """Format summary for inclusion in LLM prompt.

        Args:
            summary: Rolling summary string

        Returns:
            Formatted summary for prompt, or empty string if no summary
        """
        if not summary:
            return ""

        return f"Previous discussion points: {summary}"

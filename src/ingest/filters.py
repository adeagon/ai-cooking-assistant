"""Quality filtering for recipes."""

from src.domain.models import RatingStats
from src.app.logging_config import get_logger

logger = get_logger(__name__)


def apply_quality_filters(
    recipe: dict,
    rating_stats: RatingStats | None,
    min_rating_count: int = 3,
    min_rating_avg: float = 3.5,
    max_minutes: int = 1440
) -> bool:
    """Return True if recipe passes all quality filters.

    Filters:
    - rating_count >= min_rating_count
    - rating_avg >= min_rating_avg
    - has non-empty ingredients
    - has non-empty instructions (steps)
    - has title (name)
    - cooking time <= max_minutes (default 24 hours)

    Args:
        recipe: Recipe dict from CSV
        rating_stats: RatingStats for this recipe (or None if no ratings)
        min_rating_count: Minimum number of ratings required
        min_rating_avg: Minimum average rating required
        max_minutes: Maximum cooking time in minutes (default 1440 = 24 hours)

    Returns:
        True if recipe passes all filters, False otherwise
    """
    # Must have title
    if not recipe.get('name'):
        return False

    # Must have ingredients
    ingredients = recipe.get('ingredients', [])
    if not ingredients or len(ingredients) == 0:
        return False

    # Must have steps
    steps = recipe.get('steps', [])
    if not steps or len(steps) == 0:
        return False

    # Must have sufficient ratings
    if rating_stats is None:
        return False

    if rating_stats.rating_count < min_rating_count:
        return False

    if rating_stats.rating_avg < min_rating_avg:
        return False

    # Filter out recipes with unrealistic cooking times
    minutes = recipe.get('minutes')
    if minutes is not None and minutes > max_minutes:
        return False

    return True

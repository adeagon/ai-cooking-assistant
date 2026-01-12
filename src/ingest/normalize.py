"""Ingredient normalization using ingredient-parser-nlp."""

from ingredient_parser import parse_ingredient
from src.domain.models import NormalizedIngredient
from src.app.logging_config import get_logger

logger = get_logger(__name__)


def normalize_ingredient(raw: str) -> NormalizedIngredient:
    """Parse ingredient with ingredient-parser-nlp.

    Args:
        raw: Raw ingredient string

    Returns:
        NormalizedIngredient with parsed components
    """
    try:
        parsed = parse_ingredient(raw)

        # Extract normalized name (fallback to raw if parsing fails)
        name = parsed.get('name', raw).strip().lower()
        quantity = parsed.get('quantity')
        unit = parsed.get('unit')

        # Convert quantity to string if present
        if quantity is not None:
            quantity = str(quantity)

        return NormalizedIngredient(
            name=name,
            quantity=quantity,
            unit=unit,
            raw=raw
        )

    except Exception as e:
        logger.debug(
            "Failed to parse ingredient, using raw string",
            raw=raw,
            error=str(e)
        )
        # Fallback: use raw string as name
        return NormalizedIngredient(
            name=raw.strip().lower(),
            quantity=None,
            unit=None,
            raw=raw
        )


def extract_key_ingredients(
    ingredients: list[str],
    max_count: int = 15
) -> list[str]:
    """Extract normalized ingredient names, deduplicated.

    Args:
        ingredients: List of raw ingredient strings
        max_count: Maximum number of ingredients to return

    Returns:
        List of normalized ingredient names (deduplicated, lowercase)
    """
    normalized_names = []
    seen = set()

    for raw in ingredients:
        normalized = normalize_ingredient(raw)
        name = normalized.name

        # Skip if already seen or empty
        if name and name not in seen:
            normalized_names.append(name)
            seen.add(name)

            # Stop if we've reached max count
            if len(normalized_names) >= max_count:
                break

    return normalized_names

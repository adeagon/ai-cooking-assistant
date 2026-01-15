"""Rule-based constraint extraction from user input."""

import re
from typing import Any

from langchain_core.runnables import RunnableLambda

from src.app.logging_config import get_logger
from src.domain.models import Constraints
from src.utils.tag_loader import (
    GOAL_FALLBACKS,
    load_cuisines_from_db,
    load_goals_from_db,
)

logger = get_logger(__name__)


class ConstraintExtractor:
    """Extract constraints from user input using rule-based patterns."""

    def __init__(self, db_path: str | None = None):
        """Initialize extractor with cuisines and goals from database.

        Args:
            db_path: Path to SQLite database. If None, uses default from Settings.
        """
        self._valid_cuisines = load_cuisines_from_db(db_path)
        self._valid_goals = load_goals_from_db(db_path)
        logger.debug(
            "ConstraintExtractor initialized",
            cuisine_count=len(self._valid_cuisines),
            goal_count=len(self._valid_goals),
        )

    # Common ingredients pattern
    INGREDIENT_PATTERNS = [
        r"(?:I have|using|with|got)\s+((?:\w+(?:\s+\w+)*(?:,?\s+(?:and\s+)?)?)+)",
        r"(?:ingredients?:?\s*)([\w\s,]+)",
    ]

    # Time limit patterns
    TIME_PATTERNS = [
        r"(?:under|less than|within|about|around)\s+(\d+)\s*(?:min(?:ute)?s?|hours?)",
        r"(\d+)\s*(?:min(?:ute)?s?|hours?)\s+(?:or less|max|maximum)",
    ]

    # Quick/fast patterns
    QUICK_PATTERNS = [r"\bquick\b", r"\bfast\b", r"\beasy\b", r"\bsimple\b"]

    # Dietary restriction patterns
    DIETARY_PATTERNS = {
        "vegetarian": r"\bvegetarian\b",
        "vegan": r"\bvegan\b",
        "pescatarian": r"\bpescatarian\b",
        "keto": r"\b(?:keto|ketogenic)\b",
        "gluten_free": r"\b(?:gluten[-\s]?free|celiac)\b",
        "dairy_free": r"\b(?:dairy[-\s]?free|lactose[-\s]?free)\b",
    }

    # Cuisine patterns
    CUISINE_PATTERNS = {
        "italian": r"\b(?:italian|italy)\b",
        "mexican": r"\b(?:mexican|mexico|tex[-\s]?mex)\b",
        "chinese": r"\b(?:chinese|china)\b",
        "japanese": r"\b(?:japanese|japan|sushi)\b",
        "indian": r"\b(?:indian|india|curry)\b",
        "thai": r"\b(?:thai|thailand)\b",
        "french": r"\b(?:french|france)\b",
        "american": r"\b(?:american|usa|bbq)\b",
        "mediterranean": r"\bmediterranean\b",
    }

    # Dish name to cuisine mapping (for recognizing specific dishes)
    DISH_TO_CUISINE = {
        "indian": [
            "tikka masala", "chicken tikka", "butter chicken", "biryani",
            "vindaloo", "korma", "saag", "paneer", "dal", "naan",
            "samosa", "tandoori", "masala", "chana", "aloo"
        ],
        "thai": [
            "pad thai", "green curry", "red curry", "tom yum", "tom kha",
            "massaman", "panang", "larb", "som tam", "satay"
        ],
        "chinese": [
            "kung pao", "general tso", "orange chicken", "lo mein", "chow mein",
            "fried rice", "dim sum", "wontons", "dumplings", "mapo tofu"
        ],
        "japanese": [
            "ramen", "teriyaki", "tempura", "katsu", "udon", "sashimi",
            "miso", "gyoza", "edamame", "yakitori", "tonkatsu"
        ],
        "mexican": [
            "tacos", "burrito", "enchilada", "quesadilla", "tamale",
            "carnitas", "fajita", "tostada", "chimichanga", "mole"
        ],
        "italian": [
            "carbonara", "bolognese", "lasagna", "risotto", "parmigiana",
            "alfredo", "primavera", "marinara", "pesto", "bruschetta"
        ],
        "french": [
            "coq au vin", "ratatouille", "bouillabaisse", "cassoulet",
            "quiche", "soufflé", "croque", "boeuf bourguignon"
        ],
        "mediterranean": [
            "falafel", "hummus", "shawarma", "gyro", "kebab",
            "tabbouleh", "tzatziki", "moussaka", "spanakopita"
        ],
    }

    # Goal patterns
    GOAL_PATTERNS = {
        "healthy": r"\bhealthy\b",
        "comfort": r"\bcomfort(?:\s+food)?\b",
        "spicy": r"\b(?:spicy|hot|fiery)\b",
        "mild": r"\bmild\b",
    }

    # Negative constraint patterns (things to avoid/exclude)
    AVOID_PATTERNS = [
        r"(?:no|without|avoid|skip|exclude|not?)\s+(\w+(?:\s+\w+)?)",  # "no casseroles", "without cheese"
        r"(?:but not?|except|except for)\s+(\w+(?:\s+\w+)?)",  # "but not casseroles", "except soups"
        r"(?:don'?t want|don'?t like|hate)\s+(\w+(?:\s+\w+)?)",  # "don't want casseroles"
    ]

    def extract_constraints(self, user_input: str) -> Constraints:
        """Extract constraints from user input.

        Args:
            user_input: User's natural language query

        Returns:
            Constraints object with extracted fields
        """
        text = user_input.lower()

        # Extract ingredients
        ingredients = self._extract_ingredients(text)

        # Extract time limit
        time_limit = self._extract_time_limit(text)

        # Check for "quick" patterns
        if any(re.search(pattern, text, re.IGNORECASE) for pattern in self.QUICK_PATTERNS):
            if time_limit is None:
                time_limit = 30  # Default quick = 30 minutes

        # Extract dietary restrictions
        dietary = self._extract_dietary(text)

        # Extract cuisine and dish name
        cuisine, dish_name = self._extract_cuisine_and_dish(text)

        # Extract goals
        goals = self._extract_goals(text)

        # Extract avoid/exclude constraints
        avoid = self._extract_avoid(text)

        # Build constraints object
        constraints = Constraints(
            ingredients=ingredients,
            time_limit=time_limit,
            dietary=dietary,
            cuisine=cuisine,
            goals=goals,
            dish_name=dish_name,
            avoid=avoid,
        )

        logger.info(
            "Extracted constraints",
            ingredients_count=len(ingredients),
            time_limit=time_limit,
            dietary=dietary,
            cuisine=cuisine,
            dish_name=dish_name,
            goals=goals,
            avoid=avoid,
        )

        return constraints

    def _extract_ingredients(self, text: str) -> list[str]:
        """Extract ingredients from text."""
        ingredients = []

        for pattern in self.INGREDIENT_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                # Parse ingredient list
                ingredient_text = match.group(1)
                # Split by commas and "and"
                parts = re.split(r",\s*|\s+and\s+", ingredient_text)
                ingredients.extend([p.strip() for p in parts if p.strip()])

        # Deduplicate
        return list(dict.fromkeys(ingredients))

    def _extract_time_limit(self, text: str) -> int | None:
        """Extract time limit in minutes."""
        for pattern in self.TIME_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                time_value = int(match.group(1))

                # Check if hours or minutes
                if "hour" in match.group(0).lower():
                    return time_value * 60

                return time_value

        return None

    def _extract_dietary(self, text: str) -> str | None:
        """Extract dietary restriction."""
        for diet_name, pattern in self.DIETARY_PATTERNS.items():
            if re.search(pattern, text, re.IGNORECASE):
                return diet_name

        return None

    def _extract_cuisine_and_dish(self, text: str) -> tuple[str | None, str | None]:
        """Extract cuisine preference and specific dish name.

        Uses cuisines loaded from database for dynamic matching.

        Returns:
            Tuple of (cuisine, dish_name)
        """
        cuisine = None
        dish_name = None
        text_lower = text.lower()

        # First check for cuisines from database (dynamic)
        for cuisine_name in self._valid_cuisines:
            # Create pattern that handles hyphenated cuisines (e.g., "middle-eastern")
            pattern = rf"\b{cuisine_name.replace('-', '[-\\s]?')}\b"
            if re.search(pattern, text_lower, re.IGNORECASE):
                cuisine = cuisine_name
                break

        # Fallback: check legacy hardcoded patterns for any we might have missed
        if not cuisine:
            for cuisine_name, pattern in self.CUISINE_PATTERNS.items():
                if re.search(pattern, text_lower, re.IGNORECASE):
                    cuisine = cuisine_name
                    break

        # Then check for dish names that imply cuisine
        for cuisine_name, dishes in self.DISH_TO_CUISINE.items():
            for dish in dishes:
                if dish in text_lower:
                    # Found a dish name - extract cuisine if not already set
                    if not cuisine:
                        cuisine = cuisine_name
                    # Always capture the dish name for search queries
                    dish_name = dish
                    return cuisine, dish_name

        return cuisine, dish_name

    def _extract_avoid(self, text: str) -> list[str]:
        """Extract things to avoid/exclude from user input.

        Matches patterns like:
        - "no casseroles"
        - "but not soups"
        - "without cheese"
        - "avoid spicy"
        """
        avoid = []

        for pattern in self.AVOID_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                item = match.group(1).strip().lower()
                # Skip common false positives
                if item and item not in ("a", "an", "the", "any", "some", "it"):
                    # Clean up: remove trailing punctuation
                    item = re.sub(r"[.,!?]+$", "", item)
                    if item and item not in avoid:
                        avoid.append(item)

        return avoid

    def _extract_goals(self, text: str) -> list[str]:
        """Extract user goals/preferences using DB tags + fallback mappings."""
        goals = []
        text_lower = text.lower()

        # Check for valid goal tags from database
        for goal in self._valid_goals:
            # Handle hyphenated goals (e.g., "comfort-food", "low-calorie")
            pattern = rf"\b{goal.replace('-', '[-\\s]?')}\b"
            if re.search(pattern, text_lower):
                goals.append(goal)

        # Check fallback terms (light → low-calorie, cheap → inexpensive, etc.)
        for term, mapped_goal in GOAL_FALLBACKS.items():
            if re.search(rf"\b{term}\b", text_lower):
                if mapped_goal not in goals:
                    goals.append(mapped_goal)

        # Fallback: check legacy hardcoded patterns
        for goal_name, pattern in self.GOAL_PATTERNS.items():
            if re.search(pattern, text_lower, re.IGNORECASE):
                if goal_name not in goals:
                    goals.append(goal_name)

        return goals


def extract_constraints_runnable(input_data: dict[str, Any]) -> dict[str, Any]:
    """Runnable function for constraint extraction.

    Args:
        input_data: Dictionary with "user_input" key

    Returns:
        Input data with "constraints" key added
    """
    extractor = ConstraintExtractor()
    user_input = input_data.get("user_input", "")

    constraints = extractor.extract_constraints(user_input)

    return {**input_data, "constraints": constraints}


# Create LangChain Runnable
ConstraintExtractorChain = RunnableLambda(extract_constraints_runnable)

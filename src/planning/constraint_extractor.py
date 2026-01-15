"""Extract meal planning constraints from user input."""

import re
from datetime import date, timedelta
from pathlib import Path

from src.app.logging_config import get_logger
from src.domain.models import (
    DietaryRestriction,
    ExtractionSource,
    ExtractedValue,
    IngredientCategory,
    MealPlanConstraints,
    PreferenceProfile,
)
from src.planning.exclusion_vocabulary import is_valid_exclusion_term

logger = get_logger(__name__)


class MealPlanConstraintExtractor:
    """Extract meal plan constraints from user input with audit trail."""

    # Time patterns (extracted FIRST to avoid false positives)
    TIME_PATTERNS = [
        # "30 minutes", "30 min", "30 mins"
        (r"(\d+)\s*(?:minutes?|mins?)", 1),
        # "under 30 min", "less than 30 minutes"
        (r"(?:under|less\s+than)\s*(\d+)\s*(?:minutes?|mins?)", 1),
        # "no more than 30 minutes"
        (r"no\s+more\s+than\s*(\d+)\s*(?:minutes?|mins?)", 1),
        # "quick" = 30 min, "fast" = 30 min
        (r"\b(quick|fast)\b", 30),
    ]

    # Day patterns (order matters - more specific first)
    DAY_PATTERNS = [
        (r"(\d+)\s*(?:days?|nights?)", 1),  # "5 days", "5 nights"
        (r"(\d+)\s+\w+\s*(?:dinners?|meals?|lunches?|breakfasts?)", 1),  # "5 vegetarian dinners"
        (r"weeknight", 5),  # "weeknight" = 5 days (Mon-Fri) - BEFORE "week"
        (r"monday\s*(?:through|to|-)\s*friday", 5),
        (r"mon\s*-\s*fri", 5),
        (r"\bnext\s+week\b", 7),  # "next week" - for start date, not days
        (r"\bthe\s+week\b", 7),  # "the week"
        (r"\bweek\b(?!\s*night)", 7),  # "week" but not "weeknight"
    ]

    # Dietary patterns
    DIETARY_PATTERNS = {
        r"\bvegetarian\b": DietaryRestriction.VEGETARIAN,
        r"\bvegan\b": DietaryRestriction.VEGAN,
        r"\bpescatarian\b": DietaryRestriction.PESCATARIAN,
        r"\bketo\b": DietaryRestriction.KETO,
        r"\bgluten[- ]?free\b": DietaryRestriction.GLUTEN_FREE,
        r"\bdairy[- ]?free\b": DietaryRestriction.DAIRY_FREE,
    }

    # Category keywords for "no X" patterns
    CATEGORY_KEYWORDS = {
        "dairy": IngredientCategory.DAIRY,
        "meat": IngredientCategory.MEAT,
        "seafood": IngredientCategory.SEAFOOD,
        "fish": IngredientCategory.SEAFOOD,
        "gluten": IngredientCategory.GLUTEN,
        "nuts": IngredientCategory.NUTS,
        "eggs": IngredientCategory.EGGS,
        "poultry": IngredientCategory.POULTRY,
        "pork": IngredientCategory.MEAT,
        "beef": IngredientCategory.MEAT,
        "chicken": IngredientCategory.POULTRY,
        "turkey": IngredientCategory.POULTRY,
    }

    # Dish type tags
    DISH_TAGS = {
        "casserole",
        "casseroles",
        "soup",
        "soups",
        "stew",
        "stews",
        "salad",
        "salads",
        "sandwich",
        "sandwiches",
        "pizza",
        "pizzas",
        "curry",
        "curries",
        "pasta",
    }

    # Meal type patterns
    MEAL_TYPE_PATTERNS = {
        r"\bbreakfast": "breakfast",
        r"\blunch": "lunch",
        r"\bdinner": "dinner",
        r"\bbrunch": "breakfast",  # Map brunch to breakfast
    }

    def __init__(self, db_path: Path | None = None):
        """Initialize extractor with optional database path for vocabulary validation.

        Args:
            db_path: Path to SQLite database for exclusion vocabulary validation
        """
        self.db_path = db_path

    def extract(
        self,
        user_input: str,
        profile: PreferenceProfile | None = None,
    ) -> MealPlanConstraints:
        """Extract constraints from user input with source tracking.

        Args:
            user_input: User's natural language input
            profile: User's preference profile (for defaults)

        Returns:
            MealPlanConstraints with extraction_sources populated
        """
        constraints = MealPlanConstraints()
        sources: dict[str, ExtractedValue] = {}

        text = user_input.lower()

        # 1. Extract time FIRST (before exclusions to avoid "no more than 30" confusion)
        max_time = self._extract_time(text)
        if max_time:
            constraints.max_prep_time = max_time
            sources["max_prep_time"] = ExtractedValue(
                value=max_time, source=ExtractionSource.RULE
            )

        # 2. Extract days
        days = self._extract_days(text)
        if days:
            constraints.days = days
            sources["days"] = ExtractedValue(value=days, source=ExtractionSource.RULE)

        # 3. Extract dietary restriction
        dietary = self._extract_dietary(text, profile)
        if dietary:
            constraints.dietary = dietary
            sources["dietary"] = ExtractedValue(
                value=dietary.value,
                source=(
                    ExtractionSource.RULE
                    if self._has_dietary_pattern(text)
                    else ExtractionSource.USER_PROFILE
                ),
            )

        # 4. Extract meal types
        meal_types = self._extract_meal_types(text)
        if meal_types:
            constraints.meal_types = meal_types
            sources["meal_types"] = ExtractedValue(
                value=meal_types, source=ExtractionSource.RULE
            )

        # 5. Extract exclusions (AFTER time to avoid false positives)
        excluded_ings, excluded_tags, excluded_cats = self._extract_exclusions(text)
        if excluded_ings:
            constraints.excluded_ingredients = excluded_ings
        if excluded_tags:
            constraints.excluded_tags = excluded_tags
        if excluded_cats:
            constraints.excluded_categories = excluded_cats

        if excluded_ings or excluded_tags or excluded_cats:
            sources["exclusions"] = ExtractedValue(
                value={
                    "ingredients": excluded_ings,
                    "tags": excluded_tags,
                    "categories": [c.value for c in excluded_cats],
                },
                source=ExtractionSource.RULE,
            )

        # 6. Extract start date
        start_date = self._extract_start_date(text)
        if start_date:
            constraints.start_date = start_date
            sources["start_date"] = ExtractedValue(
                value=start_date.isoformat(), source=ExtractionSource.RULE
            )

        # Apply profile defaults where not explicitly specified
        if profile:
            if "dietary" not in sources and profile.diet != "none":
                try:
                    constraints.dietary = DietaryRestriction(profile.diet)
                    sources["dietary"] = ExtractedValue(
                        value=profile.diet, source=ExtractionSource.USER_PROFILE
                    )
                except ValueError:
                    pass

        constraints.extraction_sources = sources

        logger.info(
            "Extracted meal plan constraints",
            days=constraints.days,
            dietary=constraints.dietary.value,
            max_prep_time=constraints.max_prep_time,
            excluded_count=len(excluded_ings) + len(excluded_tags) + len(excluded_cats),
        )

        return constraints

    def _extract_time(self, text: str) -> int | None:
        """Extract time constraint from text.

        Args:
            text: Lowercase user input

        Returns:
            Time in minutes or None
        """
        for pattern, group_or_value in self.TIME_PATTERNS:
            match = re.search(pattern, text)
            if match:
                if isinstance(group_or_value, int) and group_or_value > 1:
                    # It's a fixed value (e.g., "quick" = 30)
                    return group_or_value
                elif match.groups():
                    # Extract from regex group
                    return int(match.group(1))
        return None

    def _extract_days(self, text: str) -> int | None:
        """Extract number of days from text.

        Explicit numbers take precedence over keywords like "week".

        Args:
            text: Lowercase user input

        Returns:
            Number of days or None (uses default)
        """
        explicit_days = None
        keyword_days = None

        for pattern, group_or_value in self.DAY_PATTERNS:
            match = re.search(pattern, text)
            if match:
                if match.groups():
                    # Explicit numeric match
                    explicit_days = int(match.group(1))
                    break  # Explicit numbers take priority
                elif isinstance(group_or_value, int) and group_or_value > 1:
                    # Keyword match (week, weeknight, etc.)
                    if keyword_days is None:
                        keyword_days = group_or_value

        # Explicit numbers override keywords
        return explicit_days if explicit_days is not None else keyword_days

    def _extract_dietary(
        self, text: str, profile: PreferenceProfile | None
    ) -> DietaryRestriction | None:
        """Extract dietary restriction from text.

        Args:
            text: Lowercase user input
            profile: User profile for defaults

        Returns:
            DietaryRestriction or None
        """
        for pattern, restriction in self.DIETARY_PATTERNS.items():
            if re.search(pattern, text):
                return restriction

        # Fall back to profile
        if profile and profile.diet != "none":
            try:
                return DietaryRestriction(profile.diet)
            except ValueError:
                pass

        return None

    def _has_dietary_pattern(self, text: str) -> bool:
        """Check if text contains a dietary pattern.

        Args:
            text: Lowercase user input

        Returns:
            True if dietary pattern found
        """
        for pattern in self.DIETARY_PATTERNS:
            if re.search(pattern, text):
                return True
        return False

    def _extract_meal_types(self, text: str) -> list[str] | None:
        """Extract meal types from text.

        Args:
            text: Lowercase user input

        Returns:
            List of meal types or None (uses default)
        """
        found_types = []
        for pattern, meal_type in self.MEAL_TYPE_PATTERNS.items():
            if re.search(pattern, text):
                if meal_type not in found_types:
                    found_types.append(meal_type)

        return found_types if found_types else None

    def _extract_exclusions(
        self, text: str
    ) -> tuple[list[str], list[str], list[IngredientCategory]]:
        """Extract exclusions from text with vocabulary validation.

        Args:
            text: Lowercase user input

        Returns:
            Tuple of (excluded_ingredients, excluded_tags, excluded_categories)
        """
        ingredients: list[str] = []
        tags: list[str] = []
        categories: list[IngredientCategory] = []

        # Patterns: "no X", "without X", "avoid X", "but not X"
        # Use non-greedy matching and stop at common separators
        exclusion_patterns = [
            r"\bno\s+(\w+)",  # Simpler: just capture single word
            r"\bwithout\s+(\w+)",
            r"\bavoid\s+(\w+)",
            r"\bbut\s+not\s+(\w+)",
            r"\bexclude\s+(\w+)",
        ]

        for pattern in exclusion_patterns:
            for match in re.finditer(pattern, text):
                term = match.group(1).strip()

                # Skip time-related phrases (handled by time extraction)
                if term in ("more", "less", "than"):
                    continue
                if re.match(r"more\s+than", term):
                    continue

                # Check if it's a category keyword
                if term in self.CATEGORY_KEYWORDS:
                    cat = self.CATEGORY_KEYWORDS[term]
                    if cat not in categories:
                        categories.append(cat)
                    continue

                # Check if it's a dish type (tag)
                term_singular = term.rstrip("s") if term.endswith("s") else term
                if term_singular in self.DISH_TAGS or term in self.DISH_TAGS:
                    if term_singular not in tags:
                        tags.append(term_singular)
                    continue

                # Validate against database vocabulary if available
                if self.db_path:
                    if not is_valid_exclusion_term(term, self.db_path):
                        logger.debug(
                            "Exclusion term not in vocabulary",
                            term=term,
                        )
                        continue

                # It's an ingredient exclusion
                if term not in ingredients:
                    ingredients.append(term)

        return ingredients, tags, categories

    def _extract_start_date(self, text: str) -> date | None:
        """Extract start date from text.

        Args:
            text: Lowercase user input

        Returns:
            Start date or None (uses today)
        """
        today = date.today()

        # "starting tomorrow"
        if "tomorrow" in text:
            return today + timedelta(days=1)

        # "starting monday", "next monday"
        weekdays = {
            "monday": 0,
            "tuesday": 1,
            "wednesday": 2,
            "thursday": 3,
            "friday": 4,
            "saturday": 5,
            "sunday": 6,
        }

        for day_name, day_num in weekdays.items():
            if day_name in text:
                # Calculate next occurrence of this day
                current_day = today.weekday()
                days_ahead = day_num - current_day
                if days_ahead <= 0:  # Target day already happened this week
                    days_ahead += 7
                return today + timedelta(days=days_ahead)

        # "next week"
        if "next week" in text:
            # Next Monday
            current_day = today.weekday()
            days_ahead = 7 - current_day
            return today + timedelta(days=days_ahead)

        return None

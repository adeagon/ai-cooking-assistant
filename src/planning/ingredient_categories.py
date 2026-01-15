"""Ingredient category classification using n-gram matching.

This classifier assigns ingredients to categories like dairy, meat, seafood, etc.
for use in dietary exclusion filtering. It uses n-gram matching to avoid
substring false positives.
"""

from src.app.logging_config import get_logger
from src.domain.models import IngredientCategory

logger = get_logger(__name__)


class IngredientCategoryClassifier:
    """Classify ingredients into categories using n-gram matching.

    NOTE: This is a best-effort classifier. Coverage is not 100%.
    Misses are logged for future improvement.
    """

    CATEGORY_KEYWORDS: dict[IngredientCategory, set[str]] = {
        IngredientCategory.DAIRY: {
            "milk",
            "cheese",
            "cream",
            "butter",
            "yogurt",
            "sour_cream",
            "parmesan",
            "mozzarella",
            "cheddar",
            "feta",
            "ricotta",
            "cottage_cheese",
            "cream_cheese",
            "half_and_half",
            "whey",
            "heavy_cream",
            "buttermilk",
            "ghee",
            "mascarpone",
            "brie",
            "gouda",
            "swiss",
            "provolone",
            "gruyere",
            "goat_cheese",
            "blue_cheese",
            "queso",
        },
        IngredientCategory.MEAT: {
            "beef",
            "pork",
            "lamb",
            "veal",
            "bacon",
            "ham",
            "sausage",
            "steak",
            "ground_beef",
            "ground_pork",
            "roast",
            "ribs",
            "brisket",
            "prosciutto",
            "pancetta",
            "salami",
            "pepperoni",
            "chorizo",
            "hot_dog",
            "meatball",
            "tenderloin",
            "sirloin",
            "chuck",
            "flank",
        },
        IngredientCategory.POULTRY: {
            "chicken",
            "turkey",
            "duck",
            "goose",
            "cornish_hen",
            "chicken_breast",
            "chicken_thigh",
            "ground_chicken",
            "ground_turkey",
            "chicken_wing",
            "drumstick",
            "quail",
            "pheasant",
        },
        IngredientCategory.SEAFOOD: {
            "fish",
            "salmon",
            "tuna",
            "shrimp",
            "crab",
            "lobster",
            "scallop",
            "clam",
            "mussel",
            "oyster",
            "cod",
            "tilapia",
            "halibut",
            "trout",
            "anchovy",
            "sardine",
            "prawn",
            "calamari",
            "squid",
            "octopus",
            "snapper",
            "mahi",
            "swordfish",
            "catfish",
            "bass",
            "perch",
            "mackerel",
            "haddock",
            "sole",
            "flounder",
        },
        IngredientCategory.GLUTEN: {
            "flour",
            "bread",
            "pasta",
            "noodle",
            "wheat",
            "barley",
            "rye",
            "couscous",
            "breadcrumb",
            "bread_crumbs",
            "crouton",
            "tortilla",
            "pita",
            "spaghetti",
            "linguine",
            "fettuccine",
            "lasagna",
            "macaroni",
            "penne",
            "ravioli",
            "orzo",
            "bulgur",
            "seitan",
            "panko",
            "panko_bread_crumbs",
        },
        IngredientCategory.NUTS: {
            "almond",
            "walnut",
            "pecan",
            "cashew",
            "peanut",
            "pistachio",
            "hazelnut",
            "macadamia",
            "pine_nut",
            "chestnut",
            "peanut_butter",
            "almond_butter",
            "almond_milk",
            "cashew_milk",
            "nut",
        },
        IngredientCategory.SOY: {
            "soy",
            "tofu",
            "tempeh",
            "edamame",
            "soy_sauce",
            "miso",
            "soybean",
            "soy_milk",
        },
        IngredientCategory.EGGS: {
            "egg",
            "yolk",
            "whites",
            "egg_white",
            "egg_yolk",
            "mayonnaise",
            "mayo",
        },
    }

    def __init__(self) -> None:
        """Initialize classifier with empty miss log."""
        self._miss_log: set[str] = set()

    def _get_ngrams(self, normalized: str, max_n: int = 3) -> set[str]:
        """Get 1-grams, 2-grams, and 3-grams from normalized text.

        Args:
            normalized: Normalized ingredient string (may contain underscores)
            max_n: Maximum n-gram size to generate

        Returns:
            Set of n-grams from the text
        """
        tokens = normalized.split()
        ngrams = set(tokens)  # 1-grams

        for n in range(2, min(max_n + 1, len(tokens) + 1)):
            for i in range(len(tokens) - n + 1):
                ngrams.add("_".join(tokens[i : i + n]))

        return ngrams

    def classify(self, normalized_ingredient: str) -> set[IngredientCategory]:
        """Return categories that this ingredient belongs to.

        Uses n-gram matching to avoid substring false positives.
        Logs unclassified ingredients for future improvement.

        Args:
            normalized_ingredient: Already-normalized ingredient string

        Returns:
            Set of categories the ingredient belongs to
        """
        categories: set[IngredientCategory] = set()
        ngrams = self._get_ngrams(normalized_ingredient)

        for category, keywords in self.CATEGORY_KEYWORDS.items():
            if ngrams & keywords:
                categories.add(category)

        # Log miss for future vocabulary expansion (once per ingredient)
        if not categories and normalized_ingredient not in self._miss_log:
            self._miss_log.add(normalized_ingredient)
            logger.debug(
                "Ingredient not classified into any category",
                ingredient=normalized_ingredient,
            )

        return categories

    def contains_excluded_category(
        self,
        normalized_ingredient: str,
        excluded_categories: list[IngredientCategory],
    ) -> bool:
        """Check if ingredient belongs to any excluded category.

        Args:
            normalized_ingredient: Already-normalized ingredient string
            excluded_categories: List of categories to exclude

        Returns:
            True if ingredient is in any excluded category
        """
        if not excluded_categories:
            return False
        ingredient_categories = self.classify(normalized_ingredient)
        return bool(ingredient_categories & set(excluded_categories))

    def get_primary_category(
        self, normalized_ingredient: str
    ) -> IngredientCategory | None:
        """Get the primary (most specific) category for an ingredient.

        Useful for grocery list organization.

        Args:
            normalized_ingredient: Already-normalized ingredient string

        Returns:
            Primary category or None if not classified
        """
        categories = self.classify(normalized_ingredient)

        if not categories:
            return None

        # Priority order for grocery organization
        priority = [
            IngredientCategory.SEAFOOD,
            IngredientCategory.POULTRY,
            IngredientCategory.MEAT,
            IngredientCategory.DAIRY,
            IngredientCategory.EGGS,
            IngredientCategory.NUTS,
            IngredientCategory.GLUTEN,
            IngredientCategory.SOY,
        ]

        for cat in priority:
            if cat in categories:
                return cat

        # Return any category if not in priority list
        return next(iter(categories))

    def get_miss_count(self) -> int:
        """Get count of unclassified ingredients seen.

        Useful for monitoring classification coverage.

        Returns:
            Number of unique ingredients that weren't classified
        """
        return len(self._miss_log)

    def get_missed_ingredients(self) -> set[str]:
        """Get set of unclassified ingredients.

        Useful for debugging and vocabulary expansion.

        Returns:
            Set of ingredient strings that weren't classified
        """
        return self._miss_log.copy()

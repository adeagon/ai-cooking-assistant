"""Token-based ingredient normalization with phrase preservation."""

import re


class IngredientNormalizer:
    """Token-based ingredient normalization with phrase preservation.

    Uses regex word boundaries for safe phrase preservation and token-level
    stop word filtering to avoid false positives.
    """

    # Stop tokens (checked at token level)
    STOP_TOKENS = {
        "salt",
        "pepper",
        "water",
        "oil",
        "butter",
        "spray",
    }

    # Stop phrase tokens (already underscored, checked as tokens)
    # NOTE: Named STOP_PHRASE_TOKENS not STOP_PHRASES - these are tokens after phrase preservation
    STOP_PHRASE_TOKENS = {
        "olive_oil",
        "vegetable_oil",
        "cooking_spray",
        "black_pepper",
        "kosher_salt",
        "sea_salt",
        "white_pepper",
        "canola_oil",
        "sesame_oil",
        "soy_sauce",
        "fish_sauce",
    }

    # Known bigrams to preserve as single tokens
    # Applied using regex word boundaries, NOT text.replace()
    # Order matters: longer/more specific first (includes plural forms)
    PRESERVE_PHRASES = [
        # Plurals first (more specific)
        ("green onions", "green_onion"),
        ("green onion", "green_onion"),
        ("chicken breasts", "chicken_breast"),
        ("chicken breast", "chicken_breast"),
        ("chicken thighs", "chicken_thigh"),
        ("chicken thigh", "chicken_thigh"),
        ("bell peppers", "bell_pepper"),
        ("bell pepper", "bell_pepper"),
        ("garlic cloves", "garlic_clove"),
        ("garlic clove", "garlic_clove"),
        ("bay leaves", "bay_leaf"),
        ("bay leaf", "bay_leaf"),
        ("pine nuts", "pine_nut"),
        ("pine nut", "pine_nut"),
        ("red onions", "red_onion"),
        ("red onion", "red_onion"),
        ("yellow onions", "yellow_onion"),
        ("yellow onion", "yellow_onion"),
        ("white onions", "white_onion"),
        ("white onion", "white_onion"),
        ("red peppers", "red_pepper"),
        ("red pepper", "red_pepper"),
        # Standard phrases
        ("soy sauce", "soy_sauce"),
        ("sour cream", "sour_cream"),
        ("olive oil", "olive_oil"),
        ("vegetable oil", "vegetable_oil"),
        ("sesame oil", "sesame_oil"),
        ("cream cheese", "cream_cheese"),
        ("ground beef", "ground_beef"),
        ("ground turkey", "ground_turkey"),
        ("ground pork", "ground_pork"),
        ("ground chicken", "ground_chicken"),
        ("black pepper", "black_pepper"),
        ("white pepper", "white_pepper"),
        ("fish sauce", "fish_sauce"),
        ("rice vinegar", "rice_vinegar"),
        ("balsamic vinegar", "balsamic_vinegar"),
        ("tomato paste", "tomato_paste"),
        ("tomato sauce", "tomato_sauce"),
        ("coconut milk", "coconut_milk"),
        ("heavy cream", "heavy_cream"),
        ("peanut butter", "peanut_butter"),
        ("almond butter", "almond_butter"),
        ("chili powder", "chili_powder"),
        ("curry powder", "curry_powder"),
        ("garlic powder", "garlic_powder"),
        ("onion powder", "onion_powder"),
        ("cayenne pepper", "cayenne_pepper"),
        ("lemon juice", "lemon_juice"),
        ("lime juice", "lime_juice"),
        ("orange juice", "orange_juice"),
        ("cream of tartar", "cream_of_tartar"),
        ("baking powder", "baking_powder"),
        ("baking soda", "baking_soda"),
        ("brown sugar", "brown_sugar"),
        ("powdered sugar", "powdered_sugar"),
        ("maple syrup", "maple_syrup"),
        ("corn starch", "corn_starch"),
        ("cornstarch", "corn_starch"),
        ("bread crumbs", "bread_crumbs"),
        ("breadcrumbs", "bread_crumbs"),
        ("panko bread crumbs", "panko_bread_crumbs"),
    ]

    # Synonym mapping (canonical forms) - applied after phrase preservation
    SYNONYMS = [
        ("scallion", "green_onion"),
        ("scallions", "green_onion"),
        ("spring onion", "green_onion"),
        ("spring onions", "green_onion"),
        ("capsicum", "bell_pepper"),
        ("coriander", "cilantro"),
        ("aubergine", "eggplant"),
        ("courgette", "zucchini"),
        ("courgettes", "zucchini"),
        ("rocket", "arugula"),
        ("prawns", "shrimp"),
        ("prawn", "shrimp"),
        ("gammon", "ham"),
        ("mince", "ground_beef"),
        ("minced beef", "ground_beef"),
        ("minced pork", "ground_pork"),
        ("minced turkey", "ground_turkey"),
    ]

    # Whitelist for safe plural→singular (only these, no naive rules)
    PLURAL_WHITELIST = {
        "tomatoes": "tomato",
        "potatoes": "potato",
        "onions": "onion",
        "peppers": "pepper",
        "carrots": "carrot",
        "mushrooms": "mushroom",
        "cloves": "clove",
        "eggs": "egg",
        "lemons": "lemon",
        "limes": "lime",
        "oranges": "orange",
        "apples": "apple",
        "berries": "berry",
        "cherries": "cherry",
        "olives": "olive",
        "noodles": "noodle",
        "tortillas": "tortilla",
        "beans": "bean",
        "peas": "pea",
        "chickpeas": "chickpea",
        "celeries": "celery",
        "anchovies": "anchovy",
        "strawberries": "strawberry",
        "blueberries": "blueberry",
        "raspberries": "raspberry",
        "blackberries": "blackberry",
        "cranberries": "cranberry",
        "zucchinis": "zucchini",
        "cucumbers": "cucumber",
        "avocados": "avocado",
        "mangoes": "mango",
        "mangos": "mango",
        "bananas": "banana",
        "peaches": "peach",
        "pears": "pear",
        "plums": "plum",
        "grapes": "grape",
        "shallots": "shallot",
        "radishes": "radish",
        "spinaches": "spinach",
        "lettuces": "lettuce",
        "cabbages": "cabbage",
        "cauliflowers": "cauliflower",
        "broccolis": "broccoli",
        "asparaguses": "asparagus",
        "eggplants": "eggplant",
        "shrimps": "shrimp",
        "clams": "clam",
        "mussels": "mussel",
        "scallops": "scallop",
        "oysters": "oyster",
    }

    # Modifiers to strip from ingredients
    MODIFIERS = {
        # Preparation methods
        "diced",
        "chopped",
        "minced",
        "sliced",
        "fresh",
        "dried",
        "canned",
        "frozen",
        "boneless",
        "skinless",
        "lean",
        "finely",
        "coarsely",
        "roughly",
        "thinly",
        "julienned",
        "shredded",
        "grated",
        "cubed",
        "crushed",
        "peeled",
        "seeded",
        "pitted",
        "trimmed",
        "rinsed",
        "drained",
        "packed",
        "loosely",
        "firmly",
        "softened",
        "melted",
        # Size/quantity modifiers
        "large",
        "small",
        "medium",
        "thin",
        "thick",
        "whole",
        "half",
        # Units - full names
        "cup",
        "cups",
        "tablespoon",
        "tablespoons",
        "teaspoon",
        "teaspoons",
        "pound",
        "pounds",
        "ounce",
        "ounces",
        # Units - abbreviations
        "oz",
        "lb",
        "lbs",
        "tbsp",
        "tsp",
        "ml",
        "g",
        "kg",
        "c",
        "pt",
        "qt",
        "gal",
        # Quality modifiers
        "extra",
        "virgin",
        "organic",
        "raw",
        "cooked",
        "uncooked",
        "unsalted",
        "salted",
        "sweetened",
        "unsweetened",
        "reduced",
        "low",
        "fat",
        "free",
        "light",
        "lite",
        # Temperature
        "room",
        "temperature",
        "cold",
        "warm",
        "hot",
        "chilled",
        # Misc
        "optional",
        "divided",
        "plus",
        "more",
        "for",
        "garnish",
        "to",
        "taste",
        "needed",
        "as",
        "about",
        "approximately",
    }

    def normalize(self, ingredient: str) -> str:
        """Normalize ingredient for comparison (returns canonical tokens with phrases).

        Args:
            ingredient: Raw ingredient string to normalize

        Returns:
            Normalized string with preserved phrases and canonical tokens
        """
        text = ingredient.lower().strip()

        # Remove quantity patterns (e.g., "2 cups", "1/2 lb")
        text = re.sub(r"^\d+[\d/.\s]*", "", text).strip()

        # Preserve known bigrams FIRST (before stripping modifiers that might break phrases)
        # Use regex word boundaries for safety
        for phrase, token in self.PRESERVE_PHRASES:
            text = re.sub(rf"\b{re.escape(phrase)}\b", token, text)

        # Remove common modifiers
        text = self._strip_modifiers(text)

        # Apply synonym replacement (word boundary safe)
        for syn, canonical in self.SYNONYMS:
            text = re.sub(rf"\b{re.escape(syn)}\b", canonical, text)

        # Apply whitelist singularization
        tokens = text.split()
        tokens = [self.PLURAL_WHITELIST.get(t, t) for t in tokens]

        # Remove empty tokens and rejoin
        tokens = [t for t in tokens if t]
        return " ".join(tokens)

    def _strip_modifiers(self, text: str) -> str:
        """Remove common modifiers (diced, fresh, etc.)."""
        tokens = text.split()
        tokens = [t for t in tokens if t not in self.MODIFIERS]
        return " ".join(tokens)

    def is_stop_ingredient(self, normalized: str) -> bool:
        """Check if ingredient should be excluded from overlap scoring.

        Stop ingredients are common staples that appear in most recipes
        and shouldn't contribute to overlap calculations.

        Args:
            normalized: Already-normalized ingredient string

        Returns:
            True if this is a stop ingredient
        """
        tokens = set(normalized.split())

        # Check phrase tokens (already underscored)
        if tokens & self.STOP_PHRASE_TOKENS:
            return True

        # Check individual tokens
        return bool(tokens & self.STOP_TOKENS)

    def tokenize(self, normalized: str) -> set[str]:
        """Get ingredient tokens for overlap comparison.

        Preserved phrases stay as single tokens (e.g., "green_onion").
        Stop ingredients are excluded from the result.

        Args:
            normalized: Already-normalized ingredient string

        Returns:
            Set of tokens for comparison, excluding stop ingredients
        """
        tokens = set(normalized.split())
        # Remove stop tokens and stop phrase tokens
        return tokens - self.STOP_TOKENS - self.STOP_PHRASE_TOKENS

    def get_tokens_with_stops(self, normalized: str) -> set[str]:
        """Get all tokens including stop ingredients.

        Useful for grocery list generation where we want all ingredients.

        Args:
            normalized: Already-normalized ingredient string

        Returns:
            Set of all tokens including stop ingredients
        """
        return set(normalized.split())

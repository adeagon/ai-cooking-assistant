"""Grocery list generation from meal plans."""

from collections import defaultdict
from datetime import datetime

from src.app.logging_config import get_logger
from src.domain.models import GroceryItem, GroceryList, MealPlan, Recipe
from src.planning.ingredient_categories import IngredientCategoryClassifier
from src.planning.ingredient_normalizer import IngredientNormalizer

logger = get_logger(__name__)


# Grocery category mapping (for display grouping)
GROCERY_CATEGORIES = {
    "produce": {
        "tomato",
        "onion",
        "garlic",
        "broccoli",
        "carrot",
        "celery",
        "bell_pepper",
        "potato",
        "lettuce",
        "spinach",
        "mushroom",
        "zucchini",
        "cucumber",
        "avocado",
        "lemon",
        "lime",
        "orange",
        "apple",
        "banana",
        "berry",
        "ginger",
        "cilantro",
        "basil",
        "parsley",
        "green_onion",
        "kale",
        "cabbage",
        "cauliflower",
        "asparagus",
        "corn",
        "eggplant",
        "pea",
        "bean",
        "squash",
        "jalapeno",
    },
    "protein": {
        "chicken",
        "chicken_breast",
        "chicken_thigh",
        "beef",
        "ground_beef",
        "pork",
        "ground_pork",
        "turkey",
        "ground_turkey",
        "lamb",
        "salmon",
        "tuna",
        "shrimp",
        "fish",
        "bacon",
        "sausage",
        "ham",
        "tofu",
        "tempeh",
        "egg",
    },
    "dairy": {
        "milk",
        "cheese",
        "butter",
        "cream",
        "yogurt",
        "sour_cream",
        "cream_cheese",
        "parmesan",
        "mozzarella",
        "cheddar",
        "feta",
        "heavy_cream",
        "buttermilk",
    },
    "pantry": {
        "pasta",
        "rice",
        "noodle",
        "flour",
        "sugar",
        "brown_sugar",
        "honey",
        "maple_syrup",
        "soy_sauce",
        "vinegar",
        "oil",
        "olive_oil",
        "vegetable_oil",
        "sesame_oil",
        "stock",
        "broth",
        "tomato_sauce",
        "tomato_paste",
        "coconut_milk",
        "canned",
        "bread_crumbs",
        "tortilla",
        "bread",
    },
    "spices": {
        "salt",
        "pepper",
        "cumin",
        "paprika",
        "oregano",
        "thyme",
        "rosemary",
        "cinnamon",
        "nutmeg",
        "chili_powder",
        "curry_powder",
        "garlic_powder",
        "onion_powder",
        "cayenne_pepper",
        "turmeric",
        "coriander",
        "bay_leaf",
        "red_pepper_flakes",
    },
}


class GroceryListGenerator:
    """Generate grocery lists from meal plans."""

    def __init__(self) -> None:
        """Initialize generator with normalizer and classifier."""
        self.normalizer = IngredientNormalizer()
        self.category_classifier = IngredientCategoryClassifier()

    def generate(
        self,
        plan: MealPlan,
        recipes: dict[str, Recipe],
        exclude_pantry_staples: bool = True,
    ) -> GroceryList:
        """Generate a grocery list from a meal plan.

        Args:
            plan: The meal plan to generate a list for
            recipes: Dictionary mapping recipe_id to Recipe objects
            exclude_pantry_staples: If True, exclude common staples (salt, pepper, oil)

        Returns:
            GroceryList with aggregated items
        """
        # Aggregate ingredients by normalized form
        ingredient_map: dict[str, dict] = defaultdict(
            lambda: {"original": set(), "recipes": set(), "normalized": ""}
        )

        for meal in plan.meals:
            recipe = recipes.get(meal.recipe_id)
            if not recipe:
                logger.warning(
                    "Recipe not found for meal",
                    recipe_id=meal.recipe_id,
                    title=meal.title,
                )
                continue

            for ingredient in recipe.ingredients_normalized:
                normalized = self.normalizer.normalize(ingredient)

                # Skip stop ingredients if excluding pantry staples
                if exclude_pantry_staples and self.normalizer.is_stop_ingredient(normalized):
                    continue

                ingredient_map[normalized]["original"].add(ingredient)
                ingredient_map[normalized]["recipes"].add(meal.title)
                ingredient_map[normalized]["normalized"] = normalized

        # Convert to GroceryItem objects
        items: list[GroceryItem] = []
        for normalized, data in ingredient_map.items():
            # Find the best display name (shortest original)
            originals = data["original"]
            display = min(originals, key=len) if originals else normalized

            # Determine category
            category = self._get_grocery_category(normalized)

            items.append(
                GroceryItem(
                    ingredient=display,
                    normalized=normalized,
                    recipes=sorted(data["recipes"]),
                    category=category,
                )
            )

        # Sort items by category then alphabetically
        category_order = ["produce", "protein", "dairy", "pantry", "spices", "other"]
        items.sort(
            key=lambda x: (
                category_order.index(x.category) if x.category in category_order else len(category_order),
                x.ingredient.lower(),
            )
        )

        logger.info(
            "Generated grocery list",
            plan_id=plan.id,
            item_count=len(items),
        )

        return GroceryList(
            plan_id=plan.id or 0,
            items=items,
            generated_at=datetime.now(),
        )

    def _get_grocery_category(self, normalized: str) -> str:
        """Determine grocery category for an ingredient.

        Args:
            normalized: Normalized ingredient string

        Returns:
            Category name for grocery list grouping
        """
        tokens = self.normalizer.get_tokens_with_stops(normalized)

        # Check each category
        for category, keywords in GROCERY_CATEGORIES.items():
            if tokens & keywords:
                return category

        # Check ingredient category classifier for protein/dairy
        ingredient_cats = self.category_classifier.classify(normalized)

        from src.domain.models import IngredientCategory

        if IngredientCategory.MEAT in ingredient_cats or IngredientCategory.POULTRY in ingredient_cats:
            return "protein"
        if IngredientCategory.SEAFOOD in ingredient_cats:
            return "protein"
        if IngredientCategory.DAIRY in ingredient_cats:
            return "dairy"
        if IngredientCategory.EGGS in ingredient_cats:
            return "dairy"  # Group eggs with dairy for shopping

        return "other"

    def format_for_display(
        self,
        grocery_list: GroceryList,
        show_recipes: bool = True,
        group_by_category: bool = True,
    ) -> str:
        """Format grocery list for display.

        Args:
            grocery_list: The grocery list to format
            show_recipes: Whether to show which recipes need each item
            group_by_category: Whether to group items by category

        Returns:
            Formatted string for display
        """
        if not grocery_list.items:
            return "No items in grocery list."

        lines: list[str] = []

        if group_by_category:
            # Group items by category
            by_category: dict[str, list[GroceryItem]] = defaultdict(list)
            for item in grocery_list.items:
                by_category[item.category or "other"].append(item)

            # Display order
            category_order = ["produce", "protein", "dairy", "pantry", "spices", "other"]

            for category in category_order:
                if category not in by_category:
                    continue

                items = by_category[category]
                lines.append(f"\n**{category.title()}**")

                for item in items:
                    if show_recipes and item.recipes:
                        recipe_list = ", ".join(item.recipes[:3])
                        if len(item.recipes) > 3:
                            recipe_list += f" (+{len(item.recipes) - 3} more)"
                        lines.append(f"- {item.ingredient} ({recipe_list})")
                    else:
                        lines.append(f"- {item.ingredient}")
        else:
            for item in grocery_list.items:
                if show_recipes and item.recipes:
                    recipe_list = ", ".join(item.recipes[:3])
                    lines.append(f"- {item.ingredient} ({recipe_list})")
                else:
                    lines.append(f"- {item.ingredient}")

        return "\n".join(lines)

    def get_summary(self, grocery_list: GroceryList) -> dict[str, int]:
        """Get summary counts by category.

        Args:
            grocery_list: The grocery list to summarize

        Returns:
            Dictionary mapping category to item count
        """
        summary: dict[str, int] = defaultdict(int)
        for item in grocery_list.items:
            summary[item.category or "other"] += 1
        return dict(summary)

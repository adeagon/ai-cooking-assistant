"""Recipe card generation for LLM context."""

from pathlib import Path
from src.domain.models import Recipe, RecipeCard, RetrievalResult
from src.ingest.build_db import get_recipe_by_id
from src.app.logging_config import get_logger

logger = get_logger(__name__)


class RecipeCardBuilder:
    """
    Builds compact RecipeCard objects for LLM prompts.

    Transforms full Recipe objects into compact representations
    suitable for inclusion in LLM context (120-250 tokens target).
    """

    def __init__(self, db_path: Path):
        """
        Initialize the card builder.

        Args:
            db_path: Path to SQLite database containing recipes
        """
        self.db_path = db_path
        logger.info(f"RecipeCardBuilder initialized with DB: {db_path}")

    def build_cards(
        self,
        results: list[RetrievalResult],
        query: str,
        max_cards: int = 6
    ) -> list[RecipeCard]:
        """
        Build RecipeCards from retrieval results.

        Args:
            results: Reranked RetrievalResult objects
            query: Original user query (for why_match computation)
            max_cards: Maximum number of cards to generate

        Returns:
            List of RecipeCard objects
        """
        cards = []

        for result in results[:max_cards]:
            # Fetch full recipe from database
            recipe = get_recipe_by_id(self.db_path, result.recipe_id)
            if recipe is None:
                logger.warning(f"Recipe {result.recipe_id} not found in database")
                continue

            # Build card
            card = self.build_card(recipe, query, result.score)
            cards.append(card)

        logger.info(f"Built {len(cards)} recipe cards")
        return cards

    def build_card(
        self,
        recipe: Recipe,
        query: str,
        score: float
    ) -> RecipeCard:
        """
        Build a single RecipeCard from a Recipe.

        Args:
            recipe: Full Recipe object from database
            query: Original user query
            score: Relevance score from reranking

        Returns:
            Compact RecipeCard for LLM prompt
        """
        return RecipeCard(
            recipe_id=recipe.recipe_id,
            title=recipe.title,
            rating_avg=recipe.rating_avg,
            rating_count=recipe.rating_count,
            tags=recipe.tags[:10] if recipe.tags else [],  # Limit to 10 tags
            time_total=recipe.minutes,
            key_ingredients=self.select_key_ingredients(recipe.ingredients_normalized),
            one_sentence_summary=self.generate_summary(recipe),
            why_match=self.compute_why_match(recipe, query)
        )

    def generate_summary(self, recipe: Recipe) -> str:
        """
        Generate a one-sentence summary of the recipe.

        Uses heuristic template-based approach (not LLM).

        Args:
            recipe: Recipe to summarize

        Returns:
            One-sentence summary (max ~30 words)
        """
        title = recipe.title
        minutes = recipe.minutes
        tags = recipe.tags or []
        ingredients = recipe.ingredients_normalized or []

        # Identify dish type from tags or title
        dish_types = [
            "soup", "salad", "pasta", "casserole", "stir-fry", "stir fry",
            "sandwich", "cake", "bread", "pizza", "curry", "stew", "pie",
            "cookie", "muffin", "taco", "burrito", "burger", "risotto",
            "lasagna", "chili", "quiche", "tart"
        ]
        dish_type = None
        title_lower = title.lower()
        for dt in dish_types:
            if dt in tags or dt in title_lower:
                dish_type = dt
                break
        if not dish_type:
            dish_type = "dish"

        # Identify cuisine
        cuisines = [
            "mexican", "italian", "chinese", "indian", "thai", "greek",
            "japanese", "french", "american", "mediterranean", "korean",
            "spanish", "vietnamese", "middle eastern"
        ]
        cuisine = None
        for c in cuisines:
            if c in tags:
                cuisine = c
                break

        # Identify cooking method
        methods = [
            "baked", "grilled", "fried", "slow-cooked", "roasted",
            "steamed", "broiled", "sauteed"
        ]
        method = None
        for m in methods:
            if any(m in t for t in tags):
                method = m
                break

        # Build top ingredients string (first 3 main ingredients)
        main_ingredients = self.select_key_ingredients(ingredients, max_count=3)
        ingredients_str = ", ".join(main_ingredients) if main_ingredients else ""

        # Build time string
        time_str = ""
        if minutes:
            if minutes <= 20:
                time_str = "quick"
            elif minutes <= 45:
                time_str = f"{minutes}-minute"
            else:
                hours = minutes // 60
                if hours >= 1:
                    time_str = f"{hours}+ hour" if hours == 1 else f"{hours}+ hours"
                else:
                    time_str = f"{minutes}-minute"

        # Assemble summary using best available elements
        parts = []

        # Add time qualifier at start if "quick"
        if time_str == "quick":
            parts.append("Quick")

        if cuisine:
            parts.append(cuisine.capitalize())
        if method:
            parts.append(method)

        parts.append(dish_type)

        if ingredients_str:
            parts.append(f"featuring {ingredients_str}")

        if time_str and time_str != "quick":
            parts.append(f"({time_str})")

        summary = " ".join(parts)

        # Capitalize and add period
        if summary:
            return summary[0].upper() + summary[1:] + "."
        else:
            return f"A {dish_type} recipe."

    def compute_why_match(
        self,
        recipe: Recipe,
        query: str
    ) -> str:
        """
        Compute why this recipe matches the query.

        Args:
            recipe: Recipe to analyze
            query: User's search query

        Returns:
            Brief explanation of match (e.g., "matches chicken, tomato; quick prep")
        """
        # Normalize query terms
        query_terms = set(query.lower().split())

        # Remove common stop words
        stop_words = {
            "a", "an", "the", "with", "and", "or", "for", "to", "in",
            "on", "of", "at", "by", "from", "as", "is", "was", "are",
            "be", "been", "being", "have", "has", "had", "do", "does",
            "did", "will", "would", "should", "could", "may", "might"
        }
        query_terms -= stop_words

        matches = []

        # Check ingredient matches
        ingredients_lower = [i.lower() for i in (recipe.ingredients_normalized or [])]
        matched_ingredients = [
            term for term in query_terms
            if any(term in ing for ing in ingredients_lower)
        ]
        if matched_ingredients:
            matches.append(f"contains {', '.join(matched_ingredients[:3])}")

        # Check tag matches
        tags_lower = [t.lower() for t in (recipe.tags or [])]
        matched_tags = [term for term in query_terms if term in tags_lower]
        if matched_tags:
            matches.append(f"tagged {', '.join(matched_tags[:2])}")

        # Check time-related matches
        time_terms = {"quick", "fast", "easy", "simple"}
        if query_terms & time_terms and recipe.minutes and recipe.minutes <= 30:
            matches.append("quick prep")

        # Check dietary matches
        diet_terms = {"healthy", "vegetarian", "vegan", "low-carb", "keto", "gluten-free"}
        diet_matches = query_terms & diet_terms
        if diet_matches:
            matching_diet_tags = [t for t in diet_matches if t in tags_lower]
            if matching_diet_tags:
                matches.append(f"{matching_diet_tags[0]}")

        # Add rating info if highly rated
        if (recipe.rating_avg and recipe.rating_avg >= 4.5 and
                recipe.rating_count and recipe.rating_count >= 10):
            matches.append(f"highly rated ({recipe.rating_avg:.1f}/5)")

        # Assemble
        if matches:
            return "; ".join(matches)
        else:
            return "matches search query"

    def select_key_ingredients(
        self,
        ingredients: list[str],
        max_count: int = 12
    ) -> list[str]:
        """
        Select the most important ingredients for the card.

        Args:
            ingredients: Full list of normalized ingredients
            max_count: Maximum number to include

        Returns:
            List of key ingredients (prioritizing proteins, vegetables, main items)
        """
        if not ingredients:
            return []

        # Ingredients to deprioritize (common seasonings/basics)
        low_priority = {
            "salt", "pepper", "water", "oil", "olive oil", "vegetable oil",
            "black pepper", "white pepper", "kosher salt", "sea salt",
            "cooking spray", "nonstick cooking spray"
        }

        # Split into high and low priority
        high_priority = []
        low_priority_items = []

        for ing in ingredients:
            if ing.lower() in low_priority:
                low_priority_items.append(ing)
            else:
                high_priority.append(ing)

        # Take high priority first, then low priority if needed
        key_ingredients = high_priority[:max_count]

        # If we have room and low priority items, add some
        remaining = max_count - len(key_ingredients)
        if remaining > 0:
            key_ingredients.extend(low_priority_items[:remaining])

        return key_ingredients

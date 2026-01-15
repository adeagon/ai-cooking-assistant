"""Retrieval chain wrapping vector search, reranking, and card building."""

from typing import Any

from langchain_core.runnables import Runnable

from src.app.logging_config import get_logger
from src.app.settings import Settings
from src.domain.models import Constraints, PreferenceProfile, RecipeCard
from src.retrieval.recipe_cards import RecipeCardBuilder
from src.retrieval.rerank import RecipeReranker
from src.retrieval.retriever import RecipeRetriever

logger = get_logger(__name__)


class RetrievalRunnable(Runnable):
    """LangChain Runnable for recipe retrieval pipeline.

    Pipeline: Vector search (k=100) → Cross-encoder rerank (k=20) → Recipe cards (k=6)
    """

    def __init__(
        self,
        retriever: RecipeRetriever,
        reranker: RecipeReranker,
        card_builder: RecipeCardBuilder,
        settings: Settings,
    ):
        """Initialize retrieval runnable.

        Args:
            retriever: RecipeRetriever instance
            reranker: RecipeReranker instance
            card_builder: RecipeCardBuilder instance
            settings: Settings with retrieval parameters
        """
        super().__init__()
        self.retriever = retriever
        self.reranker = reranker
        self.card_builder = card_builder
        self.settings = settings

    def invoke(self, input_data: dict[str, Any], config: dict | None = None) -> dict[str, Any]:
        """Execute retrieval pipeline.

        Args:
            input_data: Dictionary with "user_input", "constraints", "rolling_summary",
                        "profile", and optional "exclude_recipe_ids" keys
            config: Optional LangChain config

        Returns:
            Input data with "cards" and "cards_text" keys added
        """
        user_input = input_data.get("user_input", "")
        constraints: Constraints = input_data.get("constraints", Constraints())
        exclude_ids: set[str] = input_data.get("exclude_recipe_ids", set())
        profile: PreferenceProfile | None = input_data.get("profile")

        # Build query from user input, constraints, and profile
        query = self._build_query(user_input, constraints, profile)

        logger.info(
            "Starting retrieval pipeline",
            query=query,
            k_retrieve=self.settings.k_retrieve,
            dietary=constraints.dietary,
            cuisine=constraints.cuisine,
            time_limit=constraints.time_limit,
        )

        # Step 1: Vector search with constraint filters
        results = self.retriever.search_with_constraints(
            query=query,
            k=self.settings.k_retrieve,
            max_minutes=constraints.time_limit,
            dietary=constraints.dietary,
            cuisine=constraints.cuisine,
        )

        logger.info(f"Retrieved {len(results)} candidates from vector search")

        # Filter out excluded recipes (liked, disliked, recently cooked)
        if exclude_ids:
            before_count = len(results)
            results = [r for r in results if r.recipe_id not in exclude_ids]
            logger.info(f"Filtered {before_count - len(results)} excluded recipes, {len(results)} remaining")

        # Filter out recipes matching avoid constraints (e.g., "no casseroles")
        if constraints.avoid:
            before_count = len(results)
            results = self._filter_avoid_constraints(results, constraints.avoid)
            logger.info(
                f"Filtered {before_count - len(results)} avoided recipes, {len(results)} remaining",
                avoid=constraints.avoid
            )

        # Step 2: Rerank with cross-encoder
        reranked = self.reranker.rerank(query, results, top_k=self.settings.k_rerank)

        logger.info(f"Reranked to top {len(reranked)} candidates")

        # Step 3: Build recipe cards
        cards = self.card_builder.build_cards(reranked[: self.settings.k_context], query)

        logger.info(f"Built {len(cards)} recipe cards for LLM context")

        # Format cards as text
        cards_text = self._format_cards(cards)

        return {**input_data, "cards": cards, "cards_text": cards_text}

    def _build_query(
        self,
        user_input: str,
        constraints: Constraints,
        profile: PreferenceProfile | None = None,
    ) -> str:
        """Build search query from input, constraints, and profile.

        Args:
            user_input: Original user input
            constraints: Extracted constraints
            profile: Optional user profile for preference boosting

        Returns:
            Enhanced query string
        """
        query_parts = [user_input]

        # NOTE: rolling_summary is intentionally NOT added to the query.
        # It was causing context pollution where previous session context
        # (e.g., "cuisine: indian") would influence unrelated searches.
        # The session context is passed to the LLM via session_context in the prompt.

        # Add ingredients to query
        if constraints.ingredients:
            query_parts.append(" ".join(constraints.ingredients))

        # Add dish name (important for specific dish requests like "tikka masala")
        if constraints.dish_name:
            query_parts.append(constraints.dish_name)

        # Add cuisine to query (from constraints OR profile preference)
        if constraints.cuisine:
            query_parts.append(constraints.cuisine)
        elif profile and profile.preferred_cuisines:
            # Use profile preference when no explicit cuisine mentioned
            preferred = profile.preferred_cuisines[0]
            query_parts.append(preferred)
            logger.debug("Added profile cuisine preference to query", cuisine=preferred)

        # Add dietary restrictions (from constraints OR profile)
        if constraints.dietary:
            query_parts.append(constraints.dietary)
        elif profile and profile.diet and profile.diet != "none":
            query_parts.append(profile.diet)

        # Add goals
        if constraints.goals:
            query_parts.extend(constraints.goals)

        return " ".join(query_parts)

    def _filter_avoid_constraints(
        self,
        results: list,
        avoid: list[str],
    ) -> list:
        """Filter out recipes that match avoid constraints.

        Args:
            results: List of RetrievalResult objects
            avoid: List of terms to avoid (e.g., ["casseroles", "soups"])

        Returns:
            Filtered list of results
        """
        if not avoid:
            return results

        filtered = []
        for result in results:
            title_lower = result.title.lower()
            # Check if any avoid term appears in the title
            should_avoid = False
            for term in avoid:
                term_lower = term.lower()
                # Check for exact word match or partial match
                if term_lower in title_lower:
                    should_avoid = True
                    break
                # Also check singular/plural variations
                if term_lower.endswith("s"):
                    singular = term_lower[:-1]
                    if singular in title_lower:
                        should_avoid = True
                        break
                else:
                    plural = term_lower + "s"
                    if plural in title_lower:
                        should_avoid = True
                        break

            if not should_avoid:
                filtered.append(result)

        return filtered

    def _format_cards(self, cards: list[RecipeCard]) -> str:
        """Format recipe cards for LLM prompt.

        Args:
            cards: List of RecipeCard objects

        Returns:
            Formatted text block
        """
        if not cards:
            return "No recipes found matching your criteria."

        lines = []
        for i, card in enumerate(cards, 1):
            lines.append(f"### {i}. {card.title}")

            # Rating
            if card.rating_avg and card.rating_count:
                lines.append(f"- Rating: {card.rating_avg:.1f}/5 ({card.rating_count} reviews)")

            # Time
            if card.time_total:
                lines.append(f"- Time: {card.time_total} minutes")

            # Tags
            if card.tags:
                tags_str = ", ".join(card.tags[:8])  # Limit to 8 tags
                lines.append(f"- Tags: {tags_str}")

            # Key ingredients
            if card.key_ingredients:
                ingredients_str = ", ".join(card.key_ingredients[:10])
                lines.append(f"- Key ingredients: {ingredients_str}")

            # Summary
            if card.one_sentence_summary:
                lines.append(f"- Summary: {card.one_sentence_summary}")

            # Why it matches
            if card.why_match:
                lines.append(f"- Why it matches: {card.why_match}")

            lines.append("")  # Empty line between cards

        return "\n".join(lines)

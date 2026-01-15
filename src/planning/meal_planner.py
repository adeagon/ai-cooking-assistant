"""Deterministic meal planner with beam search for ingredient overlap optimization."""

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, timedelta

from src.app.logging_config import get_logger
from src.domain.models import (
    IngredientCategory,
    MealPlanConstraints,
    PlannedMeal,
    PlanMetrics,
    PreferenceProfile,
    Recipe,
)
from src.planning.ingredient_categories import IngredientCategoryClassifier
from src.planning.ingredient_normalizer import IngredientNormalizer

logger = get_logger(__name__)


@dataclass
class RecipeFeatures:
    """Cached per-recipe features to avoid recomputation."""

    recipe_id: str
    title: str
    source: str  # "box" or "discovery"
    protein: str | None = None
    cuisine: str | None = None
    minutes: int | None = None
    rating_avg: float | None = None
    normalized_ingredients: set[str] = field(default_factory=set)  # Tokenized, stop-filtered
    base_score: float = 0.0  # Rating + box bonus for prefiltering


class MealPlanner:
    """Deterministic meal planner with overlap optimization.

    Uses beam search to select recipes that maximize ingredient overlap
    while respecting diversity constraints (max same protein, max same cuisine).
    """

    MAX_CANDIDATES = 300  # Prefilter to this many before beam search
    DEFAULT_BEAM_WIDTH = 5

    def __init__(self) -> None:
        """Initialize planner with normalizer and classifier."""
        self.normalizer = IngredientNormalizer()
        self.category_classifier = IngredientCategoryClassifier()

    def generate_plan(
        self,
        recipes: list[Recipe],
        constraints: MealPlanConstraints,
        profile: PreferenceProfile | None = None,
        box_recipe_ids: set[str] | None = None,
    ) -> tuple[list[PlannedMeal], PlanMetrics]:
        """Generate optimized meal plan using beam search.

        Args:
            recipes: Available recipes to choose from
            constraints: Planning constraints
            profile: User preferences (optional)
            box_recipe_ids: Recipe IDs in user's Recipe Box (get priority)

        Returns:
            Tuple of (list of PlannedMeal, PlanMetrics)
        """
        num_meals = constraints.days * len(constraints.meal_types)

        if num_meals == 0:
            logger.warning("No meals to plan", days=constraints.days, meal_types=constraints.meal_types)
            return [], self._empty_metrics()

        # Build and prefilter candidate pool
        candidates = self._filter_candidates(recipes, constraints)
        if not candidates:
            logger.warning("No candidates after filtering")
            return [], self._empty_metrics()

        # Compute features for each candidate
        candidate_features = self._compute_features(
            candidates, constraints, box_recipe_ids or set()
        )

        # DETERMINISTIC: Stable sort by (base_score DESC, recipe_id ASC)
        candidate_features.sort(key=lambda f: (-f.base_score, f.recipe_id))
        candidate_features = candidate_features[: self.MAX_CANDIDATES]

        if len(candidate_features) < num_meals:
            logger.warning(
                "Fewer candidates than meals requested",
                candidates=len(candidate_features),
                meals_needed=num_meals,
            )
            num_meals = len(candidate_features)

        # Beam search with width 5
        selected = self._beam_search_select(
            candidate_features, num_meals, constraints, beam_width=self.DEFAULT_BEAM_WIDTH
        )

        # Local optimization (try swaps)
        selected = self._local_optimize(selected, candidate_features, constraints)

        # Assign to days and compute metrics
        meals = self._assign_to_days(selected, constraints)
        metrics = self._compute_metrics(selected, num_meals)

        logger.info(
            "Generated meal plan",
            num_meals=len(meals),
            overlap_ratio=metrics.overlap_ratio,
            unique_per_meal=metrics.unique_per_meal,
        )

        return meals, metrics

    def _filter_candidates(
        self,
        recipes: list[Recipe],
        constraints: MealPlanConstraints,
    ) -> list[Recipe]:
        """Filter recipes based on constraints.

        Args:
            recipes: All available recipes
            constraints: Constraints to apply

        Returns:
            Filtered list of recipes
        """
        filtered = []

        for recipe in recipes:
            # Time constraint
            if constraints.max_prep_time and recipe.minutes:
                if recipe.minutes > constraints.max_prep_time:
                    continue

            # Dietary constraints (check tags for vegetarian/vegan)
            if constraints.dietary.value != "none":
                tags_lower = {t.lower() for t in recipe.tags}
                if constraints.dietary.value == "vegetarian":
                    if "vegetarian" not in tags_lower and "vegan" not in tags_lower:
                        continue
                elif constraints.dietary.value == "vegan":
                    if "vegan" not in tags_lower:
                        continue
                # Other dietary restrictions would need tag-based filtering

            # Excluded tags
            if constraints.excluded_tags:
                tags_lower = {t.lower() for t in recipe.tags}
                if any(excl.lower() in tags_lower for excl in constraints.excluded_tags):
                    continue

            # Excluded ingredients
            if constraints.excluded_ingredients:
                recipe_ingredients = " ".join(
                    self.normalizer.normalize(ing) for ing in recipe.ingredients_normalized
                )
                if any(
                    excl.lower() in recipe_ingredients.lower()
                    for excl in constraints.excluded_ingredients
                ):
                    continue

            # Excluded categories
            if constraints.excluded_categories:
                has_excluded = False
                for ing in recipe.ingredients_normalized:
                    normalized = self.normalizer.normalize(ing)
                    if self.category_classifier.contains_excluded_category(
                        normalized, constraints.excluded_categories
                    ):
                        has_excluded = True
                        break
                if has_excluded:
                    continue

            filtered.append(recipe)

        logger.debug(
            "Filtered candidates",
            original=len(recipes),
            filtered=len(filtered),
        )

        return filtered

    def _compute_features(
        self,
        recipes: list[Recipe],
        constraints: MealPlanConstraints,
        box_recipe_ids: set[str],
    ) -> list[RecipeFeatures]:
        """Compute features for each recipe candidate.

        Args:
            recipes: Recipes to compute features for
            constraints: Constraints (for weighting)
            box_recipe_ids: Recipe IDs in user's Recipe Box

        Returns:
            List of RecipeFeatures
        """
        features = []

        for recipe in recipes:
            # Compute normalized ingredients
            normalized_tokens: set[str] = set()
            for ing in recipe.ingredients_normalized:
                normalized = self.normalizer.normalize(ing)
                tokens = self.normalizer.tokenize(normalized)
                normalized_tokens.update(tokens)

            # Extract protein from tags or ingredients
            protein = self._extract_protein(recipe)

            # Extract cuisine from tags
            cuisine = self._extract_cuisine(recipe)

            # Compute base score
            is_box = recipe.recipe_id in box_recipe_ids
            box_bonus = 2.0 if is_box and constraints.prefer_recipe_box else 0.0
            rating_score = recipe.rating_avg or 0.0
            base_score = rating_score + box_bonus

            features.append(
                RecipeFeatures(
                    recipe_id=recipe.recipe_id,
                    title=recipe.title,
                    source="box" if is_box else "discovery",
                    protein=protein,
                    cuisine=cuisine,
                    minutes=recipe.minutes,
                    rating_avg=recipe.rating_avg,
                    normalized_ingredients=normalized_tokens,
                    base_score=base_score,
                )
            )

        return features

    def _extract_protein(self, recipe: Recipe) -> str | None:
        """Extract primary protein from recipe tags or ingredients."""
        protein_keywords = {
            "chicken": "chicken",
            "beef": "beef",
            "pork": "pork",
            "turkey": "turkey",
            "lamb": "lamb",
            "fish": "fish",
            "salmon": "fish",
            "shrimp": "seafood",
            "seafood": "seafood",
            "tofu": "tofu",
            "vegetarian": None,  # No protein
            "vegan": None,
        }

        # Check tags first
        for tag in recipe.tags:
            tag_lower = tag.lower()
            if tag_lower in protein_keywords:
                return protein_keywords[tag_lower]

        # Check ingredients
        for ing in recipe.ingredients_normalized:
            ing_lower = ing.lower()
            for keyword, protein in protein_keywords.items():
                if keyword in ing_lower and protein:
                    return protein

        return None

    def _extract_cuisine(self, recipe: Recipe) -> str | None:
        """Extract cuisine from recipe tags."""
        cuisine_keywords = {
            "italian",
            "mexican",
            "asian",
            "chinese",
            "japanese",
            "thai",
            "indian",
            "greek",
            "mediterranean",
            "french",
            "american",
            "southern",
            "korean",
            "vietnamese",
            "middle-eastern",
            "spanish",
        }

        for tag in recipe.tags:
            tag_lower = tag.lower()
            if tag_lower in cuisine_keywords:
                return tag_lower

        return None

    def _beam_search_select(
        self,
        candidates: list[RecipeFeatures],
        num_meals: int,
        constraints: MealPlanConstraints,
        beam_width: int = 5,
    ) -> list[RecipeFeatures]:
        """Beam search selection with diversity constraints.

        State: (selected_list, selected_ids, ingredient_counter, cuisine_counts, protein_counts, score)

        IMPORTANT:
        - selected_list is a tuple to preserve selection order
        - Tie-breaking uses selected_list for determinism

        Args:
            candidates: Pre-filtered and sorted recipe features
            num_meals: Number of meals to select
            constraints: Constraints for diversity limits
            beam_width: Width of beam search

        Returns:
            List of selected RecipeFeatures in order
        """
        # Initial state
        initial_state = (
            (),  # selected_list: tuple[str, ...] (ORDERED)
            frozenset(),  # selected_ids: frozenset[str] (for fast membership)
            Counter(),  # ingredient_counter
            Counter(),  # cuisine_counts
            Counter(),  # protein_counts
            0.0,  # score
        )
        beam = [initial_state]

        candidate_map = {f.recipe_id: f for f in candidates}

        for slot in range(num_meals):
            next_beam: list[tuple] = []

            for state in beam:
                (
                    selected_list,
                    selected_ids,
                    ing_counter,
                    cuisine_counts,
                    protein_counts,
                    score,
                ) = state

                for features in candidates:
                    if features.recipe_id in selected_ids:
                        continue

                    # Check diversity constraints
                    if (
                        features.protein
                        and protein_counts[features.protein] >= constraints.max_same_protein
                    ):
                        continue
                    if (
                        features.cuisine
                        and cuisine_counts[features.cuisine] >= constraints.max_same_cuisine
                    ):
                        continue

                    # Score this addition
                    add_score = self._score_candidate(features, ing_counter, constraints)

                    # Build new state (immutable updates, preserving order)
                    new_selected_list = selected_list + (features.recipe_id,)
                    new_selected_ids = selected_ids | {features.recipe_id}

                    new_ing_counter = Counter(ing_counter)
                    for token in features.normalized_ingredients:
                        new_ing_counter[token] += 1

                    new_cuisine_counts = Counter(cuisine_counts)
                    if features.cuisine:
                        new_cuisine_counts[features.cuisine] += 1

                    new_protein_counts = Counter(protein_counts)
                    if features.protein:
                        new_protein_counts[features.protein] += 1

                    next_beam.append(
                        (
                            new_selected_list,
                            new_selected_ids,
                            new_ing_counter,
                            new_cuisine_counts,
                            new_protein_counts,
                            score + add_score,
                        )
                    )

            if not next_beam:
                logger.warning(
                    "Beam search exhausted candidates early",
                    slot=slot,
                    num_meals=num_meals,
                )
                break

            # DETERMINISTIC: Sort by (score DESC, selected_list ASC for tie-breaking)
            next_beam.sort(key=lambda x: (-x[5], x[0]))
            beam = next_beam[:beam_width]

        # Return best plan's recipe features IN ORDER
        if beam:
            best_list = beam[0][0]
            return [candidate_map[rid] for rid in best_list if rid in candidate_map]

        return []

    def _score_candidate(
        self,
        features: RecipeFeatures,
        current_ingredients: Counter,
        constraints: MealPlanConstraints,
    ) -> float:
        """Score a candidate recipe for overlap and quality.

        Args:
            features: Recipe features
            current_ingredients: Current ingredient counter from selected recipes
            constraints: Constraints (for overlap weight)

        Returns:
            Score for this candidate
        """
        # Base quality score
        quality_score = features.base_score

        # Overlap score: count how many of this recipe's ingredients are already used
        overlap_count = sum(
            1 for token in features.normalized_ingredients if current_ingredients[token] > 0
        )

        # Normalize overlap score by total ingredients
        if features.normalized_ingredients:
            overlap_ratio = overlap_count / len(features.normalized_ingredients)
        else:
            overlap_ratio = 0.0

        # Weighted combination
        overlap_weight = constraints.ingredient_overlap_weight
        total_score = (1 - overlap_weight) * quality_score + overlap_weight * overlap_ratio * 10

        return total_score

    def _local_optimize(
        self,
        selected: list[RecipeFeatures],
        all_candidates: list[RecipeFeatures],
        constraints: MealPlanConstraints,
    ) -> list[RecipeFeatures]:
        """Try local swaps to improve the plan.

        Simple hill-climbing: for each position, try swapping with unused candidates.

        Args:
            selected: Currently selected recipes
            all_candidates: All available candidates
            constraints: Constraints for validation

        Returns:
            Optimized list of RecipeFeatures
        """
        if len(selected) <= 1:
            return selected

        current_score = self._compute_plan_score(selected, constraints)
        improved = list(selected)
        selected_ids = {f.recipe_id for f in selected}

        for position in range(len(improved)):
            best_swap = None
            best_delta = 0.0

            for candidate in all_candidates:
                if candidate.recipe_id in selected_ids:
                    continue

                # Check constraints for swap
                if not self._is_valid_swap(improved, position, candidate, constraints):
                    continue

                # Compute score delta
                trial = improved[:position] + [candidate] + improved[position + 1 :]
                trial_score = self._compute_plan_score(trial, constraints)
                delta = trial_score - current_score

                if delta > best_delta:
                    best_delta = delta
                    best_swap = candidate

            if best_swap is not None:
                # Apply swap
                selected_ids.remove(improved[position].recipe_id)
                selected_ids.add(best_swap.recipe_id)
                improved[position] = best_swap
                current_score += best_delta

        return improved

    def _is_valid_swap(
        self,
        current: list[RecipeFeatures],
        position: int,
        candidate: RecipeFeatures,
        constraints: MealPlanConstraints,
    ) -> bool:
        """Check if swapping at position with candidate is valid.

        Args:
            current: Current selection
            position: Position to swap
            candidate: Candidate to swap in
            constraints: Constraints to check

        Returns:
            True if swap is valid
        """
        # Build counts excluding the position being swapped
        remaining = current[:position] + current[position + 1 :]
        protein_counts = Counter(f.protein for f in remaining if f.protein)
        cuisine_counts = Counter(f.cuisine for f in remaining if f.cuisine)

        # Check if candidate would violate constraints
        if (
            candidate.protein
            and protein_counts[candidate.protein] >= constraints.max_same_protein
        ):
            return False
        if (
            candidate.cuisine
            and cuisine_counts[candidate.cuisine] >= constraints.max_same_cuisine
        ):
            return False

        return True

    def _compute_plan_score(
        self,
        selected: list[RecipeFeatures],
        constraints: MealPlanConstraints,
    ) -> float:
        """Compute total score for a plan.

        Args:
            selected: Selected recipes
            constraints: Constraints (for overlap weight)

        Returns:
            Total plan score
        """
        if not selected:
            return 0.0

        # Quality score
        quality = sum(f.base_score for f in selected)

        # Overlap score
        all_tokens: Counter[str] = Counter()
        for f in selected:
            for token in f.normalized_ingredients:
                all_tokens[token] += 1

        # Count tokens that appear more than once
        overlap_tokens = sum(1 for count in all_tokens.values() if count > 1)
        total_unique = len(all_tokens)
        overlap_ratio = overlap_tokens / total_unique if total_unique > 0 else 0.0

        overlap_weight = constraints.ingredient_overlap_weight
        return (1 - overlap_weight) * quality + overlap_weight * overlap_ratio * 10 * len(selected)

    def _assign_to_days(
        self,
        selected: list[RecipeFeatures],
        constraints: MealPlanConstraints,
    ) -> list[PlannedMeal]:
        """Assign selected recipes to days and meal types.

        Args:
            selected: Selected recipe features
            constraints: Constraints (for start_date and meal_types)

        Returns:
            List of PlannedMeal objects
        """
        start = constraints.start_date or date.today()
        meal_types = constraints.meal_types or ["dinner"]
        meals: list[PlannedMeal] = []

        recipe_idx = 0
        for day_offset in range(constraints.days):
            current_day = start + timedelta(days=day_offset)

            for position, meal_type in enumerate(meal_types):
                if recipe_idx >= len(selected):
                    break

                features = selected[recipe_idx]
                meals.append(
                    PlannedMeal(
                        day=current_day,
                        meal_type=meal_type,
                        recipe_id=features.recipe_id,
                        title=features.title,
                        position=position,
                        source=features.source,
                    )
                )
                recipe_idx += 1

        return meals

    def _compute_metrics(
        self,
        selected: list[RecipeFeatures],
        num_meals: int,
    ) -> PlanMetrics:
        """Compute plan scoring metrics.

        Args:
            selected: Selected recipe features
            num_meals: Total number of meals planned

        Returns:
            PlanMetrics object
        """
        if not selected:
            return self._empty_metrics()

        all_tokens: Counter[str] = Counter()
        for features in selected:
            for token in features.normalized_ingredients:
                all_tokens[token] += 1

        unique = len(all_tokens)
        total = sum(all_tokens.values())

        protein_dist = Counter(f.protein for f in selected if f.protein)
        cuisine_dist = Counter(f.cuisine for f in selected if f.cuisine)

        return PlanMetrics(
            unique_ingredients=unique,
            total_ingredient_uses=total,
            overlap_ratio=1 - (unique / total) if total > 0 else 0,
            unique_per_meal=unique / num_meals if num_meals > 0 else 0,
            top_shared_ingredients=all_tokens.most_common(10),
            protein_distribution=dict(protein_dist),
            cuisine_distribution=dict(cuisine_dist),
            box_recipe_count=sum(1 for f in selected if f.source == "box"),
            discovery_recipe_count=sum(1 for f in selected if f.source == "discovery"),
        )

    def _empty_metrics(self) -> PlanMetrics:
        """Return empty metrics object."""
        return PlanMetrics(
            unique_ingredients=0,
            total_ingredient_uses=0,
            overlap_ratio=0,
            unique_per_meal=0,
            top_shared_ingredients=[],
            protein_distribution={},
            cuisine_distribution={},
            box_recipe_count=0,
            discovery_recipe_count=0,
        )

    def suggest_swaps(
        self,
        plan_features: list[RecipeFeatures],
        all_candidates: list[RecipeFeatures],
        constraints: MealPlanConstraints,
        position: int,
        k: int = 5,
    ) -> list[tuple[RecipeFeatures, float]]:
        """Suggest k best swap candidates for a position with score deltas.

        IMPORTANT: Filters candidates by constraints BEFORE scoring.

        Args:
            plan_features: Current plan's recipe features
            all_candidates: All available candidates
            constraints: Constraints (for validation)
            position: Position in plan to swap
            k: Number of suggestions to return

        Returns:
            List of (candidate_features, score_delta) tuples
        """
        if position < 0 or position >= len(plan_features):
            return []

        current_score = self._compute_plan_score(plan_features, constraints)
        selected_ids = {f.recipe_id for f in plan_features}

        swaps: list[tuple[RecipeFeatures, float]] = []

        for candidate in all_candidates:
            # Skip if already in plan
            if candidate.recipe_id in selected_ids:
                continue

            # Check constraints
            if not self._is_valid_swap(plan_features, position, candidate, constraints):
                continue

            # Score the swap
            trial = plan_features[:position] + [candidate] + plan_features[position + 1 :]
            trial_score = self._compute_plan_score(trial, constraints)
            delta = trial_score - current_score

            swaps.append((candidate, delta))

        # Return top k by score delta (deterministic: tie-break by recipe_id)
        swaps.sort(key=lambda x: (-x[1], x[0].recipe_id))
        return swaps[:k]

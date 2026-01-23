"""Domain models for recipes, preferences, and session state."""

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field


# ============================================================================
# Meal Planning Enums
# ============================================================================


class DietaryRestriction(str, Enum):
    """Dietary restrictions for meal planning."""

    NONE = "none"
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    PESCATARIAN = "pescatarian"
    KETO = "keto"
    GLUTEN_FREE = "gluten_free"
    DAIRY_FREE = "dairy_free"


class IngredientCategory(str, Enum):
    """Categories for ingredient-level exclusions."""

    DAIRY = "dairy"
    MEAT = "meat"
    POULTRY = "poultry"
    SEAFOOD = "seafood"
    GLUTEN = "gluten"
    NUTS = "nuts"
    SOY = "soy"
    EGGS = "eggs"


class ExtractionSource(str, Enum):
    """Source of extracted constraint value."""

    RULE = "rule"
    LLM = "llm"
    DEFAULT = "default"
    USER_PROFILE = "user_profile"


# ============================================================================
# Meal Planning Models
# ============================================================================


class ExtractedValue(BaseModel):
    """Wrapper for extracted constraint with audit trail."""

    value: Any
    source: ExtractionSource
    confidence: float = 1.0  # 0-1, lower for LLM-extracted


class PlannedMeal(BaseModel):
    """A single meal in a meal plan."""

    id: int | None = None
    plan_id: int | None = None
    day: date
    meal_type: Literal["breakfast", "lunch", "dinner"]
    recipe_id: str
    title: str
    position: int = 0  # Order within day
    source: Literal["box", "discovery"] = "discovery"
    # NOTE: locked feature deferred to v2 (requires swap UX)


class PlanMetrics(BaseModel):
    """Scoring metrics for a meal plan - testable and displayable."""

    unique_ingredients: int
    total_ingredient_uses: int
    overlap_ratio: float  # 1 - (unique / total)
    unique_per_meal: float  # unique / num_meals (more intuitive)
    top_shared_ingredients: list[tuple[str, int]] = Field(default_factory=list)
    protein_distribution: dict[str, int] = Field(default_factory=dict)
    cuisine_distribution: dict[str, int] = Field(default_factory=dict)
    box_recipe_count: int = 0
    discovery_recipe_count: int = 0


class MealPlanConstraints(BaseModel):
    """Constraints for meal plan generation."""

    days: int = 5  # Default Mon-Fri
    start_date: date | None = None
    meal_types: list[Literal["breakfast", "lunch", "dinner"]] = Field(
        default_factory=lambda: ["dinner"]
    )
    dietary: DietaryRestriction = DietaryRestriction.NONE
    max_prep_time: int | None = None
    servings: int | None = None
    prefer_recipe_box: bool = True
    ingredient_overlap_weight: float = 0.3
    # Exclusions (ingredient-level)
    excluded_ingredients: list[str] = Field(default_factory=list)
    excluded_categories: list[IngredientCategory] = Field(default_factory=list)
    excluded_tags: list[str] = Field(default_factory=list)  # e.g., "casserole"
    # Diversity constraints
    max_same_protein: int = 2  # Don't pick 3 chicken dinners
    max_same_cuisine: int = 2
    # Extraction audit trail
    extraction_sources: dict[str, ExtractedValue] = Field(default_factory=dict)


class MealPlan(BaseModel):
    """A complete meal plan for a date range."""

    id: int | None = None
    user_id: str | None = None  # For future multi-user
    name: str | None = None
    start_date: date
    end_date: date
    meal_types: list[Literal["breakfast", "lunch", "dinner"]] = Field(
        default_factory=lambda: ["dinner"]
    )
    status: Literal["draft", "active", "completed", "archived"] = "draft"
    schema_version: int = 1  # For future migrations
    constraints: dict | None = None  # Store constraints dict for regeneration/debug
    metrics: PlanMetrics | None = None  # Computed scoring metrics
    created_at: datetime | None = None
    updated_at: datetime | None = None
    meals: list[PlannedMeal] = Field(default_factory=list)


class GroceryItem(BaseModel):
    """A single item on the grocery list."""

    ingredient: str  # Original text for display
    normalized: str  # Normalized form for grouping
    recipes: list[str] = Field(default_factory=list)  # Recipe titles using this
    category: str | None = None  # produce, protein, dairy, etc.


class GroceryList(BaseModel):
    """Aggregated grocery list from a meal plan."""

    plan_id: int
    items: list[GroceryItem] = Field(default_factory=list)
    generated_at: datetime | None = None


# ============================================================================
# Recipe Models
# ============================================================================


class NormalizedIngredient(BaseModel):
    """Normalized ingredient with parsed components."""

    name: str  # Normalized ingredient name
    quantity: str | None = None  # e.g., "2", "1/2"
    unit: str | None = None  # e.g., "cup", "tablespoon"
    raw: str  # Original raw ingredient string


class RatingStats(BaseModel):
    """Aggregated rating statistics for a recipe."""

    rating_avg: float
    rating_count: int


class Recipe(BaseModel):
    """Canonical recipe model."""

    recipe_id: str
    title: str
    ingredients: list[str] = Field(default_factory=list)  # Raw ingredients
    ingredients_normalized: list[str] = Field(default_factory=list)  # Normalized names only
    instructions: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)  # Simplified to list of tags
    rating_avg: float | None = None
    rating_count: int | None = None
    minutes: int | None = None  # Cooking time in minutes
    n_steps: int | None = None  # Number of steps
    n_ingredients: int | None = None  # Number of ingredients
    source: str = "foodcom"


class RecipeCard(BaseModel):
    """Compact recipe card for LLM prompts (120-250 tokens)."""

    recipe_id: str
    title: str
    rating_avg: float | None = None
    rating_count: int | None = None
    tags: list[str] = Field(default_factory=list)
    time_total: int | None = None  # minutes
    key_ingredients: list[str] = Field(default_factory=list)  # 8-15 ingredients
    one_sentence_summary: str = ""
    why_match: str = ""  # computed at query time


class PreferenceProfile(BaseModel):
    """User's long-term preferences."""

    spice_level: Literal["none", "mild", "medium", "hot"] = "medium"
    diet: Literal[
        "none",
        "vegetarian",
        "vegan",
        "pescatarian",
        "keto",
        "gluten_free"
    ] = "none"
    avoid_ingredients: list[str] = Field(default_factory=list)
    preferred_cuisines: list[str] = Field(default_factory=list)
    time_limit_default_minutes: int | None = None


class SessionState(BaseModel):
    """Session-specific constraints (tonight's dinner)."""

    ingredients_on_hand: list[str] = Field(default_factory=list)
    avoid_tonight: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)  # e.g., ["healthy", "quick"]
    time_limit_minutes: int | None = None
    servings: int | None = None


class RecipeFeedback(BaseModel):
    """User feedback on a recipe (like/dislike/rating)."""

    id: int | None = None
    recipe_id: str
    feedback_type: Literal["like", "dislike", "rate"]
    rating: int | None = None  # 1-5 for ratings, NULL for like/dislike
    session_id: str | None = None
    created_at: datetime | None = None


class CookingHistoryEntry(BaseModel):
    """Record of a cooked recipe."""

    id: int | None = None
    recipe_id: str
    cooked_at: datetime | None = None
    notes: str | None = None


class SavedRecipe(BaseModel):
    """Saved recipe in Recipe Box."""

    id: int | None = None
    recipe_id: str
    title: str  # Store title for display without DB join
    saved_at: datetime | None = None
    notes: str | None = None


class Constraints(BaseModel):
    """Extracted constraints from user input."""

    ingredients: list[str] = Field(default_factory=list)
    avoid: list[str] = Field(default_factory=list)
    time_limit: int | None = None
    dietary: str | None = None
    cuisine: str | None = None
    goals: list[str] = Field(default_factory=list)
    dish_name: str | None = None  # Specific dish being requested (e.g., "tikka masala")


class RetrievalResult(BaseModel):
    """Result from vector search retrieval."""

    recipe_id: str
    title: str
    score: float  # similarity score (0-1, higher is better)
    rating_avg: float | None = None
    rating_count: int | None = None
    minutes: int | None = None


class IntentClassification(BaseModel):
    """Result of intent classification from user input."""

    intent: Literal[
        "save", "like", "dislike", "rate", "show", "cooked",
        "history", "box", "unsave", "new", "prefs", "commands",
        "addpref", "filter_previous", "mealplan", "show_plan",
        "grocery_list", "login", "logout", "whoami", "conversation"
    ]
    confidence: Literal["high", "medium", "low"]
    recipe_reference: str | None = None  # e.g., "first one", "2", "the pasta", "it"
    rating_value: int | None = None  # 1-5 for rate intent
    source: Literal["box", "recommendations"] | None = None  # Where to look for recipe
    filter_type: str | None = None  # For filter_previous: "best_rated", "quickest", "most_reviewed"
    reasoning: str = ""  # Brief explanation of why this intent was chosen

"""Domain models for recipes, preferences, and session state."""

from datetime import datetime
from typing import Literal
from pydantic import BaseModel, Field


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
        "addpref", "filter_previous", "conversation"
    ]
    confidence: Literal["high", "medium", "low"]
    recipe_reference: str | None = None  # e.g., "first one", "2", "the pasta", "it"
    rating_value: int | None = None  # 1-5 for rate intent
    source: Literal["box", "recommendations"] | None = None  # Where to look for recipe
    filter_type: str | None = None  # For filter_previous: "best_rated", "quickest", "most_reviewed"
    reasoning: str = ""  # Brief explanation of why this intent was chosen

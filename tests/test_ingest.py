"""Tests for data ingestion modules."""

import pytest
from pathlib import Path
from src.ingest.normalize import normalize_ingredient, extract_key_ingredients
from src.ingest.filters import apply_quality_filters
from src.domain.models import RatingStats, NormalizedIngredient


def test_normalize_ingredient():
    """Test ingredient normalization."""
    # Test basic ingredient
    result = normalize_ingredient("2 cups flour")
    assert result.name == "flour" or "flour" in result.name
    assert result.raw == "2 cups flour"

    # Test ingredient without quantity
    result = normalize_ingredient("salt")
    assert result.name == "salt"
    assert result.raw == "salt"


def test_extract_key_ingredients():
    """Test key ingredient extraction."""
    ingredients = [
        "2 cups all-purpose flour",
        "1 cup sugar",
        "1/2 teaspoon salt",
        "1 cup all-purpose flour",  # Duplicate
    ]

    normalized = extract_key_ingredients(ingredients, max_count=10)

    # Should have deduplicated ingredients
    assert len(normalized) <= len(ingredients)
    assert isinstance(normalized, list)
    assert all(isinstance(name, str) for name in normalized)


def test_apply_quality_filters_pass():
    """Test that good recipes pass filters."""
    recipe = {
        'name': 'Chocolate Cake',
        'ingredients': ['flour', 'sugar', 'cocoa'],
        'steps': ['Mix', 'Bake'],
    }
    rating_stats = RatingStats(rating_avg=4.5, rating_count=10)

    assert apply_quality_filters(recipe, rating_stats, min_rating_count=3, min_rating_avg=3.5, max_minutes=1440) is True


def test_apply_quality_filters_fail_no_ratings():
    """Test that recipes without ratings are filtered out."""
    recipe = {
        'name': 'Chocolate Cake',
        'ingredients': ['flour', 'sugar', 'cocoa'],
        'steps': ['Mix', 'Bake'],
    }

    assert apply_quality_filters(recipe, None) is False


def test_apply_quality_filters_fail_low_rating():
    """Test that low-rated recipes are filtered out."""
    recipe = {
        'name': 'Bad Recipe',
        'ingredients': ['flour'],
        'steps': ['Mix'],
    }
    rating_stats = RatingStats(rating_avg=2.0, rating_count=5)

    assert apply_quality_filters(recipe, rating_stats, min_rating_avg=3.5) is False


def test_apply_quality_filters_fail_few_ratings():
    """Test that recipes with few ratings are filtered out."""
    recipe = {
        'name': 'Untested Recipe',
        'ingredients': ['flour'],
        'steps': ['Mix'],
    }
    rating_stats = RatingStats(rating_avg=4.5, rating_count=2)

    assert apply_quality_filters(recipe, rating_stats, min_rating_count=3) is False


def test_apply_quality_filters_fail_missing_fields():
    """Test that recipes with missing fields are filtered out."""
    rating_stats = RatingStats(rating_avg=4.5, rating_count=10)

    # No name
    assert apply_quality_filters({'ingredients': ['flour'], 'steps': ['Mix']}, rating_stats) is False

    # No ingredients
    assert apply_quality_filters({'name': 'Test', 'steps': ['Mix']}, rating_stats) is False

    # No steps
    assert apply_quality_filters({'name': 'Test', 'ingredients': ['flour']}, rating_stats) is False


def test_normalized_ingredient_model():
    """Test NormalizedIngredient model."""
    ing = NormalizedIngredient(
        name="flour",
        quantity="2",
        unit="cups",
        raw="2 cups all-purpose flour"
    )

    assert ing.name == "flour"
    assert ing.quantity == "2"
    assert ing.unit == "cups"
    assert ing.raw == "2 cups all-purpose flour"


def test_rating_stats_model():
    """Test RatingStats model."""
    stats = RatingStats(rating_avg=4.25, rating_count=100)

    assert stats.rating_avg == 4.25
    assert stats.rating_count == 100

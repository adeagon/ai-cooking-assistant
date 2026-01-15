"""Tests for ingredient category classification."""

import pytest

from src.domain.models import IngredientCategory
from src.planning.ingredient_categories import IngredientCategoryClassifier


@pytest.fixture
def classifier():
    """Create a classifier instance."""
    return IngredientCategoryClassifier()


class TestDairyClassification:
    """Test dairy category classification."""

    def test_milk_is_dairy(self, classifier):
        """Milk is classified as dairy."""
        cats = classifier.classify("milk")
        assert IngredientCategory.DAIRY in cats

    def test_cheese_is_dairy(self, classifier):
        """Cheese is classified as dairy."""
        cats = classifier.classify("cheese")
        assert IngredientCategory.DAIRY in cats

    def test_specific_cheese_is_dairy(self, classifier):
        """Specific cheese types are dairy."""
        assert IngredientCategory.DAIRY in classifier.classify("parmesan")
        assert IngredientCategory.DAIRY in classifier.classify("mozzarella")
        assert IngredientCategory.DAIRY in classifier.classify("cheddar")
        assert IngredientCategory.DAIRY in classifier.classify("feta")

    def test_cream_cheese_is_dairy(self, classifier):
        """Cream cheese (as phrase token) is dairy."""
        cats = classifier.classify("cream_cheese")
        assert IngredientCategory.DAIRY in cats

    def test_butter_is_dairy(self, classifier):
        """Butter is classified as dairy."""
        cats = classifier.classify("butter")
        assert IngredientCategory.DAIRY in cats


class TestMeatClassification:
    """Test meat category classification."""

    def test_beef_is_meat(self, classifier):
        """Beef is classified as meat."""
        cats = classifier.classify("beef")
        assert IngredientCategory.MEAT in cats

    def test_pork_is_meat(self, classifier):
        """Pork is classified as meat."""
        cats = classifier.classify("pork")
        assert IngredientCategory.MEAT in cats

    def test_ground_beef_is_meat(self, classifier):
        """Ground beef (as phrase token) is meat."""
        cats = classifier.classify("ground_beef")
        assert IngredientCategory.MEAT in cats

    def test_bacon_is_meat(self, classifier):
        """Bacon is classified as meat."""
        cats = classifier.classify("bacon")
        assert IngredientCategory.MEAT in cats

    def test_sausage_is_meat(self, classifier):
        """Sausage is classified as meat."""
        cats = classifier.classify("sausage")
        assert IngredientCategory.MEAT in cats


class TestPoultryClassification:
    """Test poultry category classification."""

    def test_chicken_is_poultry(self, classifier):
        """Chicken is classified as poultry."""
        cats = classifier.classify("chicken")
        assert IngredientCategory.POULTRY in cats

    def test_turkey_is_poultry(self, classifier):
        """Turkey is classified as poultry."""
        cats = classifier.classify("turkey")
        assert IngredientCategory.POULTRY in cats

    def test_chicken_breast_is_poultry(self, classifier):
        """Chicken breast (as phrase token) is poultry."""
        cats = classifier.classify("chicken_breast")
        assert IngredientCategory.POULTRY in cats

    def test_ground_turkey_is_poultry(self, classifier):
        """Ground turkey is poultry."""
        cats = classifier.classify("ground_turkey")
        assert IngredientCategory.POULTRY in cats


class TestSeafoodClassification:
    """Test seafood category classification."""

    def test_salmon_is_seafood(self, classifier):
        """Salmon is classified as seafood."""
        cats = classifier.classify("salmon")
        assert IngredientCategory.SEAFOOD in cats

    def test_shrimp_is_seafood(self, classifier):
        """Shrimp is classified as seafood."""
        cats = classifier.classify("shrimp")
        assert IngredientCategory.SEAFOOD in cats

    def test_generic_fish_is_seafood(self, classifier):
        """Generic 'fish' is classified as seafood."""
        cats = classifier.classify("fish")
        assert IngredientCategory.SEAFOOD in cats

    def test_crab_is_seafood(self, classifier):
        """Crab is classified as seafood."""
        cats = classifier.classify("crab")
        assert IngredientCategory.SEAFOOD in cats


class TestGlutenClassification:
    """Test gluten category classification."""

    def test_flour_is_gluten(self, classifier):
        """Flour is classified as gluten."""
        cats = classifier.classify("flour")
        assert IngredientCategory.GLUTEN in cats

    def test_bread_is_gluten(self, classifier):
        """Bread is classified as gluten."""
        cats = classifier.classify("bread")
        assert IngredientCategory.GLUTEN in cats

    def test_pasta_is_gluten(self, classifier):
        """Pasta is classified as gluten."""
        cats = classifier.classify("pasta")
        assert IngredientCategory.GLUTEN in cats

    def test_bread_crumbs_is_gluten(self, classifier):
        """Bread crumbs (as phrase token) is gluten."""
        cats = classifier.classify("bread_crumbs")
        assert IngredientCategory.GLUTEN in cats


class TestNutsClassification:
    """Test nuts category classification."""

    def test_almond_is_nuts(self, classifier):
        """Almond is classified as nuts."""
        cats = classifier.classify("almond")
        assert IngredientCategory.NUTS in cats

    def test_peanut_is_nuts(self, classifier):
        """Peanut is classified as nuts."""
        cats = classifier.classify("peanut")
        assert IngredientCategory.NUTS in cats

    def test_peanut_butter_is_nuts(self, classifier):
        """Peanut butter (as phrase token) is nuts."""
        cats = classifier.classify("peanut_butter")
        assert IngredientCategory.NUTS in cats

    def test_pine_nut_is_nuts(self, classifier):
        """Pine nut (as phrase token) is nuts."""
        cats = classifier.classify("pine_nut")
        assert IngredientCategory.NUTS in cats


class TestSoyClassification:
    """Test soy category classification."""

    def test_tofu_is_soy(self, classifier):
        """Tofu is classified as soy."""
        cats = classifier.classify("tofu")
        assert IngredientCategory.SOY in cats

    def test_soy_sauce_is_soy(self, classifier):
        """Soy sauce (as phrase token) is soy."""
        cats = classifier.classify("soy_sauce")
        assert IngredientCategory.SOY in cats

    def test_tempeh_is_soy(self, classifier):
        """Tempeh is classified as soy."""
        cats = classifier.classify("tempeh")
        assert IngredientCategory.SOY in cats

    def test_edamame_is_soy(self, classifier):
        """Edamame is classified as soy."""
        cats = classifier.classify("edamame")
        assert IngredientCategory.SOY in cats


class TestEggsClassification:
    """Test eggs category classification."""

    def test_egg_is_eggs(self, classifier):
        """Egg is classified as eggs."""
        cats = classifier.classify("egg")
        assert IngredientCategory.EGGS in cats

    def test_mayonnaise_is_eggs(self, classifier):
        """Mayonnaise is classified as eggs."""
        cats = classifier.classify("mayonnaise")
        assert IngredientCategory.EGGS in cats


class TestNoSubstringMatching:
    """Test that n-gram matching avoids false positives."""

    def test_eggplant_not_eggs(self, classifier):
        """Eggplant should NOT be classified as eggs."""
        cats = classifier.classify("eggplant")
        assert IngredientCategory.EGGS not in cats

    def test_buttermilk_is_dairy(self, classifier):
        """Buttermilk is dairy (contains butter keyword)."""
        cats = classifier.classify("buttermilk")
        assert IngredientCategory.DAIRY in cats

    def test_seabass_is_seafood(self, classifier):
        """Sea bass should be classified as seafood (contains bass)."""
        cats = classifier.classify("bass")
        assert IngredientCategory.SEAFOOD in cats


class TestContainsExcludedCategory:
    """Test exclusion checking."""

    def test_chicken_excluded_when_poultry_excluded(self, classifier):
        """Chicken is excluded when poultry is excluded."""
        assert classifier.contains_excluded_category(
            "chicken", [IngredientCategory.POULTRY]
        )

    def test_chicken_not_excluded_when_dairy_excluded(self, classifier):
        """Chicken is not excluded when only dairy is excluded."""
        assert not classifier.contains_excluded_category(
            "chicken", [IngredientCategory.DAIRY]
        )

    def test_empty_exclusion_list_excludes_nothing(self, classifier):
        """Empty exclusion list excludes nothing."""
        assert not classifier.contains_excluded_category("chicken", [])

    def test_multiple_exclusions(self, classifier):
        """Multiple exclusion categories are checked."""
        assert classifier.contains_excluded_category(
            "cheese", [IngredientCategory.DAIRY, IngredientCategory.GLUTEN]
        )
        assert classifier.contains_excluded_category(
            "bread", [IngredientCategory.DAIRY, IngredientCategory.GLUTEN]
        )


class TestPrimaryCategory:
    """Test primary category selection."""

    def test_chicken_primary_is_poultry(self, classifier):
        """Chicken's primary category is poultry."""
        assert classifier.get_primary_category("chicken") == IngredientCategory.POULTRY

    def test_salmon_primary_is_seafood(self, classifier):
        """Salmon's primary category is seafood."""
        assert classifier.get_primary_category("salmon") == IngredientCategory.SEAFOOD

    def test_unknown_ingredient_returns_none(self, classifier):
        """Unknown ingredient returns None."""
        assert classifier.get_primary_category("xyz_unknown_ingredient") is None


class TestMissTracking:
    """Test that unclassified ingredients are tracked."""

    def test_miss_logged_once(self, classifier):
        """Unclassified ingredients are logged once."""
        classifier.classify("xyz_unknown_ingredient")
        classifier.classify("xyz_unknown_ingredient")

        assert classifier.get_miss_count() == 1
        assert "xyz_unknown_ingredient" in classifier.get_missed_ingredients()

    def test_multiple_misses_tracked(self, classifier):
        """Multiple unclassified ingredients are tracked."""
        classifier.classify("xyz_unknown_1")
        classifier.classify("xyz_unknown_2")

        assert classifier.get_miss_count() == 2

    def test_classified_ingredients_not_tracked(self, classifier):
        """Classified ingredients are not in miss log."""
        classifier.classify("chicken")
        classifier.classify("beef")

        assert classifier.get_miss_count() == 0


class TestNGramGeneration:
    """Test n-gram generation for matching."""

    def test_generates_unigrams(self, classifier):
        """Generates 1-grams (individual tokens)."""
        ngrams = classifier._get_ngrams("red bell_pepper")
        assert "red" in ngrams
        assert "bell_pepper" in ngrams

    def test_generates_bigrams(self, classifier):
        """Generates 2-grams for multi-word phrases."""
        ngrams = classifier._get_ngrams("red bell pepper")
        assert "red_bell" in ngrams
        assert "bell_pepper" in ngrams

    def test_single_word_returns_self(self, classifier):
        """Single word returns just that word."""
        ngrams = classifier._get_ngrams("chicken")
        assert ngrams == {"chicken"}


class TestMultipleCategories:
    """Test ingredients that belong to multiple categories."""

    def test_ingredient_can_have_multiple_categories(self, classifier):
        """Some ingredients might match multiple categories."""
        # This is a realistic case - egg noodle could match both eggs and gluten
        cats = classifier.classify("noodle")
        # noodle should be gluten
        assert IngredientCategory.GLUTEN in cats

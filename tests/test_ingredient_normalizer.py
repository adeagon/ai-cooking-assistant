"""Tests for ingredient normalization."""

import pytest

from src.planning.ingredient_normalizer import IngredientNormalizer


@pytest.fixture
def normalizer():
    """Create a normalizer instance."""
    return IngredientNormalizer()


class TestBasicNormalization:
    """Test basic normalization functionality."""

    def test_lowercase(self, normalizer):
        """Normalize converts to lowercase."""
        assert normalizer.normalize("CHICKEN") == "chicken"
        assert normalizer.normalize("Tomato") == "tomato"

    def test_strips_whitespace(self, normalizer):
        """Normalize strips leading/trailing whitespace."""
        assert normalizer.normalize("  chicken  ") == "chicken"

    def test_strips_quantity_prefix(self, normalizer):
        """Normalize removes quantity prefixes."""
        assert normalizer.normalize("2 cups flour") == "flour"
        assert normalizer.normalize("1/2 lb chicken") == "chicken"
        assert normalizer.normalize("1.5 oz cheese") == "cheese"

    def test_strips_modifiers(self, normalizer):
        """Normalize removes common cooking modifiers."""
        assert normalizer.normalize("diced tomatoes") == "tomato"
        assert normalizer.normalize("fresh basil") == "basil"
        assert normalizer.normalize("finely chopped onion") == "onion"
        assert normalizer.normalize("boneless skinless chicken breast") == "chicken_breast"


class TestPhrasePreservation:
    """Test that bigrams are preserved as single tokens."""

    def test_green_onion_preserved(self, normalizer):
        """Green onion becomes green_onion."""
        assert normalizer.normalize("green onion") == "green_onion"
        assert normalizer.normalize("2 green onions") == "green_onion"

    def test_soy_sauce_preserved(self, normalizer):
        """Soy sauce becomes soy_sauce."""
        assert normalizer.normalize("soy sauce") == "soy_sauce"
        assert normalizer.normalize("2 tbsp soy sauce") == "soy_sauce"

    def test_olive_oil_preserved(self, normalizer):
        """Olive oil becomes olive_oil."""
        assert normalizer.normalize("olive oil") == "olive_oil"
        assert normalizer.normalize("extra virgin olive oil") == "olive_oil"

    def test_chicken_breast_preserved(self, normalizer):
        """Chicken breast stays as chicken_breast."""
        assert normalizer.normalize("chicken breast") == "chicken_breast"
        assert normalizer.normalize("boneless chicken breast") == "chicken_breast"

    def test_ground_beef_preserved(self, normalizer):
        """Ground beef becomes ground_beef."""
        assert normalizer.normalize("ground beef") == "ground_beef"
        assert normalizer.normalize("lean ground beef") == "ground_beef"

    def test_bell_pepper_preserved(self, normalizer):
        """Bell pepper becomes bell_pepper."""
        assert normalizer.normalize("bell pepper") == "bell_pepper"
        assert normalizer.normalize("red bell pepper") == "red bell_pepper"

    def test_cream_cheese_preserved(self, normalizer):
        """Cream cheese becomes cream_cheese."""
        assert normalizer.normalize("cream cheese") == "cream_cheese"
        assert normalizer.normalize("softened cream cheese") == "cream_cheese"


class TestWordBoundaryPreservation:
    """Test that phrase preservation respects word boundaries."""

    def test_no_partial_match_oil(self, normalizer):
        """'oil' in 'foil' should not be affected."""
        # This tests that we use word boundaries, not substring replacement
        result = normalizer.normalize("aluminum foil")
        assert "olive_oil" not in result

    def test_no_partial_match_cream(self, normalizer):
        """'cream' in 'ice cream' should not become 'sour_cream'."""
        result = normalizer.normalize("ice cream")
        assert result == "ice cream"
        assert "sour_cream" not in result


class TestSynonymNormalization:
    """Test synonym mapping to canonical forms."""

    def test_scallion_to_green_onion(self, normalizer):
        """Scallions normalize to green_onion."""
        assert normalizer.normalize("scallion") == "green_onion"
        assert normalizer.normalize("scallions") == "green_onion"

    def test_spring_onion_to_green_onion(self, normalizer):
        """Spring onion normalizes to green_onion."""
        assert normalizer.normalize("spring onion") == "green_onion"

    def test_coriander_to_cilantro(self, normalizer):
        """Coriander normalizes to cilantro."""
        assert normalizer.normalize("coriander") == "cilantro"
        assert normalizer.normalize("fresh coriander") == "cilantro"

    def test_aubergine_to_eggplant(self, normalizer):
        """Aubergine normalizes to eggplant."""
        assert normalizer.normalize("aubergine") == "eggplant"

    def test_courgette_to_zucchini(self, normalizer):
        """Courgette normalizes to zucchini."""
        assert normalizer.normalize("courgette") == "zucchini"

    def test_prawn_to_shrimp(self, normalizer):
        """Prawns normalize to shrimp."""
        assert normalizer.normalize("prawns") == "shrimp"
        assert normalizer.normalize("prawn") == "shrimp"


class TestPluralNormalization:
    """Test whitelist-based plural normalization."""

    def test_tomatoes_to_tomato(self, normalizer):
        """Tomatoes normalizes to tomato."""
        assert normalizer.normalize("tomatoes") == "tomato"

    def test_potatoes_to_potato(self, normalizer):
        """Potatoes normalizes to potato."""
        assert normalizer.normalize("potatoes") == "potato"

    def test_berries_to_berry(self, normalizer):
        """Berries normalizes to berry."""
        assert normalizer.normalize("berries") == "berry"

    def test_cherries_to_cherry(self, normalizer):
        """Cherries normalizes to cherry."""
        assert normalizer.normalize("cherries") == "cherry"

    def test_unknown_plural_unchanged(self, normalizer):
        """Unknown plurals are not modified (no naive -s stripping)."""
        # 'bass' should not become 'bas'
        assert normalizer.normalize("bass") == "bass"
        # 'grass' should not become 'gras'
        assert normalizer.normalize("grass") == "grass"


class TestStopIngredients:
    """Test stop ingredient detection."""

    def test_salt_is_stop(self, normalizer):
        """Salt is a stop ingredient."""
        assert normalizer.is_stop_ingredient("salt")
        assert normalizer.is_stop_ingredient("kosher salt")

    def test_pepper_is_stop(self, normalizer):
        """Pepper is a stop ingredient."""
        assert normalizer.is_stop_ingredient("pepper")
        assert normalizer.is_stop_ingredient("black_pepper")

    def test_oil_is_stop(self, normalizer):
        """Oil is a stop ingredient."""
        assert normalizer.is_stop_ingredient("oil")
        assert normalizer.is_stop_ingredient("olive_oil")
        assert normalizer.is_stop_ingredient("vegetable_oil")

    def test_water_is_stop(self, normalizer):
        """Water is a stop ingredient."""
        assert normalizer.is_stop_ingredient("water")

    def test_chicken_not_stop(self, normalizer):
        """Chicken is not a stop ingredient."""
        assert not normalizer.is_stop_ingredient("chicken")
        assert not normalizer.is_stop_ingredient("chicken_breast")

    def test_tomato_not_stop(self, normalizer):
        """Tomato is not a stop ingredient."""
        assert not normalizer.is_stop_ingredient("tomato")

    def test_garlic_not_stop(self, normalizer):
        """Garlic is not a stop ingredient."""
        assert not normalizer.is_stop_ingredient("garlic")


class TestTokenization:
    """Test tokenization for overlap comparison."""

    def test_simple_tokenize(self, normalizer):
        """Simple ingredients tokenize to single token."""
        tokens = normalizer.tokenize("chicken")
        assert tokens == {"chicken"}

    def test_phrase_stays_together(self, normalizer):
        """Preserved phrases stay as single tokens."""
        tokens = normalizer.tokenize("green_onion")
        assert tokens == {"green_onion"}

    def test_multi_word_tokenize(self, normalizer):
        """Multi-word ingredients split into tokens."""
        tokens = normalizer.tokenize("red bell_pepper")
        assert tokens == {"red", "bell_pepper"}

    def test_stop_tokens_excluded(self, normalizer):
        """Stop tokens are excluded from tokenization."""
        tokens = normalizer.tokenize("salt pepper")
        assert tokens == set()

    def test_stop_phrase_tokens_excluded(self, normalizer):
        """Stop phrase tokens are excluded from tokenization."""
        tokens = normalizer.tokenize("olive_oil")
        assert tokens == set()

    def test_mixed_tokens(self, normalizer):
        """Mixed stop and non-stop tokens filter correctly."""
        # This would be unusual but tests the filtering
        tokens = normalizer.tokenize("chicken salt")
        assert tokens == {"chicken"}


class TestGetTokensWithStops:
    """Test tokenization that includes stop ingredients."""

    def test_includes_stop_tokens(self, normalizer):
        """get_tokens_with_stops includes stop tokens."""
        tokens = normalizer.get_tokens_with_stops("chicken salt pepper")
        assert tokens == {"chicken", "salt", "pepper"}

    def test_includes_stop_phrase_tokens(self, normalizer):
        """get_tokens_with_stops includes stop phrase tokens."""
        tokens = normalizer.get_tokens_with_stops("olive_oil")
        assert tokens == {"olive_oil"}


class TestEndToEnd:
    """End-to-end normalization tests."""

    def test_recipe_ingredient_list(self, normalizer):
        """Test normalizing a typical recipe ingredient list."""
        ingredients = [
            "2 boneless skinless chicken breasts",
            "1/2 cup soy sauce",
            "2 tablespoons olive oil",
            "3 green onions, sliced",
            "2 garlic cloves, minced",  # garlic cloves (not cloves garlic)
            "1 red bell pepper, diced",
            "salt and pepper to taste",
        ]

        normalized = [normalizer.normalize(ing) for ing in ingredients]

        assert normalized == [
            "chicken_breast",
            "soy_sauce",
            "olive_oil",
            "green_onion,",  # comma remains after sliced is removed
            "garlic_clove,",  # garlic cloves becomes garlic_clove
            "red bell_pepper,",  # comma remains after diced is removed
            "salt and pepper",  # to/taste removed as modifiers
        ]

    def test_overlap_calculation_scenario(self, normalizer):
        """Test a scenario where we calculate ingredient overlap."""
        recipe1_ingredients = ["chicken", "garlic", "olive_oil", "lemon"]
        recipe2_ingredients = ["chicken", "garlic", "soy_sauce", "ginger"]

        tokens1 = set()
        tokens2 = set()

        for ing in recipe1_ingredients:
            tokens1.update(normalizer.tokenize(ing))

        for ing in recipe2_ingredients:
            tokens2.update(normalizer.tokenize(ing))

        # olive_oil and soy_sauce are stop ingredients, excluded from overlap
        assert tokens1 == {"chicken", "garlic", "lemon"}
        assert tokens2 == {"chicken", "garlic", "ginger"}

        overlap = tokens1 & tokens2
        assert overlap == {"chicken", "garlic"}

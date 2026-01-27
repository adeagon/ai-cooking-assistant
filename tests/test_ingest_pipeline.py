"""Tests for the data ingestion pipeline modules."""

import json
import pytest
import sqlite3
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call
import pandas as pd

from src.domain.models import Recipe, RatingStats
from src.ingest.load_foodcom import load_recipes, load_interactions, compute_ratings
from src.ingest.build_db import (
    create_tables,
    insert_recipes,
    get_recipe_by_id,
    get_all_recipes,
    get_stats
)

# Conditionally import vectorstore tests (requires sentence-transformers)
try:
    from src.ingest.build_vectorstore import (
        _extract_primary_cuisine,
        get_or_create_collection,
        build_vectorstore
    )
    HAS_VECTORSTORE = True
except ImportError:
    HAS_VECTORSTORE = False
    _extract_primary_cuisine = None
    get_or_create_collection = None
    build_vectorstore = None


# Check if kaggle is available and configured
def _check_kaggle_available():
    """Check if kaggle module can be imported without authentication errors."""
    try:
        # We need to check without triggering auth
        import importlib.util
        spec = importlib.util.find_spec("kaggle")
        return spec is not None
    except Exception:
        return False


HAS_KAGGLE = _check_kaggle_available()


# ============================================================================
# Tests for download.py
# ============================================================================


@pytest.mark.skipif(not HAS_KAGGLE, reason="kaggle module not available or not configured")
class TestDownloadFoodcomDataset:
    """Tests for the Kaggle dataset download functionality.

    Note: These tests mock the Kaggle API to avoid authentication issues.
    The download module is imported dynamically to avoid triggering Kaggle auth at import time.
    These tests are skipped if kaggle is not properly configured.
    """

    @pytest.fixture(autouse=True)
    def skip_if_kaggle_auth_fails(self):
        """Skip test if kaggle authentication causes issues."""
        pytest.skip("Kaggle tests require kaggle.json configuration")


# ============================================================================
# Tests for load_foodcom.py
# ============================================================================


class TestLoadRecipes:
    """Tests for loading recipes from CSV."""

    @pytest.fixture
    def sample_recipes_csv(self, tmp_path):
        """Create a sample recipes CSV file."""
        csv_path = tmp_path / "RAW_recipes.csv"
        csv_content = """id,name,minutes,contributor_id,submitted,tags,nutrition,n_steps,steps,description,ingredients,n_ingredients
1,Chocolate Cake,60,100,2008-01-01,"['dessert', 'chocolate']","[400, 20, 50, 30, 10, 5, 2]",3,"['mix', 'bake', 'cool']",A delicious cake,"['flour', 'sugar', 'cocoa']",3
2,Simple Pasta,30,101,2010-05-15,"['italian', 'quick']","[300, 15, 40, 20, 8, 3, 1]",2,"['boil', 'serve']",Easy pasta dish,"['pasta', 'sauce', 'cheese']",3"""
        csv_path.write_text(csv_content)
        return csv_path

    def test_load_recipes_yields_correct_fields(self, sample_recipes_csv):
        """Test that loaded recipes have all expected fields."""
        recipes = list(load_recipes(sample_recipes_csv, chunksize=10))

        assert len(recipes) == 2
        recipe = recipes[0]

        assert recipe['recipe_id'] == '1'
        assert recipe['name'] == 'Chocolate Cake'
        assert recipe['minutes'] == 60
        assert recipe['contributor_id'] == '100'
        assert recipe['tags'] == ['dessert', 'chocolate']
        assert recipe['steps'] == ['mix', 'bake', 'cool']
        assert recipe['ingredients'] == ['flour', 'sugar', 'cocoa']
        assert recipe['n_steps'] == 3
        assert recipe['n_ingredients'] == 3

    def test_load_recipes_parses_python_lists(self, sample_recipes_csv):
        """Test that string representations of lists are parsed correctly."""
        recipes = list(load_recipes(sample_recipes_csv, chunksize=10))

        # Verify tags, steps, and ingredients are actual lists
        for recipe in recipes:
            assert isinstance(recipe['tags'], list)
            assert isinstance(recipe['steps'], list)
            assert isinstance(recipe['ingredients'], list)
            assert isinstance(recipe['nutrition'], list)

    def test_load_recipes_handles_malformed_data(self, tmp_path):
        """Test that malformed recipes are skipped with warning."""
        csv_path = tmp_path / "malformed.csv"
        # Invalid Python list syntax in tags
        csv_content = """id,name,minutes,contributor_id,submitted,tags,nutrition,n_steps,steps,description,ingredients,n_ingredients
1,Good Recipe,30,100,2008-01-01,"['valid']","[100]",1,"['step']",desc,"['ing']",1
2,Bad Recipe,30,100,2008-01-01,"[invalid syntax","[100]",1,"['step']",desc,"['ing']",1"""
        csv_path.write_text(csv_content)

        recipes = list(load_recipes(csv_path, chunksize=10))

        # Only the valid recipe should be yielded
        assert len(recipes) == 1
        assert recipes[0]['name'] == 'Good Recipe'

    def test_load_recipes_handles_missing_values(self, tmp_path):
        """Test handling of missing/null values."""
        csv_path = tmp_path / "missing.csv"
        csv_content = """id,name,minutes,contributor_id,submitted,tags,nutrition,n_steps,steps,description,ingredients,n_ingredients
1,Recipe,,,2008-01-01,"[]","[]",,"[]",,"[]","""
        csv_path.write_text(csv_content)

        recipes = list(load_recipes(csv_path, chunksize=10))

        assert len(recipes) == 1
        recipe = recipes[0]
        assert recipe['minutes'] is None
        assert recipe['contributor_id'] is None
        assert recipe['description'] == ''
        assert recipe['tags'] == []


class TestLoadInteractions:
    """Tests for loading interactions from CSV."""

    @pytest.fixture
    def sample_interactions_csv(self, tmp_path):
        """Create a sample interactions CSV file."""
        csv_path = tmp_path / "RAW_interactions.csv"
        csv_content = """user_id,recipe_id,date,rating,review
100,1,2020-01-15,5,Great recipe!
101,1,2020-02-20,4,Pretty good
102,2,2020-03-10,3,Okay"""
        csv_path.write_text(csv_content)
        return csv_path

    def test_load_interactions_yields_correct_fields(self, sample_interactions_csv):
        """Test that loaded interactions have all expected fields."""
        interactions = list(load_interactions(sample_interactions_csv, chunksize=10))

        assert len(interactions) == 3
        interaction = interactions[0]

        assert interaction['user_id'] == '100'
        assert interaction['recipe_id'] == '1'
        assert interaction['rating'] == 5
        assert interaction['review'] == 'Great recipe!'

    def test_load_interactions_handles_missing_rating(self, tmp_path):
        """Test handling of missing ratings."""
        csv_path = tmp_path / "interactions.csv"
        csv_content = """user_id,recipe_id,date,rating,review
100,1,2020-01-15,,No rating given"""
        csv_path.write_text(csv_content)

        interactions = list(load_interactions(csv_path, chunksize=10))

        assert len(interactions) == 1
        assert interactions[0]['rating'] is None


class TestComputeRatings:
    """Tests for computing rating statistics."""

    def test_compute_ratings_basic(self, tmp_path):
        """Test basic rating computation."""
        recipes_csv = tmp_path / "recipes.csv"
        interactions_csv = tmp_path / "interactions.csv"

        recipes_csv.write_text("id,name\n1,Recipe1\n2,Recipe2")
        interactions_csv.write_text("""user_id,recipe_id,date,rating,review
100,1,2020-01-01,5,
101,1,2020-01-02,4,
102,2,2020-01-03,3,""")

        stats = compute_ratings(recipes_csv, interactions_csv)

        assert '1' in stats
        assert stats['1'].rating_avg == 4.5
        assert stats['1'].rating_count == 2
        assert '2' in stats
        assert stats['2'].rating_avg == 3.0
        assert stats['2'].rating_count == 1

    def test_compute_ratings_excludes_zero_ratings(self, tmp_path):
        """Test that zero ratings (reviews without ratings) are excluded."""
        recipes_csv = tmp_path / "recipes.csv"
        interactions_csv = tmp_path / "interactions.csv"

        recipes_csv.write_text("id,name\n1,Recipe1")
        interactions_csv.write_text("""user_id,recipe_id,date,rating,review
100,1,2020-01-01,5,
101,1,2020-01-02,0,Just a comment
102,1,2020-01-03,4,""")

        stats = compute_ratings(recipes_csv, interactions_csv)

        # Zero rating should be excluded, avg of 5 and 4 = 4.5
        assert stats['1'].rating_avg == 4.5
        assert stats['1'].rating_count == 2


# ============================================================================
# Tests for build_db.py
# ============================================================================


class TestCreateTables:
    """Tests for database table creation."""

    def test_create_tables_creates_directory(self, tmp_path):
        """Test that create_tables creates parent directories."""
        db_path = tmp_path / "nested" / "dir" / "recipes.db"

        create_tables(db_path)

        assert db_path.parent.exists()
        assert db_path.exists()

    def test_create_tables_creates_recipes_table(self, tmp_path):
        """Test that recipes table is created with correct schema."""
        db_path = tmp_path / "recipes.db"

        create_tables(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='recipes'"
        )
        assert cursor.fetchone() is not None

        # Check columns
        cursor.execute("PRAGMA table_info(recipes)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}

        assert 'recipe_id' in columns
        assert 'title' in columns
        assert 'ingredients_raw' in columns
        assert 'rating_avg' in columns
        assert 'minutes' in columns

        conn.close()

    def test_create_tables_creates_indexes(self, tmp_path):
        """Test that indexes are created."""
        db_path = tmp_path / "recipes.db"

        create_tables(db_path)

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        )
        indexes = [row[0] for row in cursor.fetchall()]

        assert 'idx_rating_avg' in indexes
        assert 'idx_rating_count' in indexes
        assert 'idx_minutes' in indexes

        conn.close()

    def test_create_tables_idempotent(self, tmp_path):
        """Test that create_tables can be called multiple times."""
        db_path = tmp_path / "recipes.db"

        create_tables(db_path)
        create_tables(db_path)  # Should not raise

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM recipes")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 0


class TestInsertRecipes:
    """Tests for inserting recipes into the database."""

    @pytest.fixture
    def db_path(self, tmp_path):
        """Create a database with tables."""
        path = tmp_path / "recipes.db"
        create_tables(path)
        return path

    def test_insert_recipes_basic(self, db_path):
        """Test basic recipe insertion."""
        recipes = iter([
            Recipe(
                recipe_id="1",
                title="Chocolate Cake",
                ingredients=["flour", "sugar", "cocoa"],
                ingredients_normalized=["flour", "sugar", "cocoa"],
                instructions=["mix", "bake"],
                tags=["dessert"],
                rating_avg=4.5,
                rating_count=100,
                minutes=60,
                n_steps=2,
                n_ingredients=3,
                source="foodcom"
            )
        ])

        count = insert_recipes(db_path, recipes)

        assert count == 1

        # Verify data
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT title, rating_avg FROM recipes WHERE recipe_id = '1'")
        row = cursor.fetchone()
        conn.close()

        assert row[0] == "Chocolate Cake"
        assert row[1] == 4.5

    def test_insert_recipes_batching(self, db_path):
        """Test that recipes are inserted in batches."""
        # Create more recipes than batch size
        recipes = [
            Recipe(
                recipe_id=str(i),
                title=f"Recipe {i}",
                ingredients=["ing"],
                ingredients_normalized=["ing"],
                instructions=["step"],
                tags=[],
                rating_avg=4.0,
                rating_count=10,
                minutes=30,
                n_steps=1,
                n_ingredients=1,
                source="test"
            )
            for i in range(2500)
        ]

        count = insert_recipes(db_path, iter(recipes))

        assert count == 2500

    def test_insert_recipes_upsert(self, db_path):
        """Test that duplicate recipe IDs are updated."""
        recipe1 = Recipe(
            recipe_id="1",
            title="Original Title",
            ingredients=["ing"],
            ingredients_normalized=["ing"],
            instructions=["step"],
            tags=[],
            rating_avg=3.0,
            rating_count=5,
            minutes=30,
            n_steps=1,
            n_ingredients=1,
            source="test"
        )

        recipe2 = Recipe(
            recipe_id="1",
            title="Updated Title",
            ingredients=["ing"],
            ingredients_normalized=["ing"],
            instructions=["step"],
            tags=[],
            rating_avg=4.0,
            rating_count=10,
            minutes=30,
            n_steps=1,
            n_ingredients=1,
            source="test"
        )

        insert_recipes(db_path, iter([recipe1]))
        insert_recipes(db_path, iter([recipe2]))

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT title, rating_avg FROM recipes WHERE recipe_id = '1'")
        row = cursor.fetchone()
        cursor.execute("SELECT COUNT(*) FROM recipes")
        count = cursor.fetchone()[0]
        conn.close()

        assert count == 1
        assert row[0] == "Updated Title"
        assert row[1] == 4.0


class TestGetRecipeById:
    """Tests for retrieving a recipe by ID."""

    @pytest.fixture
    def db_with_recipes(self, tmp_path):
        """Create a database with test recipes."""
        db_path = tmp_path / "recipes.db"
        create_tables(db_path)

        recipes = [
            Recipe(
                recipe_id="123",
                title="Test Recipe",
                ingredients=["flour", "sugar"],
                ingredients_normalized=["flour", "sugar"],
                instructions=["mix", "bake"],
                tags=["dessert", "easy"],
                rating_avg=4.5,
                rating_count=100,
                minutes=45,
                n_steps=2,
                n_ingredients=2,
                source="foodcom"
            )
        ]
        insert_recipes(db_path, iter(recipes))

        return db_path

    def test_get_recipe_by_id_found(self, db_with_recipes):
        """Test retrieving an existing recipe."""
        recipe = get_recipe_by_id(db_with_recipes, "123")

        assert recipe is not None
        assert recipe.recipe_id == "123"
        assert recipe.title == "Test Recipe"
        assert recipe.ingredients == ["flour", "sugar"]
        assert recipe.tags == ["dessert", "easy"]
        assert recipe.rating_avg == 4.5
        assert recipe.minutes == 45

    def test_get_recipe_by_id_not_found(self, db_with_recipes):
        """Test retrieving a non-existent recipe."""
        recipe = get_recipe_by_id(db_with_recipes, "999")

        assert recipe is None


class TestGetAllRecipes:
    """Tests for retrieving all recipes."""

    @pytest.fixture
    def db_with_multiple_recipes(self, tmp_path):
        """Create a database with multiple test recipes."""
        db_path = tmp_path / "recipes.db"
        create_tables(db_path)

        recipes = [
            Recipe(
                recipe_id="1",
                title="Best Recipe",
                ingredients=["ing"],
                ingredients_normalized=["ing"],
                instructions=["step"],
                tags=[],
                rating_avg=5.0,
                rating_count=100,
                minutes=30,
                n_steps=1,
                n_ingredients=1,
                source="test"
            ),
            Recipe(
                recipe_id="2",
                title="Good Recipe",
                ingredients=["ing"],
                ingredients_normalized=["ing"],
                instructions=["step"],
                tags=[],
                rating_avg=4.0,
                rating_count=50,
                minutes=45,
                n_steps=1,
                n_ingredients=1,
                source="test"
            ),
            Recipe(
                recipe_id="3",
                title="Unpopular Recipe",
                ingredients=["ing"],
                ingredients_normalized=["ing"],
                instructions=["step"],
                tags=[],
                rating_avg=4.5,
                rating_count=3,  # Below threshold
                minutes=20,
                n_steps=1,
                n_ingredients=1,
                source="test"
            ),
        ]
        insert_recipes(db_path, iter(recipes))

        return db_path

    def test_get_all_recipes_respects_limit(self, db_with_multiple_recipes):
        """Test that limit is respected."""
        recipes = get_all_recipes(db_with_multiple_recipes, limit=1)

        assert len(recipes) == 1

    def test_get_all_recipes_ordered_by_rating(self, db_with_multiple_recipes):
        """Test that recipes are ordered by rating."""
        recipes = get_all_recipes(db_with_multiple_recipes, limit=10)

        # Should only get recipes with rating_count >= 5
        assert len(recipes) == 2
        # Best rated first
        assert recipes[0].rating_avg >= recipes[1].rating_avg

    def test_get_all_recipes_filters_low_rating_count(self, db_with_multiple_recipes):
        """Test that recipes with few ratings are filtered."""
        recipes = get_all_recipes(db_with_multiple_recipes, limit=10)

        for recipe in recipes:
            assert recipe.rating_count >= 5


class TestGetStats:
    """Tests for database statistics."""

    @pytest.fixture
    def db_with_stats_data(self, tmp_path):
        """Create a database with data for stats."""
        db_path = tmp_path / "recipes.db"
        create_tables(db_path)

        recipes = [
            Recipe(
                recipe_id="1",
                title="Recipe 1",
                ingredients=[],
                ingredients_normalized=[],
                instructions=[],
                tags=[],
                rating_avg=4.0,
                rating_count=10,
                minutes=30,
                n_steps=1,
                n_ingredients=1,
                source="test"
            ),
            Recipe(
                recipe_id="2",
                title="Recipe 2",
                ingredients=[],
                ingredients_normalized=[],
                instructions=[],
                tags=[],
                rating_avg=5.0,
                rating_count=20,
                minutes=60,
                n_steps=1,
                n_ingredients=1,
                source="test"
            ),
        ]
        insert_recipes(db_path, iter(recipes))

        return db_path

    def test_get_stats_returns_correct_values(self, db_with_stats_data):
        """Test that statistics are computed correctly."""
        stats = get_stats(db_with_stats_data)

        assert stats['total_recipes'] == 2
        assert stats['avg_rating'] == 4.5  # (4.0 + 5.0) / 2
        assert stats['avg_minutes'] == 45.0  # (30 + 60) / 2

    def test_get_stats_empty_database(self, tmp_path):
        """Test stats on empty database."""
        db_path = tmp_path / "empty.db"
        create_tables(db_path)

        stats = get_stats(db_path)

        assert stats['total_recipes'] == 0
        assert stats['avg_rating'] is None
        assert stats['avg_minutes'] is None


# ============================================================================
# Tests for build_vectorstore.py
# ============================================================================


@pytest.mark.skipif(not HAS_VECTORSTORE, reason="sentence-transformers not installed")
class TestExtractPrimaryCuisine:
    """Tests for cuisine extraction from tags."""

    def test_extract_cuisine_italian(self):
        """Test extracting Italian cuisine."""
        assert _extract_primary_cuisine(["italian", "pasta"]) == "italian"

    def test_extract_cuisine_priority_order(self):
        """Test that priority order is respected."""
        # Italian comes before American in priority
        tags = ["american", "italian", "european"]
        assert _extract_primary_cuisine(tags) == "italian"

    def test_extract_cuisine_north_american_maps_to_american(self):
        """Test that north-american maps to american."""
        assert _extract_primary_cuisine(["north-american"]) == "american"

    def test_extract_cuisine_none_found(self):
        """Test when no cuisine is found."""
        assert _extract_primary_cuisine(["dessert", "easy"]) == ""

    def test_extract_cuisine_empty_tags(self):
        """Test with empty tags."""
        assert _extract_primary_cuisine([]) == ""


@pytest.mark.skipif(not HAS_VECTORSTORE, reason="sentence-transformers not installed")
class TestGetOrCreateCollection:
    """Tests for ChromaDB collection management."""

    def test_get_existing_collection(self):
        """Test getting an existing collection."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_collection.return_value = mock_collection

        result = get_or_create_collection(mock_client, "recipes")

        mock_client.get_collection.assert_called_once_with(name="recipes")
        assert result == mock_collection

    def test_create_collection_when_not_exists(self):
        """Test creating collection when it doesn't exist."""
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_collection.side_effect = Exception("Not found")
        mock_client.create_collection.return_value = mock_collection

        result = get_or_create_collection(mock_client, "recipes")

        mock_client.create_collection.assert_called_once_with(
            name="recipes",
            metadata={"hnsw:space": "cosine"}
        )
        assert result == mock_collection


@pytest.mark.skipif(not HAS_VECTORSTORE, reason="sentence-transformers not installed")
class TestBuildVectorstore:
    """Tests for building the vector store."""

    def test_build_vectorstore_creates_directory(self, tmp_path):
        """Test that build_vectorstore creates the chroma directory."""
        recipes_path = tmp_path / "recipes.jsonl"
        chroma_dir = tmp_path / "chroma"

        # Create empty recipes file
        recipes_path.write_text("")

        with patch('src.ingest.build_vectorstore.chromadb.PersistentClient') as mock_chroma:
            with patch('src.ingest.build_vectorstore.process_recipes_in_batches') as mock_process:
                mock_process.return_value = iter([])  # Empty iterator
                mock_client = MagicMock()
                mock_chroma.return_value = mock_client

                build_vectorstore(
                    recipes_path,
                    chroma_dir,
                    "all-mpnet-base-v2",
                    batch_size=100
                )

        assert chroma_dir.exists()

    def test_build_vectorstore_indexes_recipes(self, tmp_path):
        """Test that recipes are indexed with correct metadata."""
        recipes_path = tmp_path / "recipes.jsonl"
        chroma_dir = tmp_path / "chroma"

        recipes_path.write_text("")

        test_recipe = Recipe(
            recipe_id="123",
            title="Spaghetti Carbonara",
            ingredients=["pasta", "eggs", "bacon"],
            ingredients_normalized=["pasta", "eggs", "bacon"],
            instructions=["cook"],
            tags=["italian", "dinner", "vegetarian"],
            rating_avg=4.5,
            rating_count=100,
            minutes=30,
            n_steps=1,
            n_ingredients=3,
            source="test"
        )
        test_embedding = [0.1] * 768

        with patch('src.ingest.build_vectorstore.chromadb.PersistentClient') as mock_chroma:
            with patch('src.ingest.build_vectorstore.process_recipes_in_batches') as mock_process:
                mock_process.return_value = iter([
                    ([test_recipe], [test_embedding])
                ])

                mock_client = MagicMock()
                mock_collection = MagicMock()
                mock_chroma.return_value = mock_client
                mock_client.get_collection.return_value = mock_collection

                count = build_vectorstore(
                    recipes_path,
                    chroma_dir,
                    "all-mpnet-base-v2",
                    batch_size=100
                )

        assert count == 1

        # Verify upsert was called with correct data
        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args[1]

        assert call_kwargs['ids'] == ["123"]
        assert len(call_kwargs['metadatas']) == 1

        metadata = call_kwargs['metadatas'][0]
        assert metadata['title'] == "Spaghetti Carbonara"
        assert metadata['is_vegetarian'] is True
        assert metadata['cuisine'] == "italian"
        assert metadata['rating_avg'] == 4.5
        assert metadata['minutes'] == 30

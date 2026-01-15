"""SQLite database builder for recipes."""

import json
import sqlite3
from pathlib import Path
from typing import Iterator
from src.domain.models import Recipe
from src.app.logging_config import get_logger

logger = get_logger(__name__)


def create_tables(db_path: Path) -> None:
    """Create SQLite tables for recipes.

    Args:
        db_path: Path to SQLite database file
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recipes (
            recipe_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            ingredients_raw TEXT,
            ingredients_normalized TEXT,
            instructions TEXT,
            tags TEXT,
            rating_avg REAL,
            rating_count INTEGER,
            minutes INTEGER,
            n_steps INTEGER,
            n_ingredients INTEGER,
            source TEXT DEFAULT 'foodcom',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_rating_avg
        ON recipes(rating_avg)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_rating_count
        ON recipes(rating_count)
    """)

    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_minutes
        ON recipes(minutes)
    """)

    conn.commit()
    conn.close()

    logger.info("Database tables created", db_path=str(db_path))


def insert_recipes(db_path: Path, recipes: Iterator[Recipe]) -> int:
    """Insert recipes into SQLite database.

    Args:
        db_path: Path to SQLite database file
        recipes: Iterator of Recipe objects

    Returns:
        Number of recipes inserted
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    count = 0
    batch = []
    batch_size = 1000

    for recipe in recipes:
        batch.append((
            recipe.recipe_id,
            recipe.title,
            json.dumps(recipe.ingredients),
            json.dumps(recipe.ingredients_normalized),
            json.dumps(recipe.instructions),
            json.dumps(recipe.tags),
            recipe.rating_avg,
            recipe.rating_count,
            recipe.minutes,
            recipe.n_steps,
            recipe.n_ingredients,
            recipe.source
        ))

        if len(batch) >= batch_size:
            cursor.executemany("""
                INSERT OR REPLACE INTO recipes (
                    recipe_id, title, ingredients_raw, ingredients_normalized,
                    instructions, tags, rating_avg, rating_count,
                    minutes, n_steps, n_ingredients, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, batch)
            conn.commit()
            count += len(batch)
            logger.info("Inserted recipes", count=count)
            batch = []

    # Insert remaining recipes
    if batch:
        cursor.executemany("""
            INSERT OR REPLACE INTO recipes (
                recipe_id, title, ingredients_raw, ingredients_normalized,
                instructions, tags, rating_avg, rating_count,
                minutes, n_steps, n_ingredients, source
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, batch)
        conn.commit()
        count += len(batch)

    conn.close()

    logger.info("All recipes inserted", total_count=count)
    return count


def get_recipe_by_id(db_path: Path, recipe_id: str) -> Recipe | None:
    """Retrieve a single recipe by ID.

    Args:
        db_path: Path to SQLite database file
        recipe_id: Recipe ID to retrieve

    Returns:
        Recipe object or None if not found
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM recipes WHERE recipe_id = ?
    """, (recipe_id,))

    row = cursor.fetchone()
    conn.close()

    if row is None:
        return None

    return Recipe(
        recipe_id=row['recipe_id'],
        title=row['title'],
        ingredients=json.loads(row['ingredients_raw']),
        ingredients_normalized=json.loads(row['ingredients_normalized']),
        instructions=json.loads(row['instructions']),
        tags=json.loads(row['tags']),
        rating_avg=row['rating_avg'],
        rating_count=row['rating_count'],
        minutes=row['minutes'],
        n_steps=row['n_steps'],
        n_ingredients=row['n_ingredients'],
        source=row['source']
    )


def get_all_recipes(db_path: Path, limit: int = 500) -> list[Recipe]:
    """Retrieve recipes from database for meal planning.

    Args:
        db_path: Path to SQLite database file
        limit: Maximum number of recipes to retrieve

    Returns:
        List of Recipe objects
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get recipes with good ratings, ordered by rating
    cursor.execute("""
        SELECT * FROM recipes
        WHERE rating_avg IS NOT NULL AND rating_count >= 5
        ORDER BY rating_avg DESC, rating_count DESC
        LIMIT ?
    """, (limit,))

    rows = cursor.fetchall()
    conn.close()

    recipes = []
    for row in rows:
        try:
            recipes.append(Recipe(
                recipe_id=row['recipe_id'],
                title=row['title'],
                ingredients=json.loads(row['ingredients_raw']),
                ingredients_normalized=json.loads(row['ingredients_normalized']),
                instructions=json.loads(row['instructions']),
                tags=json.loads(row['tags']),
                rating_avg=row['rating_avg'],
                rating_count=row['rating_count'],
                minutes=row['minutes'],
                n_steps=row['n_steps'],
                n_ingredients=row['n_ingredients'],
                source=row['source']
            ))
        except (json.JSONDecodeError, KeyError):
            continue

    return recipes


def get_stats(db_path: Path) -> dict:
    """Get database statistics.

    Args:
        db_path: Path to SQLite database file

    Returns:
        Dict with statistics
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM recipes")
    total_count = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(rating_avg) FROM recipes WHERE rating_avg IS NOT NULL")
    avg_rating = cursor.fetchone()[0]

    cursor.execute("SELECT AVG(minutes) FROM recipes WHERE minutes IS NOT NULL")
    avg_minutes = cursor.fetchone()[0]

    conn.close()

    return {
        'total_recipes': total_count,
        'avg_rating': round(avg_rating, 2) if avg_rating else None,
        'avg_minutes': round(avg_minutes, 1) if avg_minutes else None
    }

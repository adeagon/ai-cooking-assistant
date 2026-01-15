"""CLI application for the recipe assistant."""

import typer
from rich.console import Console
from rich.panel import Panel
from src import __version__
from src.app.settings import settings
from src.app.logging_config import configure_logging, get_logger
from src.ingest import cli as ingest_cli

# Initialize Typer app
app = typer.Typer(
    name="recipe-assistant",
    help="Local recipe assistant using RAG with local LLM",
    add_completion=False
)

# Add ingest subcommand
app.add_typer(ingest_cli.app, name="ingest")

# Initialize Rich console for pretty output
console = Console()

# Configure logging
configure_logging(settings.log_level)
logger = get_logger(__name__)


def normalize_for_matching(text: str) -> str:
    """Normalize text for fuzzy recipe name matching.

    Handles apostrophes, extra spaces, and case differences.

    Args:
        text: Text to normalize

    Returns:
        Normalized lowercase text for comparison
    """
    import re
    # Remove various apostrophe characters
    text = text.lower()
    text = text.replace("'", "").replace("'", "").replace("`", "")
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


@app.command()
def chat():
    """Start an interactive recipe assistant chat session."""
    import asyncio
    from pathlib import Path

    # Check if vector store and database exist
    chroma_dir = Path(settings.chroma_persist_dir)
    if not chroma_dir.exists():
        console.print(f"[red]ERROR: Vector store not found at: {chroma_dir}[/red]")
        console.print("[yellow]Run 'ingest embed' first to build the vector store[/yellow]")
        raise typer.Exit(1)

    if not settings.sqlite_db_path.exists():
        console.print(f"[red]ERROR: Database not found at: {settings.sqlite_db_path}[/red]")
        console.print("[yellow]Run 'ingest process' first to build the database[/yellow]")
        raise typer.Exit(1)

    # Run async chat session
    asyncio.run(async_chat_session())


def strip_articles(text: str) -> str:
    """Strip leading articles (the, a, an) from text.

    Args:
        text: Text to process

    Returns:
        Text with leading articles removed
    """
    words = text.split()
    if words and words[0].lower() in ("the", "a", "an"):
        return " ".join(words[1:])
    return text


def resolve_recipe_reference(ref: str, last_cards: list) -> tuple[str, str] | None:
    """Resolve recipe reference (number or name) to (recipe_id, title).

    Args:
        ref: Reference string (e.g., "1", "chicken tacos")
        last_cards: List of last recommended RecipeCard objects

    Returns:
        Tuple of (recipe_id, title) or None if not found
    """
    ref = ref.strip().strip('"\'')

    # Return None for empty reference
    if not ref:
        return None

    # Try as number first (1-indexed)
    if ref.isdigit():
        idx = int(ref) - 1
        if 0 <= idx < len(last_cards):
            return (last_cards[idx].recipe_id, last_cards[idx].title)
        return None

    # Normalize reference: lowercase and strip articles
    ref_normalized = strip_articles(ref.lower())

    # Try fuzzy match on title
    for card in last_cards:
        title_normalized = strip_articles(card.title.lower())

        # Check if ref is in title OR title is in ref (handles partial names)
        if ref_normalized in title_normalized or title_normalized in ref_normalized:
            return (card.recipe_id, card.title)

        # Also try matching without the full reference (for cases like "chicken orzo salad"
        # matching "asian chicken orzo salad")
        ref_words = set(ref_normalized.split())
        title_words = set(title_normalized.split())
        # If all significant ref words appear in title, consider it a match
        if ref_words and ref_words.issubset(title_words):
            return (card.recipe_id, card.title)

    return None


def display_full_recipe(recipe, console: Console):
    """Display complete recipe with ingredients and instructions.

    Args:
        recipe: Recipe object from database
        console: Rich console for output
    """
    from src.domain.models import Recipe

    rating_str = ""
    if recipe.rating_avg and recipe.rating_count:
        rating_str = f"Rating: {recipe.rating_avg:.1f}/5 ({recipe.rating_count} reviews)\n"

    time_str = ""
    if recipe.minutes:
        rating_str += f"Time: {recipe.minutes} minutes\n"

    ingredients_str = "\n".join(f"  • {ing}" for ing in recipe.ingredients)
    instructions_str = "\n".join(f"  {i+1}. {step}" for i, step in enumerate(recipe.instructions))

    console.print(Panel(
        f"[bold]{recipe.title}[/bold]\n"
        f"{rating_str}{time_str}\n"
        f"[bold]Ingredients:[/bold]\n{ingredients_str}\n\n"
        f"[bold]Instructions:[/bold]\n{instructions_str}",
        title="Recipe Details",
        border_style="cyan"
    ))


def update_learned_preferences(feedback_store, profile_store, console: Console | None = None) -> bool:
    """Update user profile with preferences learned from feedback.

    Args:
        feedback_store: FeedbackStore instance
        profile_store: ProfileStore instance
        console: Optional console for output

    Returns:
        True if preferences were updated
    """
    # Get cuisines learned from likes (requires 3+ likes)
    learned_cuisines = feedback_store.get_preferred_cuisines_from_likes(min_count=3)

    if learned_cuisines:
        # Get current profile
        profile = profile_store.load()
        current_cuisines = set(c.lower() for c in profile.preferred_cuisines)
        new_cuisines = [c for c in learned_cuisines if c.lower() not in current_cuisines]

        if new_cuisines:
            # Merge learned cuisines with existing preferences
            updated_cuisines = list(profile.preferred_cuisines) + new_cuisines
            profile_store.update(preferred_cuisines=updated_cuisines)

            if console:
                console.print(f"[dim]📚 Learned preference: {', '.join(new_cuisines)}[/dim]")
            return True

    return False


def execute_intent(
    intent_result,
    last_cards: list,
    feedback_store,
    history_store,
    recipe_box_store,
    profile_store,
    session_store,
    session_id: str,
    settings,
    console: Console
) -> bool:
    """Execute detected intent command.

    Args:
        intent_result: IntentClassification object
        last_cards: List of last recommended RecipeCard objects
        feedback_store: FeedbackStore instance
        history_store: HistoryStore instance
        recipe_box_store: RecipeBoxStore instance
        profile_store: ProfileStore instance
        session_store: SessionStore instance
        session_id: Current session ID
        settings: Application settings
        console: Rich console for output

    Returns:
        True if command was executed, False if should fall through to chat
    """
    from src.domain.models import RecipeFeedback
    from src.ingest.build_db import get_recipe_by_id

    intent = intent_result.intent
    ref = intent_result.recipe_reference

    # Skip if conversation or low confidence
    if intent == "conversation" or intent_result.confidence == "low":
        return False

    # Handle stateless commands (no recipe reference needed)
    if intent == "history":
        history = history_store.get_cooking_history(limit=10)
        if history:
            console.print(f"\n[bold]Recent Cooking History:[/bold]")
            for i, entry in enumerate(history, 1):
                recipe = get_recipe_by_id(settings.sqlite_db_path, entry.recipe_id)
                title = recipe.title if recipe else entry.recipe_id
                cooked_str = entry.cooked_at.strftime("%Y-%m-%d") if entry.cooked_at else "Unknown"
                console.print(f"  {i}. {title} (cooked: {cooked_str})")
        else:
            console.print("[yellow]No cooking history yet[/yellow]")
        return True

    if intent == "box":
        saved_recipes = recipe_box_store.get_saved_recipes(limit=50)
        if saved_recipes:
            console.print(f"\n[bold]Recipe Box ({len(saved_recipes)} saved):[/bold]")
            for i, saved in enumerate(saved_recipes, 1):
                saved_str = saved.saved_at.strftime("%Y-%m-%d") if saved.saved_at else "Unknown"
                console.print(f"  {i}. {saved.title} (saved: {saved_str})")
        else:
            console.print("[yellow]Recipe Box is empty. Use /save <recipe> to add recipes![/yellow]")
        return True

    if intent == "new":
        new_session_id = session_store.create()
        console.print("[green]✓ Started new session[/green]")
        logger.info("New session created", session_id=new_session_id)
        # Note: session_id update would need to be handled by caller
        return True

    if intent == "prefs":
        profile = profile_store.load()
        console.print(f"\n[bold]Your Preferences:[/bold]")
        console.print(f"  Spice level: {profile.spice_level}")
        console.print(f"  Diet: {profile.diet}")
        if profile.avoid_ingredients:
            console.print(f"  Avoid: {', '.join(profile.avoid_ingredients)}")
        if profile.preferred_cuisines:
            console.print(f"  Cuisines: {', '.join(profile.preferred_cuisines)}")
        return True

    if intent == "commands":
        # Display the same help as /commands slash command
        console.print(Panel.fit(
            "[bold cyan]Available Commands[/bold cyan]\n\n"
            "[bold]Session:[/bold]\n"
            "  /new             - Start a new session\n"
            "  /prefs           - Show your preferences\n"
            "  /addpref <type> <value> - Add a preference\n"
            "                    Types: cuisine, avoid, diet, spice, time\n"
            "  /commands        - Show this help\n\n"
            "[bold]Recipe Feedback:[/bold]\n"
            "  /like <ref>      - Like a recipe\n"
            "  /dislike <ref>   - Dislike a recipe\n"
            "  /rate <1-5> <ref> - Rate a recipe\n"
            "  /cooked <ref>    - Mark recipe as cooked\n\n"
            "[bold]Recipe Box:[/bold]\n"
            "  /save <ref>      - Save recipe to your box\n"
            "  /unsave <ref>    - Remove recipe from box\n"
            "  /box             - List all saved recipes\n"
            "  /show <ref>      - Show full recipe\n"
            "  /show box <N>    - Show recipe from box\n\n"
            "[bold]Meal Planning:[/bold]\n"
            "  /mealplan        - Start meal planning\n"
            "  /plan            - View current meal plan\n"
            "  /grocery         - Generate grocery list\n\n"
            "[bold]History:[/bold]\n"
            "  /history         - View cooking history\n\n"
            "[dim]<ref> can be: number, name, 'it', 'that', etc.[/dim]",
            title="Help"
        ))
        return True

    # Handle meal planning intents
    if intent == "mealplan":
        console.print("[yellow]Meal planning mode activated. Use /mealplan for the full interactive experience.[/yellow]")
        return True

    if intent == "show_plan":
        console.print("[yellow]Use /plan to view your current meal plan.[/yellow]")
        return True

    if intent == "grocery_list":
        console.print("[yellow]Use /grocery to generate a grocery list from your meal plan.[/yellow]")
        return True

    # Handle filter_previous intent - sort/filter previous recommendations
    if intent == "filter_previous":
        if not last_cards:
            console.print("[yellow]No recent recommendations to filter. Try asking for recipe suggestions first.[/yellow]")
            return True

        filter_type = intent_result.filter_type or "best_rated"
        sorted_cards = list(last_cards)  # Copy to avoid mutating original

        if filter_type in ("best_rated", "best rated", "highest rated", "best reviews"):
            # Sort by rating (highest first), handle None values
            sorted_cards.sort(key=lambda c: (c.rating_avg or 0, c.rating_count or 0), reverse=True)
            console.print(f"\n[bold]Sorted by best rating:[/bold]")
        elif filter_type in ("quickest", "fastest", "least time", "shortest"):
            # Sort by time (lowest first), handle None values
            sorted_cards.sort(key=lambda c: c.time_total if c.time_total else float('inf'))
            console.print(f"\n[bold]Sorted by quickest time:[/bold]")
        elif filter_type in ("most_reviewed", "most reviewed", "most reviews", "popular"):
            # Sort by review count (highest first)
            sorted_cards.sort(key=lambda c: c.rating_count or 0, reverse=True)
            console.print(f"\n[bold]Sorted by most reviewed:[/bold]")
        else:
            # Default to rating
            sorted_cards.sort(key=lambda c: (c.rating_avg or 0, c.rating_count or 0), reverse=True)
            console.print(f"\n[bold]Sorted by rating ({filter_type}):[/bold]")

        # Display sorted results
        for i, card in enumerate(sorted_cards, 1):
            rating_str = f"{card.rating_avg:.1f}/5" if card.rating_avg else "N/A"
            review_str = f"({card.rating_count} reviews)" if card.rating_count else ""
            time_str = f" | {card.time_total}m" if card.time_total else ""
            console.print(f"  {i}. {card.title} - {rating_str} {review_str}{time_str}")

        return True

    # Handle recipe-reference commands (need a recipe)
    if not ref:
        console.print("[yellow]Which recipe do you mean? Try being more specific or use a number.[/yellow]")
        return True

    # Check if we should resolve from Recipe Box (natural language "from my recipe box")
    result = None
    if intent_result.source == "box":
        # Try to resolve from Recipe Box first
        saved_recipes = recipe_box_store.get_saved_recipes(limit=50)
        if saved_recipes:
            # Try numeric reference first
            try:
                idx = int(ref) - 1
                if 0 <= idx < len(saved_recipes):
                    result = (saved_recipes[idx].recipe_id, saved_recipes[idx].title)
            except ValueError:
                # Try name matching with normalization (handles apostrophes, articles, etc.)
                ref_normalized = strip_articles(normalize_for_matching(ref))
                for saved in saved_recipes:
                    title_normalized = strip_articles(normalize_for_matching(saved.title))
                    # Check if ref is in title OR title is in ref
                    if ref_normalized in title_normalized or title_normalized in ref_normalized:
                        result = (saved.recipe_id, saved.title)
                        break
                    # Also try word-based matching
                    ref_words = set(ref_normalized.split())
                    title_words = set(title_normalized.split())
                    if ref_words and ref_words.issubset(title_words):
                        result = (saved.recipe_id, saved.title)
                        break

    # Fall back to last recommendations if no box match or not from box
    if not result:
        result = resolve_recipe_reference(ref, last_cards)

    # If "it" or "that" and no match, try index 0 (most recent)
    if not result and ref.lower() in ["it", "that", "this"] and last_cards:
        result = (last_cards[0].recipe_id, last_cards[0].title)

    if not result:
        console.print(f"[yellow]Couldn't find recipe: {ref}[/yellow]")
        return True

    recipe_id, title = result

    # Execute command based on intent
    if intent == "like":
        feedback_store.add_feedback(RecipeFeedback(
            recipe_id=recipe_id,
            feedback_type="like",
            session_id=session_id
        ))
        console.print(f"[green]✓ Liked: {title}[/green]")
        # Check for learned preferences after liking
        update_learned_preferences(feedback_store, profile_store, console)
        return True

    if intent == "dislike":
        feedback_store.add_feedback(RecipeFeedback(
            recipe_id=recipe_id,
            feedback_type="dislike",
            session_id=session_id
        ))
        console.print(f"[yellow]✓ Disliked: {title}[/yellow]")
        return True

    if intent == "rate":
        rating = intent_result.rating_value
        if not rating or not 1 <= rating <= 5:
            console.print("[yellow]What rating would you give (1-5 stars)?[/yellow]")
            return True

        feedback_store.add_feedback(RecipeFeedback(
            recipe_id=recipe_id,
            feedback_type="rate",
            rating=rating,
            session_id=session_id
        ))
        console.print(f"[green]✓ Rated {title}: {rating}/5[/green]")
        return True

    if intent == "show":
        recipe = get_recipe_by_id(settings.sqlite_db_path, recipe_id)
        if recipe:
            display_full_recipe(recipe, console)
        else:
            console.print(f"[yellow]Recipe not found in database: {recipe_id}[/yellow]")
        return True

    if intent == "cooked":
        history_store.add_cooked(recipe_id)
        console.print(f"[green]✓ Marked as cooked: {title}[/green]")
        return True

    if intent == "save":
        try:
            recipe_box_store.save_recipe(recipe_id, title)
            console.print(f"[green]✓ Saved to Recipe Box: {title}[/green]")
        except Exception as e:
            if "UNIQUE" in str(e):
                console.print(f"[yellow]Already saved: {title}[/yellow]")
            else:
                console.print(f"[red]Error saving recipe: {e}[/red]")
        return True

    if intent == "unsave":
        if recipe_box_store.remove_recipe(recipe_id):
            console.print(f"[green]✓ Removed from Recipe Box: {title}[/green]")
        else:
            console.print(f"[yellow]Recipe not found in box: {title}[/yellow]")
        return True

    # Unknown intent (shouldn't happen, but be safe)
    return False


async def handle_mealplan_command(
    input_text: str,
    profile,
    recipe_box_store,
    meal_plan_store,
    settings,
    console: Console
):
    """Handle the /mealplan command to generate a meal plan.

    Args:
        input_text: Optional constraints from user (e.g., "5 vegetarian dinners")
        profile: User's preference profile
        recipe_box_store: RecipeBoxStore for saved recipes
        meal_plan_store: MealPlanStore for storing plans
        settings: Application settings
        console: Rich console for output
    """
    from datetime import date, timedelta
    from src.planning.constraint_extractor import MealPlanConstraintExtractor
    from src.planning.meal_planner import MealPlanner
    from src.ingest.build_db import get_recipe_by_id

    console.print(Panel.fit(
        "[bold cyan]Meal Planning[/bold cyan]\n\n"
        "I'll help you plan meals for the week.\n"
        "Defaults: 5 dinners, starting today\n\n"
        "[dim]Examples:[/dim]\n"
        "  /mealplan\n"
        "  /mealplan 5 vegetarian dinners\n"
        "  /mealplan quick weeknight meals\n"
        "  /mealplan 7 days, no casseroles",
        border_style="cyan"
    ))

    # Extract constraints from input
    extractor = MealPlanConstraintExtractor(db_path=settings.sqlite_db_path)
    constraints = extractor.extract(
        input_text if input_text else "plan dinners",
        profile=profile
    )

    # Show extracted constraints
    console.print(f"\n[bold]Planning {constraints.days} {constraints.meal_types[0] if constraints.meal_types else 'dinner'}(s)[/bold]")
    if constraints.dietary.value != "none":
        console.print(f"  Diet: {constraints.dietary.value}")
    if constraints.max_prep_time:
        console.print(f"  Max time: {constraints.max_prep_time} minutes")
    if constraints.excluded_categories:
        console.print(f"  Excluding: {', '.join(c.value for c in constraints.excluded_categories)}")
    if constraints.excluded_tags:
        console.print(f"  No: {', '.join(constraints.excluded_tags)}")

    console.print("\n[dim]Generating plan...[/dim]")

    try:
        # Get saved recipes from Recipe Box
        saved_recipes = recipe_box_store.get_saved_recipes(limit=100)
        box_recipe_ids = {r.recipe_id for r in saved_recipes}

        # Initialize planner and generate plan
        planner = MealPlanner(db_path=settings.sqlite_db_path)
        meals, metrics = planner.generate_plan(constraints, profile, box_recipe_ids=box_recipe_ids)

        if not meals:
            console.print("[yellow]No recipes found matching your criteria. Try relaxing some constraints.[/yellow]")
            return

        # Create meal plan object
        from src.domain.models import MealPlan
        start_date = constraints.start_date or date.today()
        end_date = start_date + timedelta(days=constraints.days - 1)

        plan = MealPlan(
            start_date=start_date,
            end_date=end_date,
            meal_types=constraints.meal_types or ["dinner"],
            status="draft",
            constraints=constraints.model_dump(exclude={"extraction_sources"}),
            metrics=metrics,
            meals=meals
        )

        # Save plan
        plan_id = meal_plan_store.save_plan(plan)
        plan.id = plan_id

        # Display the plan
        console.print(f"\n[bold green]✓ Meal Plan Generated[/bold green] (ID: {plan_id})\n")

        # Group meals by day
        from collections import defaultdict
        meals_by_day = defaultdict(list)
        for meal in meals:
            meals_by_day[meal.day].append(meal)

        for day in sorted(meals_by_day.keys()):
            day_meals = meals_by_day[day]
            day_str = day.strftime("%A, %b %d")
            console.print(f"[bold]{day_str}[/bold]")
            for meal in day_meals:
                source_icon = "📦" if meal.source == "box" else "🔍"
                console.print(f"  {source_icon} {meal.title}")

        # Show metrics
        console.print(f"\n[dim]Metrics:[/dim]")
        console.print(f"  [dim]Unique ingredients: {metrics.unique_ingredients}[/dim]")
        console.print(f"  [dim]Ingredient overlap: {metrics.overlap_ratio:.0%}[/dim]")
        console.print(f"  [dim]From Recipe Box: {metrics.box_recipe_count}, Discovery: {metrics.discovery_recipe_count}[/dim]")

        if metrics.top_shared_ingredients:
            shared = ", ".join(f"{ing}" for ing, _ in metrics.top_shared_ingredients[:5])
            console.print(f"  [dim]Common ingredients: {shared}[/dim]")

        console.print(f"\n[dim]Use /plan to view, /grocery for shopping list[/dim]")

    except Exception as e:
        console.print(f"[red]Error generating meal plan: {e}[/red]")
        logger.exception("Meal plan generation error")


def display_current_meal_plan(meal_plan_store, settings, console: Console):
    """Display the current/most recent meal plan.

    Args:
        meal_plan_store: MealPlanStore instance
        settings: Application settings
        console: Rich console for output
    """
    from src.ingest.build_db import get_recipe_by_id

    # Get most recent active or draft plan
    plans = meal_plan_store.get_plans(limit=1)
    if not plans:
        console.print("[yellow]No meal plans found. Use /mealplan to create one.[/yellow]")
        return

    plan = plans[0]

    console.print(Panel.fit(
        f"[bold cyan]Meal Plan[/bold cyan] (ID: {plan.id})\n"
        f"Status: {plan.status}\n"
        f"Period: {plan.start_date} to {plan.end_date}",
        border_style="cyan"
    ))

    if not plan.meals:
        console.print("[yellow]Plan has no meals yet.[/yellow]")
        return

    # Group meals by day
    from collections import defaultdict
    meals_by_day = defaultdict(list)
    for meal in plan.meals:
        meals_by_day[meal.day].append(meal)

    for day in sorted(meals_by_day.keys()):
        day_meals = meals_by_day[day]
        day_str = day.strftime("%A, %b %d")
        console.print(f"\n[bold]{day_str}[/bold]")
        for meal in day_meals:
            source_icon = "📦" if meal.source == "box" else "🔍"
            console.print(f"  {source_icon} {meal.title}")

    if plan.metrics:
        console.print(f"\n[dim]Metrics: {plan.metrics.unique_ingredients} ingredients, "
                      f"{plan.metrics.overlap_ratio:.0%} overlap[/dim]")


def generate_and_display_grocery_list(meal_plan_store, settings, console: Console):
    """Generate and display grocery list for current meal plan.

    Args:
        meal_plan_store: MealPlanStore instance
        settings: Application settings
        console: Rich console for output
    """
    from src.planning.grocery_list import GroceryListGenerator
    from src.ingest.build_db import get_recipe_by_id

    # Get most recent plan
    plans = meal_plan_store.get_plans(limit=1)
    if not plans:
        console.print("[yellow]No meal plans found. Use /mealplan to create one first.[/yellow]")
        return

    plan = plans[0]
    if not plan.meals:
        console.print("[yellow]Your meal plan has no meals yet.[/yellow]")
        return

    # Fetch full recipes
    recipes = {}
    for meal in plan.meals:
        recipe = get_recipe_by_id(settings.sqlite_db_path, meal.recipe_id)
        if recipe:
            recipes[meal.recipe_id] = recipe

    if not recipes:
        console.print("[yellow]Could not load recipes for the meal plan.[/yellow]")
        return

    # Generate grocery list
    generator = GroceryListGenerator()
    grocery_list = generator.generate(plan, recipes, exclude_pantry_staples=True)

    if not grocery_list.items:
        console.print("[yellow]No items in grocery list (all ingredients may be pantry staples).[/yellow]")
        return

    # Display using generator's format method
    formatted = generator.format_for_display(grocery_list, show_recipes=True, group_by_category=True)
    console.print(Panel.fit(
        f"[bold cyan]Grocery List[/bold cyan]\n"
        f"For meal plan {plan.start_date} to {plan.end_date}\n\n"
        f"{formatted}",
        border_style="green"
    ))

    # Show summary
    summary = generator.get_summary(grocery_list)
    total_items = sum(summary.values())
    console.print(f"\n[dim]Total: {total_items} items across {len(summary)} categories[/dim]")


async def async_chat_session():
    """Async chat session with LLM integration."""
    from pathlib import Path
    from langchain_ollama import ChatOllama
    from src.retrieval.retriever import RecipeRetriever
    from src.retrieval.rerank import RecipeReranker
    from src.retrieval.recipe_cards import RecipeCardBuilder
    from src.chains.retrieval import RetrievalRunnable
    from src.chains.chat_chain import build_chat_chain
    from src.chains.intent_classifier import classify_intent
    from src.memory import ProfileStore, SessionStore, RollingSummarizer, FeedbackStore, HistoryStore, RecipeBoxStore
    from src.domain.models import RecipeFeedback
    from src.ingest.build_db import get_recipe_by_id

    console.print(Panel.fit(
        "[bold cyan]Recipe Assistant[/bold cyan]\n"
        f"Local recipe recommendation powered by RAG + {settings.ollama_model}\n"
        "[dim](See README for Modelfile setup to optimize behavior)[/dim]\n\n"
        "Type /commands for full list. Key commands:\n"
        "  /like <ref>    - Like a recipe\n"
        "  /show <ref>    - Show full recipe details\n"
        "  /save <ref>    - Save to Recipe Box\n"
        "  /box           - View saved recipes\n"
        "  /mealplan      - Plan meals for the week\n"
        "  /grocery       - Generate grocery list\n"
        "  /commands      - Show all commands\n"
        "  quit           - Exit the chat\n\n"
        "[dim]Tip: You can also say 'plan my dinners' or 'help me plan meals'[/dim]",
        border_style="cyan"
    ))

    logger.info("Starting chat session")

    try:
        # Initialize components
        console.print("[dim]Initializing LLM and retrieval components...[/dim]")

        # Initialize LLMs
        # Main LLM for recommendations - fast, direct responses
        llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            num_predict=settings.llm_max_tokens,
            reasoning=False,  # Fast mode: no thinking for recipe presentation
        )

        # LLM for clarification - thoughtful, uses reasoning for better questions
        llm_clarification = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            num_predict=settings.llm_max_tokens * 2,  # Extra budget for thinking + response
            reasoning=True,  # Enable thinking for crafting better clarification questions
        )

        # Separate LLM for intent classification
        # Lower temperature for more deterministic classification
        intent_llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_intent_model,
            temperature=0.2,
            num_predict=256,  # Intent classification needs fewer tokens
            reasoning=False,  # Fast mode for simple classification
        )

        # Initialize retrieval components
        retriever = RecipeRetriever(
            chroma_dir=Path(settings.chroma_persist_dir),
            embedding_model=settings.embedding_model
        )

        reranker = RecipeReranker(model_name=settings.reranker_model)

        card_builder = RecipeCardBuilder(db_path=settings.sqlite_db_path)

        retrieval_chain = RetrievalRunnable(
            retriever=retriever,
            reranker=reranker,
            card_builder=card_builder,
            settings=settings
        )

        # Initialize memory stores
        profile_store = ProfileStore(db_path=settings.sqlite_db_path)
        session_store = SessionStore(db_path=settings.sqlite_db_path)
        feedback_store = FeedbackStore(db_path=settings.sqlite_db_path)
        history_store = HistoryStore(db_path=settings.sqlite_db_path)
        recipe_box_store = RecipeBoxStore(db_path=settings.sqlite_db_path)
        summarizer = RollingSummarizer()

        # Initialize meal plan store
        from src.memory.meal_plan_store import MealPlanStore
        meal_plan_store = MealPlanStore(db_path=settings.sqlite_db_path)

        # Load profile and session
        profile = profile_store.load()
        session_id, session = session_store.get_or_create_current()
        rolling_summary = session_store.get_summary(session_id)

        # Track last recommended cards for feedback commands
        last_recommended_cards = []

        console.print("[green]Ready![/green]\n")

        logger.info("Chat components initialized", session_id=session_id)

    except Exception as e:
        console.print(f"[red]ERROR: Failed to initialize chat: {e}[/red]")
        if "Connection" in str(e) or "connect" in str(e).lower():
            console.print("[yellow]Is Ollama running? Start with: ollama serve[/yellow]")
        logger.exception("Chat initialization error")
        raise typer.Exit(1)

    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = console.input("\n[bold green]You:[/bold green] ")

            # Check for commands
            if user_input.strip().lower() in ("quit", "exit"):
                console.print("[yellow]Goodbye![/yellow]")
                logger.info("Chat session ended by user")
                break

            # /commands - show all available commands
            if user_input.strip().lower() in ("/commands", "/help"):
                console.print(Panel.fit(
                    "[bold cyan]Available Commands[/bold cyan]\n\n"
                    "[bold]Session:[/bold]\n"
                    "  /new             - Start a new session\n"
                    "  /prefs           - Show your preferences\n"
                    "  /addpref <type> <value> - Add a preference\n"
                    "  /commands        - Show this help\n\n"
                    "[bold]Recipe Feedback:[/bold]\n"
                    "  /like <ref>      - Like a recipe (by number or name)\n"
                    "  /dislike <ref>   - Dislike a recipe\n"
                    "  /rate <1-5> <ref> - Rate a recipe 1-5 stars\n"
                    "  /cooked <ref>    - Mark recipe as cooked\n\n"
                    "[bold]Recipe Box:[/bold]\n"
                    "  /save <ref>      - Save recipe to Recipe Box\n"
                    "  /unsave <ref>    - Remove from Recipe Box\n"
                    "  /box             - View saved recipes\n"
                    "  /show <ref>      - Show full recipe details\n"
                    "  /show box <N>    - Show recipe N from Recipe Box\n\n"
                    "[bold]Meal Planning:[/bold]\n"
                    "  /mealplan        - Plan meals for the week\n"
                    "  /mealplan <constraints> - Plan with specific constraints\n"
                    "  /plan            - View current meal plan\n"
                    "  /grocery         - Generate grocery list\n\n"
                    "[bold]History:[/bold]\n"
                    "  /history         - Show cooking history\n\n"
                    "[bold]Preference Types for /addpref:[/bold]\n"
                    "  cuisine <name>   - Add preferred cuisine (italian, mexican, asian...)\n"
                    "  avoid <ingredient> - Add ingredient to avoid\n"
                    "  diet <type>      - Set diet (none, vegetarian, vegan, keto...)\n"
                    "  spice <level>    - Set spice level (none, mild, medium, hot)\n"
                    "  time <minutes>   - Set default cooking time limit\n\n"
                    "[dim]Tip: You can also use natural language like 'plan my dinners' or 'I loved that one'[/dim]",
                    border_style="cyan"
                ))
                continue

            # Try natural language intent classification (skip for explicit slash commands)
            if not user_input.strip().startswith("/"):
                try:
                    intent_result = classify_intent(user_input, last_recommended_cards, intent_llm)

                    # Execute intent if detected (and not conversation)
                    if execute_intent(
                        intent_result,
                        last_recommended_cards,
                        feedback_store,
                        history_store,
                        recipe_box_store,
                        profile_store,
                        session_store,
                        session_id,
                        settings,
                        console
                    ):
                        # Intent was executed, continue to next input
                        continue

                except Exception as e:
                    logger.warning("Intent classification failed, falling back to chat", error=str(e))
                    # Fall through to normal chat processing

            if user_input.strip().lower() == "/new":
                session_id = session_store.create()
                session = session_store.get(session_id)
                rolling_summary = ""
                console.print("[green]✓ Started new session[/green]")
                logger.info("New session created", session_id=session_id)
                continue

            if user_input.strip().lower() == "/prefs":
                console.print(f"\n[bold]Your Preferences:[/bold]")
                console.print(f"  Spice level: {profile.spice_level}")
                console.print(f"  Diet: {profile.diet}")
                if profile.avoid_ingredients:
                    console.print(f"  Avoid: {', '.join(profile.avoid_ingredients)}")
                if profile.preferred_cuisines:
                    console.print(f"  Cuisines: {', '.join(profile.preferred_cuisines)}")
                if profile.time_limit_default_minutes:
                    console.print(f"  Default time: {profile.time_limit_default_minutes} minutes")
                # Show learned preferences
                learned_cuisines = feedback_store.get_preferred_cuisines_from_likes(min_count=3)
                if learned_cuisines:
                    console.print(f"\n[dim]Learned from your likes ({len(learned_cuisines)} cuisines):[/dim]")
                    console.print(f"  [dim]{', '.join(learned_cuisines)}[/dim]")
                continue

            # /addpref command - manually set preferences
            if user_input.strip().lower().startswith("/addpref"):
                parts = user_input[8:].strip().split(maxsplit=1)
                if len(parts) < 2:
                    console.print("[yellow]Usage: /addpref <type> <value>[/yellow]")
                    console.print("[dim]Types: cuisine, avoid, diet, spice, time[/dim]")
                    continue
                pref_type, value = parts[0].lower(), parts[1]

                if pref_type == "cuisine":
                    current = list(profile.preferred_cuisines)
                    if value.lower() not in [c.lower() for c in current]:
                        current.append(value.lower())
                        profile_store.update(preferred_cuisines=current)
                        profile = profile_store.load()  # Reload profile
                        console.print(f"[green]✓ Added cuisine preference: {value}[/green]")
                    else:
                        console.print(f"[yellow]Already in preferences: {value}[/yellow]")

                elif pref_type == "avoid":
                    current = list(profile.avoid_ingredients)
                    if value.lower() not in [i.lower() for i in current]:
                        current.append(value.lower())
                        profile_store.update(avoid_ingredients=current)
                        profile = profile_store.load()
                        console.print(f"[green]✓ Will avoid: {value}[/green]")
                    else:
                        console.print(f"[yellow]Already avoiding: {value}[/yellow]")

                elif pref_type == "diet":
                    valid_diets = ["none", "vegetarian", "vegan", "pescatarian", "keto", "gluten_free"]
                    if value.lower() in valid_diets:
                        profile_store.update(diet=value.lower())
                        profile = profile_store.load()
                        console.print(f"[green]✓ Set diet: {value}[/green]")
                    else:
                        console.print(f"[yellow]Invalid diet. Options: {', '.join(valid_diets)}[/yellow]")

                elif pref_type == "spice":
                    valid_spice = ["none", "mild", "medium", "hot"]
                    if value.lower() in valid_spice:
                        profile_store.update(spice_level=value.lower())
                        profile = profile_store.load()
                        console.print(f"[green]✓ Set spice level: {value}[/green]")
                    else:
                        console.print(f"[yellow]Invalid spice level. Options: {', '.join(valid_spice)}[/yellow]")

                elif pref_type == "time":
                    try:
                        minutes = int(value)
                        if minutes > 0:
                            profile_store.update(time_limit_default_minutes=minutes)
                            profile = profile_store.load()
                            console.print(f"[green]✓ Set default time limit: {minutes} minutes[/green]")
                        else:
                            console.print("[yellow]Time must be positive[/yellow]")
                    except ValueError:
                        console.print("[yellow]Time must be a number (minutes)[/yellow]")

                else:
                    console.print(f"[yellow]Unknown preference type: {pref_type}[/yellow]")
                    console.print("[dim]Types: cuisine, avoid, diet, spice, time[/dim]")
                continue

            # /like command
            if user_input.strip().lower().startswith("/like"):
                ref = user_input[5:].strip()
                if not ref:
                    console.print("[yellow]Usage: /like <number or recipe name>[/yellow]")
                    continue
                result = resolve_recipe_reference(ref, last_recommended_cards)
                if result:
                    recipe_id, title = result
                    feedback_store.add_feedback(RecipeFeedback(
                        recipe_id=recipe_id,
                        feedback_type="like",
                        session_id=session_id
                    ))
                    console.print(f"[green]✓ Liked: {title}[/green]")
                    # Check for learned preferences after liking
                    if update_learned_preferences(feedback_store, profile_store, console):
                        profile = profile_store.load()  # Reload profile with new preferences
                else:
                    console.print(f"[yellow]Recipe not found: {ref}[/yellow]")
                continue

            # /dislike command
            if user_input.strip().lower().startswith("/dislike"):
                ref = user_input[8:].strip()
                if not ref:
                    console.print("[yellow]Usage: /dislike <number or recipe name>[/yellow]")
                    continue
                result = resolve_recipe_reference(ref, last_recommended_cards)
                if result:
                    recipe_id, title = result
                    feedback_store.add_feedback(RecipeFeedback(
                        recipe_id=recipe_id,
                        feedback_type="dislike",
                        session_id=session_id
                    ))
                    console.print(f"[yellow]✓ Disliked: {title}[/yellow]")
                else:
                    console.print(f"[yellow]Recipe not found: {ref}[/yellow]")
                continue

            # /rate command
            if user_input.strip().lower().startswith("/rate"):
                parts = user_input[5:].strip().split(maxsplit=1)
                if len(parts) < 2:
                    console.print("[yellow]Usage: /rate <1-5> <number or recipe name>[/yellow]")
                    continue
                try:
                    rating = int(parts[0])
                    if not 1 <= rating <= 5:
                        console.print("[yellow]Rating must be between 1 and 5[/yellow]")
                        continue
                    ref = parts[1]
                    result = resolve_recipe_reference(ref, last_recommended_cards)
                    if result:
                        recipe_id, title = result
                        feedback_store.add_feedback(RecipeFeedback(
                            recipe_id=recipe_id,
                            feedback_type="rate",
                            rating=rating,
                            session_id=session_id
                        ))
                        console.print(f"[green]✓ Rated {title}: {rating}/5[/green]")
                    else:
                        console.print(f"[yellow]Recipe not found: {ref}[/yellow]")
                except ValueError:
                    console.print("[yellow]Invalid rating. Must be a number 1-5[/yellow]")
                continue

            # /show command - supports both recommendations and Recipe Box
            if user_input.strip().lower().startswith("/show"):
                ref = user_input[5:].strip()
                if not ref:
                    console.print("[yellow]Usage: /show <number or recipe name>[/yellow]")
                    console.print("[dim]  /show box <N> - Show recipe N from Recipe Box[/dim]")
                    continue

                # Check for Recipe Box reference: "box 1", "box 2", etc.
                if ref.lower().startswith("box"):
                    box_ref = ref[3:].strip()
                    saved_recipes = recipe_box_store.get_saved_recipes(limit=50)
                    if not saved_recipes:
                        console.print("[yellow]Recipe Box is empty[/yellow]")
                        continue
                    try:
                        box_num = int(box_ref)
                        if 1 <= box_num <= len(saved_recipes):
                            saved = saved_recipes[box_num - 1]
                            recipe = get_recipe_by_id(settings.sqlite_db_path, saved.recipe_id)
                            if recipe:
                                display_full_recipe(recipe, console)
                            else:
                                console.print(f"[yellow]Recipe not found in database: {saved.recipe_id}[/yellow]")
                        else:
                            console.print(f"[yellow]Invalid box number. You have {len(saved_recipes)} saved recipes.[/yellow]")
                    except ValueError:
                        # Try matching by name in Recipe Box
                        found = False
                        for saved in saved_recipes:
                            if box_ref.lower() in saved.title.lower():
                                recipe = get_recipe_by_id(settings.sqlite_db_path, saved.recipe_id)
                                if recipe:
                                    display_full_recipe(recipe, console)
                                    found = True
                                    break
                        if not found:
                            console.print(f"[yellow]Recipe not found in box: {box_ref}[/yellow]")
                    continue

                # Standard resolution from last recommendations
                result = resolve_recipe_reference(ref, last_recommended_cards)
                if result:
                    recipe_id, title = result
                    recipe = get_recipe_by_id(settings.sqlite_db_path, recipe_id)
                    if recipe:
                        display_full_recipe(recipe, console)
                    else:
                        console.print(f"[yellow]Recipe not found in database: {recipe_id}[/yellow]")
                else:
                    # Also try Recipe Box as fallback
                    saved_recipes = recipe_box_store.get_saved_recipes(limit=50)
                    found = False
                    for saved in saved_recipes:
                        if ref.lower() in saved.title.lower():
                            recipe = get_recipe_by_id(settings.sqlite_db_path, saved.recipe_id)
                            if recipe:
                                display_full_recipe(recipe, console)
                                found = True
                                break
                    if not found:
                        console.print(f"[yellow]Recipe not found: {ref}[/yellow]")
                        console.print("[dim]Tip: Use /show box <N> for recipes from your Recipe Box[/dim]")
                continue

            # /cooked command
            if user_input.strip().lower().startswith("/cooked"):
                ref = user_input[7:].strip()
                if not ref:
                    console.print("[yellow]Usage: /cooked <number or recipe name>[/yellow]")
                    continue
                result = resolve_recipe_reference(ref, last_recommended_cards)
                if result:
                    recipe_id, title = result
                    history_store.add_cooked(recipe_id)
                    console.print(f"[green]✓ Marked as cooked: {title}[/green]")
                else:
                    console.print(f"[yellow]Recipe not found: {ref}[/yellow]")
                continue

            # /history command
            if user_input.strip().lower() == "/history":
                history = history_store.get_cooking_history(limit=10)
                if history:
                    console.print(f"\n[bold]Recent Cooking History:[/bold]")
                    for i, entry in enumerate(history, 1):
                        # Get recipe title
                        recipe = get_recipe_by_id(settings.sqlite_db_path, entry.recipe_id)
                        title = recipe.title if recipe else entry.recipe_id
                        cooked_str = entry.cooked_at.strftime("%Y-%m-%d") if entry.cooked_at else "Unknown"
                        console.print(f"  {i}. {title} (cooked: {cooked_str})")
                else:
                    console.print("[yellow]No cooking history yet[/yellow]")
                continue

            # /save command
            if user_input.strip().lower().startswith("/save"):
                ref = user_input[5:].strip()
                if not ref:
                    console.print("[yellow]Usage: /save <number or recipe name>[/yellow]")
                    continue
                result = resolve_recipe_reference(ref, last_recommended_cards)
                if result:
                    recipe_id, title = result
                    try:
                        recipe_box_store.save_recipe(recipe_id, title)
                        console.print(f"[green]✓ Saved to Recipe Box: {title}[/green]")
                    except Exception as e:
                        if "UNIQUE" in str(e):
                            console.print(f"[yellow]Already saved: {title}[/yellow]")
                        else:
                            console.print(f"[red]Error saving recipe: {e}[/red]")
                else:
                    console.print(f"[yellow]Recipe not found: {ref}[/yellow]")
                continue

            # /unsave command
            if user_input.strip().lower().startswith("/unsave"):
                ref = user_input[7:].strip()
                if not ref:
                    console.print("[yellow]Usage: /unsave <number or recipe name>[/yellow]")
                    continue
                result = resolve_recipe_reference(ref, last_recommended_cards)
                if result:
                    recipe_id, title = result
                    if recipe_box_store.remove_recipe(recipe_id):
                        console.print(f"[green]✓ Removed from Recipe Box: {title}[/green]")
                    else:
                        console.print(f"[yellow]Recipe not found in box: {title}[/yellow]")
                else:
                    console.print(f"[yellow]Recipe not found: {ref}[/yellow]")
                continue

            # /box command
            if user_input.strip().lower() == "/box":
                saved_recipes = recipe_box_store.get_saved_recipes(limit=50)
                if saved_recipes:
                    console.print(f"\n[bold]Recipe Box ({len(saved_recipes)} saved):[/bold]")
                    for i, saved in enumerate(saved_recipes, 1):
                        saved_str = saved.saved_at.strftime("%Y-%m-%d") if saved.saved_at else "Unknown"
                        console.print(f"  {i}. {saved.title} (saved: {saved_str})")
                else:
                    console.print("[yellow]Recipe Box is empty. Use /save <recipe> to add recipes![/yellow]")
                continue

            # /mealplan command - start meal planning
            if user_input.strip().lower().startswith("/mealplan"):
                input_text = user_input[9:].strip()  # Get text after /mealplan
                await handle_mealplan_command(
                    input_text=input_text,
                    profile=profile,
                    recipe_box_store=recipe_box_store,
                    meal_plan_store=meal_plan_store,
                    settings=settings,
                    console=console
                )
                continue

            # /plan command - show current meal plan
            if user_input.strip().lower() in ("/plan", "/showplan"):
                display_current_meal_plan(meal_plan_store, settings, console)
                continue

            # /grocery command - generate grocery list
            if user_input.strip().lower() in ("/grocery", "/groceries"):
                generate_and_display_grocery_list(meal_plan_store, settings, console)
                continue

            # Skip empty input
            if not user_input.strip():
                continue

            # Compute exclusion set from feedback and history
            exclude_ids = (
                feedback_store.get_liked_recipe_ids(limit=20) |
                feedback_store.get_disliked_recipe_ids() |
                history_store.get_recently_cooked_ids(days=7)
            )

            # Build chain with current context
            chain = build_chat_chain(
                llm=llm,
                retrieval_chain=retrieval_chain,
                profile=profile,
                session=session,
                rolling_summary=rolling_summary,
                exclude_recipe_ids=exclude_ids,
                llm_clarification=llm_clarification,  # Thoughtful LLM for clarification questions
            )

            # Invoke chain
            console.print("\n[dim]Thinking...[/dim]")

            result = await chain.ainvoke({"user_input": user_input})

            # Extract response and cards from chain result
            response = result.get("response", "")
            cards = result.get("cards", [])

            # Display response
            console.print(f"\n[bold blue]A:[/bold blue] {response}")

            # Update last recommended cards for feedback commands
            if cards:
                last_recommended_cards = cards

            # Re-extract constraints for rolling summary update
            from src.chains.extractors import ConstraintExtractor
            extractor = ConstraintExtractor()
            constraints = extractor.extract_constraints(user_input)

            # Update rolling summary
            rolling_summary = summarizer.update_summary(rolling_summary, constraints, user_input)
            session_store.update_summary(session_id, rolling_summary)

            logger.info("Processed user turn", input_length=len(user_input), response_length=len(response))

        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            logger.info("Chat session interrupted")
            break
        except Exception as e:
            console.print(f"\n[red]Error:[/red] {e}")
            if "Connection" in str(e) or "connect" in str(e).lower():
                console.print("[yellow]Lost connection to Ollama. Is it still running?[/yellow]")
            logger.exception("Chat error")


@app.command()
def version():
    """Show version information."""
    console.print(f"Recipe Assistant version [cyan]{__version__}[/cyan]")
    logger.info("Version command executed", version=__version__)


@app.command()
def config():
    """Show current configuration settings."""
    console.print("[bold]Current Configuration:[/bold]\n")
    console.print(f"Ollama URL: [cyan]{settings.ollama_base_url}[/cyan]")
    console.print(f"Ollama Model: [cyan]{settings.ollama_model}[/cyan]")
    console.print(f"Embedding Model: [cyan]{settings.embedding_model}[/cyan]")
    console.print(f"Reranker Model: [cyan]{settings.reranker_model}[/cyan]")
    console.print(f"Chroma Dir: [cyan]{settings.chroma_persist_dir}[/cyan]")
    console.print(f"SQLite DB: [cyan]{settings.sqlite_db_path}[/cyan]")
    console.print(f"k_retrieve: [cyan]{settings.k_retrieve}[/cyan]")
    console.print(f"k_rerank: [cyan]{settings.k_rerank}[/cyan]")
    console.print(f"k_context: [cyan]{settings.k_context}[/cyan]")
    console.print(f"Log Level: [cyan]{settings.log_level}[/cyan]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query (e.g., 'chicken tomato spicy')"),
    k: int = typer.Option(10, help="Number of results to return"),
    rerank: bool = typer.Option(False, "--rerank", "-r", help="Enable cross-encoder reranking"),
    cards: bool = typer.Option(False, "--cards", "-c", help="Show recipe cards (implies --rerank)")
):
    """Search for recipes using vector similarity with optional reranking and card display."""
    import time
    from pathlib import Path
    from src.retrieval.retriever import RecipeRetriever
    from src.retrieval.rerank import RecipeReranker
    from src.retrieval.recipe_cards import RecipeCardBuilder
    from rich.panel import Panel

    chroma_dir = Path(settings.chroma_persist_dir)

    # Check if vector store exists
    if not chroma_dir.exists():
        console.print(f"[red]ERROR Vector store not found at: {chroma_dir}[/red]")
        console.print("[yellow]Run 'ingest embed' first to build the vector store[/yellow]")
        raise typer.Exit(1)

    # Enable rerank if cards requested
    if cards:
        rerank = True

    mode = "cards" if cards else "rerank" if rerank else "vector"
    console.print(f"\n[cyan]Searching for:[/cyan] {query}")
    console.print(f"[dim]Mode: {mode} | Retrieving top {k} results...[/dim]\n")

    try:
        # Initialize retriever
        retriever = RecipeRetriever(
            chroma_dir=chroma_dir,
            embedding_model=settings.embedding_model
        )

        # Perform search
        start_time = time.time()

        # Step 1: Vector retrieval
        k_retrieve = settings.k_retrieve if rerank else k
        results = retriever.search(query, k=k_retrieve)

        # Step 2: Rerank (optional)
        if rerank and results:
            reranker = RecipeReranker(model_name=settings.reranker_model)
            results = reranker.rerank(query, results, top_k=settings.k_rerank)

        # Limit to k results if not using cards
        if not cards:
            results = results[:k]

        elapsed_ms = (time.time() - start_time) * 1000

        # Display results
        if not results:
            console.print("[yellow]No results found.[/yellow]")
        elif cards:
            # Step 3: Build and display cards
            builder = RecipeCardBuilder(db_path=settings.sqlite_db_path)
            recipe_cards = builder.build_cards(results[:settings.k_context], query)

            console.print(f"[bold]Found {len(recipe_cards)} recipe cards:[/bold]\n")

            for i, card in enumerate(recipe_cards, 1):
                # Build card content
                content = []

                # Rating
                if card.rating_avg:
                    stars = "★" * int(card.rating_avg) + "☆" * (5 - int(card.rating_avg))
                    rating_count = f"({card.rating_count} reviews)" if card.rating_count else ""
                    content.append(f"[yellow]{stars}[/yellow] {card.rating_avg:.1f}/5 {rating_count}")

                # Time
                if card.time_total:
                    content.append(f"[cyan]⏱ {card.time_total} minutes[/cyan]")

                # Tags
                if card.tags:
                    content.append(f"[dim]Tags: {', '.join(card.tags[:8])}[/dim]")

                # Key ingredients
                if card.key_ingredients:
                    ing_list = ', '.join(card.key_ingredients[:10])
                    content.append(f"[green]Key ingredients:[/green] {ing_list}")

                # Summary
                if card.one_sentence_summary:
                    content.append(f"\n{card.one_sentence_summary}")

                # Why match
                if card.why_match:
                    content.append(f"\n[bold cyan]Why this matches:[/bold cyan] {card.why_match}")

                console.print(Panel(
                    "\n".join(content),
                    title=f"[bold]{i}. {card.title}[/bold]",
                    border_style="blue"
                ))
        else:
            # Simple result list
            console.print(f"[bold]Found {len(results)} recipes in {elapsed_ms:.0f}ms:[/bold]\n")

            for i, result in enumerate(results, 1):
                rating_str = f"{result.rating_avg:.1f}" if result.rating_avg else "N/A"
                rating_count_str = f"({result.rating_count})" if result.rating_count else ""
                time_str = f"{result.minutes}m" if result.minutes else "N/A"

                console.print(
                    f"{i:2}. [bold]{result.title}[/bold] "
                    f"[dim]| Score: {result.score:.3f} | "
                    f"Rating: {rating_str} {rating_count_str} | "
                    f"Time: {time_str}[/dim]"
                )

        # Timing summary
        console.print(f"\n[dim]Completed in {elapsed_ms:.0f}ms[/dim]")

        # Warn if search is slow
        if elapsed_ms > 1000:
            console.print(f"[yellow]WARNING: Search took {elapsed_ms:.0f}ms (target <1000ms)[/yellow]")

        logger.info("Search completed", query=query, mode=mode, results_count=len(results), time_ms=elapsed_ms)

    except Exception as e:
        console.print(f"[red]ERROR Search failed: {e}[/red]")
        logger.exception("Search error")
        raise typer.Exit(1)


def main():
    """Entry point for the CLI application."""
    app()


if __name__ == "__main__":
    main()

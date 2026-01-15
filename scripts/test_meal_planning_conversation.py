#!/usr/bin/env python
"""Comprehensive conversational tests for meal planning feature.

Tests natural language meal planning, grocery list generation, and plan viewing.
Requires Ollama to be running with the configured model.

Usage:
    python scripts/test_meal_planning_conversation.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from datetime import date, timedelta

from src.app.settings import settings
from src.chains.intent_classifier import check_quick_intent, classify_intent, build_recommendations_context
from src.domain.models import (
    Recipe, RecipeCard, PreferenceProfile, MealPlanConstraints,
    DietaryRestriction, IntentClassification, MealPlan
)
from src.planning.constraint_extractor import MealPlanConstraintExtractor
from src.planning.meal_planner import MealPlanner
from src.planning.grocery_list import GroceryListGenerator
from src.planning.ingredient_normalizer import IngredientNormalizer
from src.memory.meal_plan_store import MealPlanStore

console = Console()


def create_test_recipes() -> list[Recipe]:
    """Create a diverse set of test recipes for meal planning."""
    return [
        Recipe(
            recipe_id="r1",
            title="Chicken Stir Fry",
            ingredients=["chicken breast", "soy sauce", "garlic", "bell pepper", "broccoli", "ginger"],
            ingredients_normalized=["chicken", "soy sauce", "garlic", "bell pepper", "broccoli", "ginger"],
            instructions=["Cut chicken", "Stir fry vegetables", "Add sauce"],
            tags=["asian", "quick", "healthy"],
            minutes=25,
        ),
        Recipe(
            recipe_id="r2",
            title="Vegetarian Pasta Primavera",
            ingredients=["pasta", "zucchini", "tomatoes", "garlic", "olive oil", "parmesan"],
            ingredients_normalized=["pasta", "zucchini", "tomato", "garlic", "olive oil", "parmesan"],
            instructions=["Cook pasta", "Saute vegetables", "Combine"],
            tags=["italian", "vegetarian"],
            minutes=30,
        ),
        Recipe(
            recipe_id="r3",
            title="Beef Tacos",
            ingredients=["ground beef", "taco seasoning", "tortillas", "lettuce", "tomatoes", "cheese"],
            ingredients_normalized=["beef", "taco seasoning", "tortilla", "lettuce", "tomato", "cheese"],
            instructions=["Brown beef", "Season", "Assemble tacos"],
            tags=["mexican", "quick"],
            minutes=20,
        ),
        Recipe(
            recipe_id="r4",
            title="Salmon with Vegetables",
            ingredients=["salmon fillet", "asparagus", "lemon", "garlic", "olive oil"],
            ingredients_normalized=["salmon", "asparagus", "lemon", "garlic", "olive oil"],
            instructions=["Season salmon", "Roast with vegetables"],
            tags=["healthy", "seafood"],
            minutes=35,
        ),
        Recipe(
            recipe_id="r5",
            title="Vegetable Curry",
            ingredients=["chickpeas", "coconut milk", "curry powder", "spinach", "onion", "garlic"],
            ingredients_normalized=["chickpea", "coconut milk", "curry powder", "spinach", "onion", "garlic"],
            instructions=["Saute onions", "Add spices", "Simmer with coconut milk"],
            tags=["indian", "vegan", "vegetarian"],
            minutes=40,
        ),
        Recipe(
            recipe_id="r6",
            title="Garlic Butter Shrimp",
            ingredients=["shrimp", "butter", "garlic", "lemon", "parsley"],
            ingredients_normalized=["shrimp", "butter", "garlic", "lemon", "parsley"],
            instructions=["Saute garlic in butter", "Add shrimp", "Finish with lemon"],
            tags=["seafood", "quick"],
            minutes=15,
        ),
        Recipe(
            recipe_id="r7",
            title="Greek Salad",
            ingredients=["cucumber", "tomatoes", "feta cheese", "olives", "red onion", "olive oil"],
            ingredients_normalized=["cucumber", "tomato", "feta", "olive", "red onion", "olive oil"],
            instructions=["Chop vegetables", "Add feta and olives", "Dress with olive oil"],
            tags=["greek", "vegetarian", "healthy", "quick"],
            minutes=10,
        ),
        Recipe(
            recipe_id="r8",
            title="Chicken Alfredo",
            ingredients=["chicken breast", "fettuccine", "heavy cream", "parmesan", "garlic", "butter"],
            ingredients_normalized=["chicken", "fettuccine", "heavy cream", "parmesan", "garlic", "butter"],
            instructions=["Cook pasta", "Make alfredo sauce", "Combine with chicken"],
            tags=["italian", "comfort-food"],
            minutes=35,
        ),
    ]


def test_quick_intent_patterns():
    """Test quick intent pattern matching for meal planning."""
    console.print("\n[bold cyan]Test 1: Quick Intent Pattern Matching[/bold cyan]")

    test_cases = [
        # Meal planning intents
        ("mealplan", "mealplan"),
        ("meal plan", "mealplan"),
        ("plan meals", "mealplan"),
        ("plan my meals", "mealplan"),
        ("plan dinners", "mealplan"),
        ("plan my week", "mealplan"),
        ("weekly plan", "mealplan"),
        ("help me plan meals", "mealplan"),

        # Grocery list intents
        ("grocery", "grocery_list"),
        ("groceries", "grocery_list"),
        ("grocery list", "grocery_list"),
        ("shopping list", "grocery_list"),
        ("what do i need to buy", "grocery_list"),

        # Show plan intents
        ("show plan", "show_plan"),
        ("show my plan", "show_plan"),
        ("my meal plan", "show_plan"),
        ("current plan", "show_plan"),
        ("view meal plan", "show_plan"),

        # Non-matches (should return None)
        ("I want chicken for dinner", None),
        ("show me some recipes", None),
        ("what should I cook", None),
    ]

    results = []
    for input_text, expected in test_cases:
        result = check_quick_intent(input_text)
        passed = result == expected
        results.append((input_text, expected, result, passed))

    # Display results
    table = Table(title="Quick Intent Pattern Tests")
    table.add_column("Input", style="cyan")
    table.add_column("Expected", style="yellow")
    table.add_column("Result", style="green")
    table.add_column("Status", style="bold")

    passed_count = 0
    for input_text, expected, result, passed in results:
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        table.add_row(input_text, str(expected), str(result), status)
        if passed:
            passed_count += 1

    console.print(table)
    console.print(f"\n[bold]Results: {passed_count}/{len(results)} passed[/bold]")
    return passed_count == len(results)


def test_constraint_extraction():
    """Test constraint extraction from natural language."""
    console.print("\n[bold cyan]Test 2: Constraint Extraction from Natural Language[/bold cyan]")

    extractor = MealPlanConstraintExtractor()
    profile = PreferenceProfile()

    test_cases = [
        # Basic day extraction
        ("plan 5 days of dinners", {"days": 5}),
        ("plan 3 days of meals", {"days": 3}),
        ("plan a week of dinners", {"days": 7}),
        ("plan weeknight dinners", {"days": 5}),

        # Dietary extraction
        ("plan vegetarian dinners", {"dietary": DietaryRestriction.VEGETARIAN}),
        ("plan vegan meals for the week", {"dietary": DietaryRestriction.VEGAN}),

        # Time constraints
        ("plan quick dinners under 30 minutes", {"max_prep_time": 30}),
        ("plan meals that take less than 45 minutes", {"max_prep_time": 45}),

        # Combined constraints
        ("plan 5 vegetarian dinners under 30 minutes", {
            "days": 5,
            "dietary": DietaryRestriction.VEGETARIAN,
            "max_prep_time": 30
        }),
    ]

    results = []
    for input_text, expected in test_cases:
        constraints = extractor.extract(input_text, profile)

        all_match = True
        for key, expected_value in expected.items():
            actual_value = getattr(constraints, key)
            if actual_value != expected_value:
                all_match = False
                break

        results.append((input_text, expected, constraints, all_match))

    # Display results
    table = Table(title="Constraint Extraction Tests")
    table.add_column("Input", style="cyan", width=45)
    table.add_column("Expected", style="yellow", width=30)
    table.add_column("Status", style="bold")

    passed_count = 0
    for input_text, expected, constraints, passed in results:
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        expected_str = ", ".join(f"{k}={v}" for k, v in expected.items())
        table.add_row(input_text[:44], expected_str[:29], status)
        if passed:
            passed_count += 1

    console.print(table)
    console.print(f"\n[bold]Results: {passed_count}/{len(results)} passed[/bold]")
    return passed_count == len(results)


def test_meal_plan_generation():
    """Test meal plan generation with various constraints."""
    console.print("\n[bold cyan]Test 3: Meal Plan Generation[/bold cyan]")

    recipes = create_test_recipes()
    planner = MealPlanner()
    profile = PreferenceProfile()

    test_cases = [
        # Basic 3-day plan
        (
            "Basic 3-day plan",
            MealPlanConstraints(days=3),
            lambda meals, metrics: len(meals) == 3
        ),
        # 5-day plan
        (
            "5-day dinner plan",
            MealPlanConstraints(days=5),
            lambda meals, metrics: len(meals) == 5
        ),
        # Vegetarian plan
        (
            "Vegetarian plan",
            MealPlanConstraints(days=3, dietary=DietaryRestriction.VEGETARIAN),
            lambda meals, metrics: len(meals) == 3
        ),
        # Quick meals plan
        (
            "Quick meals (under 30 min)",
            MealPlanConstraints(days=3, max_prep_time=30),
            lambda meals, metrics: len(meals) == 3
        ),
    ]

    results = []
    for name, constraints, validator in test_cases:
        try:
            meals, metrics = planner.generate_plan(
                recipes=recipes,
                constraints=constraints,
                profile=profile,
                box_recipe_ids=set()
            )
            passed = validator(meals, metrics)
            results.append((name, len(meals), metrics.overlap_ratio, passed))
        except Exception as e:
            results.append((name, 0, 0, False))
            console.print(f"[red]Error in {name}: {e}[/red]")

    # Display results
    table = Table(title="Meal Plan Generation Tests")
    table.add_column("Test Case", style="cyan")
    table.add_column("Meals Generated", style="yellow")
    table.add_column("Overlap Ratio", style="green")
    table.add_column("Status", style="bold")

    passed_count = 0
    for name, meal_count, overlap, passed in results:
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        table.add_row(name, str(meal_count), f"{overlap:.2%}", status)
        if passed:
            passed_count += 1

    console.print(table)
    console.print(f"\n[bold]Results: {passed_count}/{len(results)} passed[/bold]")
    return passed_count == len(results)


def test_determinism():
    """Test that meal planning is deterministic."""
    console.print("\n[bold cyan]Test 4: Determinism Test[/bold cyan]")

    recipes = create_test_recipes()
    planner = MealPlanner()
    profile = PreferenceProfile()
    constraints = MealPlanConstraints(days=5)

    # Generate 3 plans with same inputs
    plans = []
    for i in range(3):
        meals, _ = planner.generate_plan(
            recipes=recipes,
            constraints=constraints,
            profile=profile,
            box_recipe_ids=set()
        )
        plans.append([m.recipe_id for m in meals])

    # Check all plans are identical
    all_same = plans[0] == plans[1] == plans[2]

    console.print(f"Plan 1: {plans[0]}")
    console.print(f"Plan 2: {plans[1]}")
    console.print(f"Plan 3: {plans[2]}")

    if all_same:
        console.print("[green]PASS: All plans are identical (deterministic)[/green]")
    else:
        console.print("[red]FAIL: Plans differ (non-deterministic)[/red]")

    return all_same


def test_grocery_list_generation():
    """Test grocery list generation from meal plan."""
    console.print("\n[bold cyan]Test 5: Grocery List Generation[/bold cyan]")

    recipes = create_test_recipes()
    planner = MealPlanner()
    generator = GroceryListGenerator()
    profile = PreferenceProfile()
    constraints = MealPlanConstraints(days=3)

    # Generate a plan
    meals, _ = planner.generate_plan(
        recipes=recipes,
        constraints=constraints,
        profile=profile,
        box_recipe_ids=set()
    )

    # Get full recipe objects for the plan
    recipe_map = {r.recipe_id: r for r in recipes}

    # Create a MealPlan object for grocery list generation
    meal_plan = MealPlan(
        start_date=date.today(),
        end_date=date.today() + timedelta(days=2),
        meals=meals
    )

    # Generate grocery list
    grocery_list = generator.generate(meal_plan, recipe_map)

    console.print(f"\n[bold]Meal Plan ({len(meals)} meals):[/bold]")
    for meal in meals:
        console.print(f"  - {meal.title}")

    console.print(f"\n[bold]Grocery List ({len(grocery_list.items)} items):[/bold]")
    for item in grocery_list.items[:10]:  # Show first 10
        recipes_str = ", ".join(item.recipes[:2])
        if len(item.recipes) > 2:
            recipes_str += f" +{len(item.recipes)-2} more"
        console.print(f"  - {item.ingredient} (for: {recipes_str})")

    if len(grocery_list.items) > 10:
        console.print(f"  ... and {len(grocery_list.items) - 10} more items")

    # Verify pantry staples are excluded
    pantry_staples = {"salt", "pepper", "water", "oil"}
    excluded_correctly = all(
        item.normalized not in pantry_staples
        for item in grocery_list.items
    )

    passed = len(grocery_list.items) > 0 and excluded_correctly

    if passed:
        console.print("\n[green]PASS: Grocery list generated correctly[/green]")
    else:
        console.print("\n[red]FAIL: Grocery list generation failed[/red]")

    return passed


def test_diversity_constraints():
    """Test diversity constraints (max same protein/cuisine)."""
    console.print("\n[bold cyan]Test 6: Diversity Constraints[/bold cyan]")

    # Create recipes with repeated proteins
    recipes = [
        Recipe(
            recipe_id=f"chicken{i}",
            title=f"Chicken Dish {i}",
            ingredients=["chicken", "garlic"],
            ingredients_normalized=["chicken", "garlic"],
            tags=["poultry"],
            minutes=30,
        )
        for i in range(5)
    ] + [
        Recipe(
            recipe_id="beef1",
            title="Beef Stew",
            ingredients=["beef", "potato"],
            ingredients_normalized=["beef", "potato"],
            tags=["beef"],
            minutes=60,
        ),
        Recipe(
            recipe_id="fish1",
            title="Grilled Fish",
            ingredients=["fish", "lemon"],
            ingredients_normalized=["fish", "lemon"],
            tags=["seafood"],
            minutes=20,
        ),
    ]

    planner = MealPlanner()
    profile = PreferenceProfile()
    constraints = MealPlanConstraints(
        days=5,
        max_same_protein=2  # Should not pick more than 2 chicken dishes
    )

    meals, metrics = planner.generate_plan(
        recipes=recipes,
        constraints=constraints,
        profile=profile,
        box_recipe_ids=set()
    )

    # Count proteins
    chicken_count = sum(1 for m in meals if "chicken" in m.recipe_id.lower())

    console.print(f"\n[bold]Generated Plan:[/bold]")
    for meal in meals:
        console.print(f"  - {meal.title}")

    console.print(f"\n[bold]Protein Distribution:[/bold]")
    for protein, count in metrics.protein_distribution.items():
        console.print(f"  - {protein}: {count}")

    passed = chicken_count <= 2

    if passed:
        console.print(f"\n[green]PASS: Chicken dishes limited to {chicken_count} (max 2)[/green]")
    else:
        console.print(f"\n[red]FAIL: Too many chicken dishes ({chicken_count} > 2)[/red]")

    return passed


def test_ingredient_overlap_optimization():
    """Test that planner optimizes for ingredient overlap."""
    console.print("\n[bold cyan]Test 7: Ingredient Overlap Optimization[/bold cyan]")

    # Create recipes with overlapping ingredients
    recipes = [
        Recipe(
            recipe_id="r1",
            title="Garlic Chicken",
            ingredients=["chicken", "garlic", "olive oil"],
            ingredients_normalized=["chicken", "garlic", "olive oil"],
            tags=["quick"],
            minutes=25,
        ),
        Recipe(
            recipe_id="r2",
            title="Garlic Shrimp",
            ingredients=["shrimp", "garlic", "butter", "lemon"],
            ingredients_normalized=["shrimp", "garlic", "butter", "lemon"],
            tags=["seafood"],
            minutes=15,
        ),
        Recipe(
            recipe_id="r3",
            title="Lemon Chicken",
            ingredients=["chicken", "lemon", "herbs"],
            ingredients_normalized=["chicken", "lemon", "herb"],
            tags=["healthy"],
            minutes=30,
        ),
        Recipe(
            recipe_id="r4",
            title="Random Dish",
            ingredients=["tofu", "broccoli", "soy sauce"],
            ingredients_normalized=["tofu", "broccoli", "soy sauce"],
            tags=["vegan"],
            minutes=20,
        ),
    ]

    planner = MealPlanner()
    profile = PreferenceProfile()
    constraints = MealPlanConstraints(days=3, ingredient_overlap_weight=0.5)

    meals, metrics = planner.generate_plan(
        recipes=recipes,
        constraints=constraints,
        profile=profile,
        box_recipe_ids=set()
    )

    console.print(f"\n[bold]Generated Plan:[/bold]")
    for meal in meals:
        console.print(f"  - {meal.title}")

    console.print(f"\n[bold]Metrics:[/bold]")
    console.print(f"  - Unique ingredients: {metrics.unique_ingredients}")
    console.print(f"  - Total ingredient uses: {metrics.total_ingredient_uses}")
    console.print(f"  - Overlap ratio: {metrics.overlap_ratio:.2%}")
    console.print(f"  - Unique per meal: {metrics.unique_per_meal:.1f}")

    console.print(f"\n[bold]Top Shared Ingredients:[/bold]")
    for ing, count in metrics.top_shared_ingredients[:5]:
        console.print(f"  - {ing}: used in {count} recipes")

    # A good plan should have some overlap
    passed = metrics.overlap_ratio > 0

    if passed:
        console.print(f"\n[green]PASS: Plan has ingredient overlap ({metrics.overlap_ratio:.1%})[/green]")
    else:
        console.print(f"\n[red]FAIL: No ingredient overlap in plan[/red]")

    return passed


def test_full_conversation_flow():
    """Test full conversational flow for meal planning."""
    console.print("\n[bold cyan]Test 8: Full Conversation Flow Simulation[/bold cyan]")

    console.print("\n[dim]Simulating conversation:[/dim]")

    # Step 1: User says "plan my meals" (exact match in quick intents)
    user_input = "plan my meals"
    console.print(f"\n[bold blue]User:[/bold blue] {user_input}")

    intent = check_quick_intent(user_input)
    console.print(f"[dim]Intent detected: {intent}[/dim]")

    if intent != "mealplan":
        console.print("[red]FAIL: Should detect mealplan intent[/red]")
        return False

    console.print("[bold green]Assistant:[/bold green] I'll help you plan meals for the week!")

    # Step 2: Extract constraints from "plan 5 vegetarian dinners"
    user_input = "plan 5 vegetarian dinners"
    console.print(f"\n[bold blue]User:[/bold blue] {user_input}")

    extractor = MealPlanConstraintExtractor()
    constraints = extractor.extract(user_input, PreferenceProfile())

    console.print(f"[dim]Extracted: days={constraints.days}, dietary={constraints.dietary}[/dim]")

    # Step 3: Generate plan
    recipes = create_test_recipes()
    planner = MealPlanner()
    meals, metrics = planner.generate_plan(
        recipes=recipes,
        constraints=constraints,
        profile=PreferenceProfile(),
        box_recipe_ids=set()
    )

    console.print(f"\n[bold green]Assistant:[/bold green] Here's your {len(meals)}-day meal plan:")
    for i, meal in enumerate(meals, 1):
        console.print(f"  Day {i}: {meal.title}")

    # Step 4: User asks for grocery list
    user_input = "grocery list"
    console.print(f"\n[bold blue]User:[/bold blue] {user_input}")

    intent = check_quick_intent(user_input)
    if intent != "grocery_list":
        console.print("[red]FAIL: Should detect grocery_list intent[/red]")
        return False

    # Generate grocery list
    recipe_map = {r.recipe_id: r for r in recipes}
    generator = GroceryListGenerator()

    # Create MealPlan object
    meal_plan = MealPlan(
        start_date=date.today(),
        end_date=date.today() + timedelta(days=len(meals) - 1),
        meals=meals
    )
    grocery_list = generator.generate(meal_plan, recipe_map)

    console.print(f"\n[bold green]Assistant:[/bold green] Here's your grocery list ({len(grocery_list.items)} items):")
    for item in grocery_list.items[:5]:
        console.print(f"  - {item.ingredient}")
    if len(grocery_list.items) > 5:
        console.print(f"  ... and {len(grocery_list.items) - 5} more")

    # Step 5: User asks to see the plan
    user_input = "show my plan"
    console.print(f"\n[bold blue]User:[/bold blue] {user_input}")

    intent = check_quick_intent(user_input)
    if intent != "show_plan":
        console.print("[red]FAIL: Should detect show_plan intent[/red]")
        return False

    console.print(f"\n[bold green]Assistant:[/bold green] Your current meal plan:")
    for i, meal in enumerate(meals, 1):
        console.print(f"  Day {i}: {meal.title}")

    console.print("\n[green]PASS: Full conversation flow completed successfully[/green]")
    return True


def main():
    """Run all meal planning conversation tests."""
    console.print(Panel.fit(
        "[bold]Meal Planning Feature - Comprehensive Conversation Tests[/bold]\n"
        "Testing natural language understanding, plan generation, and grocery lists",
        title="Test Suite",
        border_style="cyan"
    ))

    tests = [
        ("Quick Intent Patterns", test_quick_intent_patterns),
        ("Constraint Extraction", test_constraint_extraction),
        ("Meal Plan Generation", test_meal_plan_generation),
        ("Determinism", test_determinism),
        ("Grocery List Generation", test_grocery_list_generation),
        ("Diversity Constraints", test_diversity_constraints),
        ("Ingredient Overlap", test_ingredient_overlap_optimization),
        ("Full Conversation Flow", test_full_conversation_flow),
    ]

    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            console.print(f"[red]Error in {name}: {e}[/red]")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Summary
    console.print("\n" + "=" * 60)
    console.print(Panel.fit("[bold]Test Summary[/bold]", border_style="cyan"))

    table = Table()
    table.add_column("Test", style="cyan")
    table.add_column("Result", style="bold")

    passed_count = 0
    for name, passed in results:
        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        table.add_row(name, status)
        if passed:
            passed_count += 1

    console.print(table)
    console.print(f"\n[bold]Overall: {passed_count}/{len(results)} tests passed[/bold]")

    if passed_count == len(results):
        console.print("\n[bold green]All tests passed! Meal planning feature is working correctly.[/bold green]")
        return 0
    else:
        console.print(f"\n[bold red]{len(results) - passed_count} test(s) failed.[/bold red]")
        return 1


if __name__ == "__main__":
    sys.exit(main())

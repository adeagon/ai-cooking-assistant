"""Intent classification for natural language commands."""

from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama

from src.app.logging_config import get_logger
from src.domain.models import IntentClassification, RecipeCard

logger = get_logger(__name__)

# Quick patterns for stateless commands (bypass LLM)
QUICK_INTENTS = {
    "history": ["history", "show history", "cooking history", "what have i cooked", "my history"],
    "box": ["box", "saved recipes", "my recipes", "bookmarks", "my bookmarks", "recipe box"],
    "new": ["new", "start over", "new session", "reset", "begin again"],
    "prefs": ["prefs", "preferences", "my preferences", "settings", "my settings"],
    "commands": ["commands", "help", "what commands", "show commands", "list commands"],
    "mealplan": [
        "mealplan", "meal plan", "plan meals", "plan my meals", "plan dinners",
        "plan my week", "weekly plan", "plan my dinners", "meal planning",
        "help me plan meals", "help me plan dinners",
    ],
    "grocery_list": [
        "grocery", "groceries", "grocery list", "shopping list",
        "what do i need to buy", "generate grocery list",
    ],
    "show_plan": [
        "show plan", "show my plan", "my meal plan", "current plan",
        "show meal plan", "view plan", "view meal plan",
    ],
    "login": ["login", "log in", "sign in", "switch user"],
    "logout": ["logout", "log out", "sign out"],
    "whoami": ["whoami", "who am i", "current user", "which user"],
}

# Intent classification prompt
INTENT_CLASSIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an intent classifier for a recipe assistant. Analyze user input and determine if they want to perform a specific action on a recipe.

AVAILABLE ACTIONS:
- save: Save/bookmark a recipe for later ("save that", "bookmark the first one", "add to my box")
- like: Express positive preference for a recipe ("I liked it", "that was great", "thumbs up", "loved the chicken one")
- dislike: Express negative preference for a recipe ("didn't like it", "not for me", "thumbs down", "that was bad")
- rate: Give a 1-5 star rating ("give it 4 stars", "rate it a 3", "5 out of 5", "3 stars")
- show: View full recipe details for a SPECIFIC previously-shown recipe ("show me the recipe", "what's in the chicken one", "full details", "ingredients list")
  IMPORTANT: "show me some X" or "show me options for X" is a RECIPE SEARCH, not show - use 'conversation' instead
  "show me some pasta recipes" → conversation (recipe search)
  "show me the pasta recipe" → show (viewing a specific recipe)
- cooked: Mark recipe as cooked/made ("I made it", "cooked that last night", "tried the first one", "made the pasta")
- history: View cooking history ("what have I cooked", "show history", "my cooking log", "recently cooked")
- box: View saved recipes ("show my saved recipes", "what's in my box", "bookmarks", "my recipe box")
- unsave: Remove from saved recipes ("unsave", "remove from bookmarks", "delete from box", "remove that")
- new: Start new session ("start over", "new search", "reset", "begin again", "clear session")
- prefs: Show preferences ("my preferences", "what are my settings", "show prefs", "dietary restrictions")
- commands: Show available commands ("help", "what commands", "list commands")
- addpref: Add a preference (NOT typical - usually handled by slash command)
- filter_previous: Query about ALREADY-SHOWN recipes - sort/filter/ask about previous recommendations
  Examples: "which of those has best reviews?", "the quickest one", "highest rated", "which is fastest?"
  IMPORTANT: Use this when user references "those", "these", "of them" and asks about ratings/time/reviews
  Set filter_type to: "best_rated", "quickest", "most_reviewed", or describe the filter
- mealplan: Start meal planning ("plan my week", "help me plan dinners", "meal plan for next week", "plan 5 dinners")
  IMPORTANT: Use for meal PLANNING requests, not individual recipe searches
  "plan my meals for the week" → mealplan
  "plan 5 vegetarian dinners" → mealplan
  "suggest a dinner" → conversation (single recipe request)
- show_plan: View current/active meal plan ("show my plan", "what's my meal plan", "view plan", "my planned meals")
- grocery_list: Generate grocery list from meal plan ("grocery list", "shopping list", "what do I need to buy")
- conversation: Regular recipe query or chat (DEFAULT - use when uncertain)

CRITICAL RULES:
1. DEFAULT TO 'conversation' when uncertain - false negatives are better than false positives
2. "I love X" where X is a food/ingredient = conversation (preference statement, not action)
3. "I loved it/that" referring to a recent recipe = like (action on specific recipe)
4. Require CLEAR action language - vague positivity is not enough
5. "that was good" without explicit action = conversation (too vague)
6. "that was good, save it" = save (explicit action verb "save")
7. Sentiment alone is NOT an action - require explicit action verbs or clear intent

RECIPE REFERENCES:
- Extract how user refers to recipe: "first one", "the pasta", "it", "that", number like "2"
- If no clear reference but action is clear, set recipe_reference to "it" or "last"
- Recipe references are REQUIRED for: like, dislike, rate, show, save, unsave, cooked
- Recipe references are NOT needed for: history, box, new, prefs, commands

SOURCE FIELD (where the recipe reference comes FROM):
- source="box" ONLY if user is referring to a recipe ALREADY IN their recipe box
  Examples: "show me the pasta from my recipe box", "rate the first one in my saved recipes"
- source="recommendations" (default) if referring to recent recommendations
- IMPORTANT: For "save" commands, source is ALWAYS "recommendations" - the recipe box is the DESTINATION, not the source
  "save it to my recipe box" → source="recommendations" (saving FROM recommendations TO box)
  "save the pasta" → source="recommendations"
- Only use source="box" for show/like/dislike/rate/cooked when user explicitly references their saved recipes

CONFIDENCE LEVELS:
- high: Clear action verb + clear recipe reference (or stateless action like history/box)
- medium: Clear action OR clear reference (not both), some ambiguity
- low: Ambiguous phrasing, use 'conversation' instead

RATING EXTRACTION:
- Extract numeric rating 1-5 from phrases like "4 stars", "rate it 3", "5/5", "3 out of 5"
- Store in rating_value field for rate intent

{last_recommendations_context}

EXAMPLES:
Input: "I loved the first one" -> like, high confidence, recipe_reference="first one"
Input: "I love Italian food" -> conversation (preference statement, not action on recipe)
Input: "Save that for later" -> save, high confidence, recipe_reference="that"
Input: "That was delicious" -> conversation (no action verb, just sentiment)
Input: "Give it 4 stars" -> rate, high confidence, recipe_reference="it", rating_value=4
Input: "Show me the pasta recipe" -> show, high confidence, recipe_reference="pasta"
Input: "What have I cooked recently" -> history, high confidence
Input: "My saved recipes" -> box, high confidence
Input: "Show me the green curry from my recipe box" -> show, high confidence, recipe_reference="green curry", source="box"
Input: "What's in that bookmarked pasta recipe" -> show, high confidence, recipe_reference="pasta", source="box"
Input: "I made the chicken tikka from my saved recipes" -> cooked, high confidence, recipe_reference="chicken tikka", source="box"
Input: "Save it to my recipe box" -> save, high confidence, recipe_reference="it", source="recommendations" (NOT box!)
Input: "Add the first one to my saved recipes" -> save, high confidence, recipe_reference="first one", source="recommendations"
Input: "Show me some salads" -> conversation (recipe search, not viewing a specific recipe)
Input: "Show me options for dinner" -> conversation (recipe search)
Input: "Show me the first one" -> show, high confidence, recipe_reference="first one"
Input: "Plan my meals for next week" -> mealplan, high confidence
Input: "Help me plan 5 vegetarian dinners" -> mealplan, high confidence
Input: "Plan dinners for Monday through Friday" -> mealplan, high confidence
Input: "Show me my meal plan" -> show_plan, high confidence
Input: "What's on my plan for this week" -> show_plan, high confidence
Input: "Generate a grocery list" -> grocery_list, high confidence
Input: "What do I need to buy for my meal plan" -> grocery_list, high confidence
Input: "I need a dinner idea" -> conversation (single recipe request, not meal planning)

Analyze the user input and output a structured classification."""),
    ("human", "{user_input}")
])


def check_quick_intent(user_input: str) -> str | None:
    """Check for stateless commands using quick pattern matching.

    Args:
        user_input: User's input string

    Returns:
        Intent name if matched, None otherwise
    """
    normalized = user_input.strip().lower()

    for intent, patterns in QUICK_INTENTS.items():
        if normalized in patterns:
            logger.info("Quick intent match", intent=intent, user_input=user_input)
            return intent

    return None


def build_recommendations_context(cards: list[RecipeCard]) -> str:
    """Build context about recent recommendations for reference resolution.

    Args:
        cards: List of recently recommended RecipeCard objects

    Returns:
        Formatted context string
    """
    if not cards:
        return "LAST RECOMMENDATIONS: None (no recent recipe suggestions)"

    lines = ["LAST RECOMMENDATIONS (for reference resolution):"]
    for i, card in enumerate(cards, 1):
        lines.append(f"  {i}. {card.title}")

    return "\n".join(lines)


def classify_intent(
    user_input: str,
    last_cards: list[RecipeCard],
    llm: ChatOllama
) -> IntentClassification:
    """Classify user intent from natural language.

    Args:
        user_input: User's input string
        last_cards: List of recently recommended recipes
        llm: LLM client for classification

    Returns:
        IntentClassification object
    """
    # Check for quick stateless intents first
    quick_intent = check_quick_intent(user_input)
    if quick_intent:
        return IntentClassification(
            intent=quick_intent,
            confidence="high",
            reasoning=f"Quick pattern match for {quick_intent} command"
        )

    # Build context
    context = build_recommendations_context(last_cards)

    # Use structured output from LLM
    structured_llm = llm.with_structured_output(IntentClassification)

    try:
        result = structured_llm.invoke(
            INTENT_CLASSIFICATION_PROMPT.format(
                user_input=user_input,
                last_recommendations_context=context
            )
        )

        logger.info(
            "Intent classified",
            intent=result.intent,
            confidence=result.confidence,
            recipe_ref=result.recipe_reference,
            reasoning=result.reasoning
        )

        return result

    except Exception as e:
        logger.error("Intent classification failed", error=str(e))
        # Fall back to conversation on error
        return IntentClassification(
            intent="conversation",
            confidence="low",
            reasoning=f"Classification failed: {e}"
        )


def classify_intent_runnable(input_data: dict[str, Any]) -> dict[str, Any]:
    """Runnable function for intent classification.

    Args:
        input_data: Dictionary with "user_input", "last_cards", "llm" keys

    Returns:
        Input data with "intent_result" key added
    """
    user_input = input_data.get("user_input", "")
    last_cards = input_data.get("last_cards", [])
    llm = input_data.get("llm")

    if not llm:
        logger.error("No LLM provided for intent classification")
        result = IntentClassification(
            intent="conversation",
            confidence="low",
            reasoning="No LLM available"
        )
    else:
        result = classify_intent(user_input, last_cards, llm)

    return {**input_data, "intent_result": result}


# Create LangChain Runnable
IntentClassifierChain = RunnableLambda(classify_intent_runnable)

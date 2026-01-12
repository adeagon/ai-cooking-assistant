"""LangChain prompt templates for chat chains."""

from langchain_core.prompts import ChatPromptTemplate

from src.domain.models import PreferenceProfile, SessionState

# Clarification prompt - ask user for more details
CLARIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful recipe recommendation assistant. The user's request lacks enough detail to search for specific recipes.

Ask 1-2 brief, conversational clarifying questions about:
- What ingredients they have on hand
- How much time they have available
- Any dietary preferences or restrictions
- What type of cuisine or dish they're in the mood for
- Any specific goals (quick, healthy, comfort food, etc.)

Keep questions natural and friendly. Don't ask for information that was already provided.""",
        ),
        ("human", "{user_input}"),
    ]
)

# Recommendation prompt - suggest recipes from retrieved cards
RECOMMENDATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful recipe recommendation assistant.

{preferences_text}

YOUR ROLE:
- Recommend 2-4 recipes from the provided recipe options
- Explain why each recipe matches the user's needs
- Keep responses concise and helpful

IMPORTANT RULES:
1. ONLY recommend recipes from the provided "Recipe Options" below
2. NEVER invent or hallucinate recipe names
3. If the user wants full recipe details, tell them to ask for the specific recipe by name
4. Match recipes to user preferences and session constraints""",
        ),
        (
            "human",
            """{user_input}

{session_context}

RECIPE OPTIONS:
{cards_text}

Based on these options, recommend 2-3 recipes that best match my needs.""",
        ),
    ]
)


def format_preferences(profile: PreferenceProfile) -> str:
    """Format user preferences for prompt.

    Args:
        profile: User's PreferenceProfile

    Returns:
        Formatted preferences text
    """
    lines = ["USER PREFERENCES:"]

    lines.append(f"- Spice level: {profile.spice_level}")
    lines.append(f"- Diet: {profile.diet}")

    if profile.avoid_ingredients:
        avoid_str = ", ".join(profile.avoid_ingredients[:5])
        if len(profile.avoid_ingredients) > 5:
            avoid_str += "..."
        lines.append(f"- Avoid: {avoid_str}")

    if profile.preferred_cuisines:
        cuisines_str = ", ".join(profile.preferred_cuisines[:5])
        lines.append(f"- Preferred cuisines: {cuisines_str}")

    if profile.time_limit_default_minutes:
        lines.append(f"- Default time limit: {profile.time_limit_default_minutes} minutes")

    return "\n".join(lines)


def format_session_context(session: SessionState, rolling_summary: str = "") -> str:
    """Format session context for prompt.

    Args:
        session: Current SessionState
        rolling_summary: Optional rolling summary text

    Returns:
        Formatted session context
    """
    lines = []

    if rolling_summary:
        lines.append(f"PREVIOUS DISCUSSION: {rolling_summary}")
        lines.append("")

    lines.append("CURRENT SESSION:")

    if session.ingredients_on_hand:
        ingredients_str = ", ".join(session.ingredients_on_hand)
        lines.append(f"- Ingredients on hand: {ingredients_str}")

    if session.avoid_tonight:
        avoid_str = ", ".join(session.avoid_tonight)
        lines.append(f"- Avoid tonight: {avoid_str}")

    if session.time_limit_minutes:
        lines.append(f"- Time limit: {session.time_limit_minutes} minutes")

    if session.goals:
        goals_str = ", ".join(session.goals)
        lines.append(f"- Goals: {goals_str}")

    if session.servings:
        lines.append(f"- Servings: {session.servings}")

    # If no session constraints, return minimal context
    if len(lines) <= 1:
        return ""

    return "\n".join(lines)

"""LangChain prompt templates for chat chains."""

from langchain_core.prompts import ChatPromptTemplate

from src.domain.models import PreferenceProfile, SessionState

# Clarification prompt - ask user for more details
CLARIFICATION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful, interactive recipe recommendation assistant. The user's request lacks enough detail to search for specific recipes.

Be PROACTIVE and HELPFUL by:
1. Suggesting specific recipe possibilities to spark ideas
2. Asking about ingredients in a way that opens up options

EXAMPLES of good responses:
- "If you have chicken and some veggies, you could make a quick stir-fry or a comforting chicken soup. Do you have any proteins on hand?"
- "I can suggest anything from a 15-minute pasta to a slow-cooked stew. How much time do you have tonight?"
- "If you're in the mood for something hearty, I know great beef stew and chili recipes. Or if you want lighter fare, there are wonderful salads and grain bowls. What sounds good?"

Ask 1-2 questions that:
- Suggest specific dishes or cuisines as examples
- Help the user discover what they're in the mood for
- Feel like chatting with a knowledgeable friend, not filling out a form

Keep it conversational and inspiring. Don't ask for information already provided.""",
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
- Recommend 6-10 recipes from the provided recipe options
- Number each recommendation (1, 2, 3, etc.) for easy reference
- Keep explanations brief (1 sentence per recipe) to fit more options
- Focus on variety - include different cuisines, cooking methods, and time requirements

IMPORTANT RULES:
1. ONLY recommend recipes from the provided "Recipe Options" below
2. NEVER invent or hallucinate recipe names
3. Always number your recommendations so the user can reference them (e.g., "/show 3" or "tell me about 2")
4. Match recipes to user preferences and session constraints""",
        ),
        (
            "human",
            """{user_input}

{session_context}

RECIPE OPTIONS:
{cards_text}

Based on these options, recommend 6-10 recipes that match my needs. Number each one and keep explanations brief.""",
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

"""Main chat chain orchestration using LangChain LCEL."""

from typing import Any

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableBranch, RunnablePassthrough

from src.app.logging_config import get_logger
from src.chains.extractors import ConstraintExtractorChain
from src.chains.prompts import (
    CLARIFICATION_PROMPT,
    RECOMMENDATION_PROMPT,
    format_preferences,
    format_session_context,
)
from src.chains.retrieval import RetrievalRunnable
from src.domain.models import Constraints, PreferenceProfile, SessionState

logger = get_logger(__name__)


def should_clarify(input_data: dict[str, Any]) -> bool:
    """Determine if we need to ask clarifying questions.

    Args:
        input_data: Dictionary with "constraints" and "session" keys

    Returns:
        True if clarification is needed, False if we can search
    """
    constraints: Constraints = input_data.get("constraints", Constraints())
    session: SessionState = input_data.get("session", SessionState())

    # Check if we have actionable constraints
    has_ingredients = bool(constraints.ingredients or session.ingredients_on_hand)
    has_goals = bool(constraints.goals or constraints.dietary)
    has_cuisine = bool(constraints.cuisine)

    # If no constraints at all, ask clarifying questions
    needs_clarification = not (has_ingredients or has_goals or has_cuisine)

    logger.info(
        "Clarification check",
        needs_clarification=needs_clarification,
        has_ingredients=has_ingredients,
        has_goals=has_goals,
        has_cuisine=has_cuisine,
    )

    return needs_clarification


def build_chat_chain(
    llm: Runnable,
    retrieval_chain: RetrievalRunnable,
    profile: PreferenceProfile,
    session: SessionState,
    rolling_summary: str = "",
) -> Runnable:
    """Build main LCEL chat chain.

    Args:
        llm: LangChain LLM (e.g., ChatOllama)
        retrieval_chain: RetrievalRunnable for recipe search
        profile: User's PreferenceProfile
        session: Current SessionState
        rolling_summary: Rolling session summary

    Returns:
        Runnable chain that processes user input and returns response
    """
    # Format static context
    preferences_text = format_preferences(profile)
    session_context = format_session_context(session, rolling_summary)

    # Build clarification chain
    clarification_chain = CLARIFICATION_PROMPT | llm | StrOutputParser()

    # Build recommendation chain (with retrieval)
    recommendation_chain = (
        # First, retrieve recipes
        retrieval_chain
        # Then, build prompt with cards
        | RunnablePassthrough.assign(
            preferences_text=lambda _: preferences_text,
            session_context=lambda _: session_context,
        )
        # Finally, generate recommendation
        | RECOMMENDATION_PROMPT
        | llm
        | StrOutputParser()
    )

    # Main chain with branching logic
    main_chain = (
        # Start with constraint extraction (adds "constraints" key)
        ConstraintExtractorChain
        # Add session to the dict
        | RunnablePassthrough.assign(session=lambda _: session)
        # Branch based on whether we need clarification
        | RunnableBranch(
            (should_clarify, clarification_chain),
            recommendation_chain,  # default: recommend recipes
        )
    )

    logger.info("Built chat chain", has_profile=bool(profile), has_session=bool(session))

    return main_chain


def build_simple_chat_chain(
    llm: Runnable,
    retrieval_chain: RetrievalRunnable,
) -> Runnable:
    """Build simplified chat chain without user context.

    Args:
        llm: LangChain LLM
        retrieval_chain: RetrievalRunnable

    Returns:
        Runnable chain
    """
    # Use default profile and session
    profile = PreferenceProfile()
    session = SessionState()

    return build_chat_chain(llm, retrieval_chain, profile, session)

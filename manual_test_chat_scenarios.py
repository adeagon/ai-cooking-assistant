#!/usr/bin/env python
"""Comprehensive test scenarios for Phase 4 chat functionality"""

import asyncio
from pathlib import Path

from langchain_ollama import ChatOllama
from src.app.settings import Settings
from src.chains.chat_chain import build_chat_chain
from src.chains.retrieval import RetrievalRunnable
from src.memory.profile_store import ProfileStore
from src.memory.session_store import SessionStore
from src.memory.summarizer import RollingSummarizer
from src.retrieval.recipe_cards import RecipeCardBuilder
from src.retrieval.rerank import RecipeReranker
from src.retrieval.retriever import RecipeRetriever

# Test scenarios
SCENARIOS = [
    {
        "name": "Scenario 1: Vague Request (Should Clarify)",
        "queries": ["What should I cook tonight?"],
        "expected": "Should ask clarifying questions about ingredients, time, or preferences"
    },
    {
        "name": "Scenario 2: Specific Ingredients + Time",
        "queries": ["I have chicken and tomatoes, something quick under 30 minutes"],
        "expected": "Should retrieve and recommend real recipes (may note time constraint issue)"
    },
    {
        "name": "Scenario 3: Dietary Restriction",
        "queries": ["Show me vegetarian pasta recipes"],
        "expected": "Should retrieve vegetarian pasta recipes"
    },
    {
        "name": "Scenario 4: Cuisine Preference",
        "queries": ["I want to make Italian food tonight"],
        "expected": "Should retrieve Italian recipes"
    },
    {
        "name": "Scenario 5: Multiple Constraints",
        "queries": ["I need a healthy, quick dinner with chicken, under 45 minutes"],
        "expected": "Should handle multiple constraints (healthy, quick, chicken, 45min)"
    },
    {
        "name": "Scenario 6: Goal-Based",
        "queries": ["Something comforting and hearty"],
        "expected": "Should retrieve comfort food recipes"
    },
    {
        "name": "Scenario 7: Vegan + Time",
        "queries": ["Quick vegan dinner ideas, I have about 20 minutes"],
        "expected": "Should retrieve vegan recipes considering time"
    },
    {
        "name": "Scenario 8: Multi-Turn Conversation",
        "queries": [
            "Show me pasta recipes",
            "Actually, I prefer something with a tomato base"
        ],
        "expected": "Should refine based on second query, using memory"
    },
    {
        "name": "Scenario 9: Gluten-Free",
        "queries": ["I need gluten-free dinner ideas"],
        "expected": "Should retrieve gluten-free recipes"
    },
    {
        "name": "Scenario 10: Just Cuisine (No Other Details)",
        "queries": ["Mexican food"],
        "expected": "Should retrieve Mexican recipes without clarification"
    }
]


async def test_scenario(llm, retrieval_chain, profile_store, session_store, scenario):
    """Test a single scenario"""
    print(f"\n{'='*80}")
    print(f"{scenario['name']}")
    print(f"{'='*80}")
    print(f"Expected: {scenario['expected']}\n")

    # Create new session for this scenario
    session_store._current_session_id = None
    profile = profile_store.load()
    session_id, session = session_store.get_or_create_current()
    rolling_summary = ""

    for i, query in enumerate(scenario['queries'], 1):
        if len(scenario['queries']) > 1:
            print(f"\n--- Turn {i} ---")

        print(f"You: {query}\n")

        # Build chain with current context
        chain = build_chat_chain(llm, retrieval_chain, profile, session, rolling_summary)

        # Get response
        print("Assistant: ", end="", flush=True)
        response = await chain.ainvoke({"user_input": query})
        print(response)

        # Update rolling summary for multi-turn
        if len(scenario['queries']) > 1:
            from src.chains.extractors import ConstraintExtractor
            extractor = ConstraintExtractor()
            constraints = extractor.extract_constraints(query)

            summarizer = RollingSummarizer()
            rolling_summary = summarizer.update_summary(rolling_summary, constraints, query)

            if rolling_summary:
                print(f"\n[Memory Updated: {rolling_summary}]")

    print(f"\n{'='*80}\n")


async def main():
    settings = Settings()

    print("="*80)
    print("PHASE 4 CHAT - COMPREHENSIVE TEST SCENARIOS")
    print("="*80)
    print(f"\nInitializing components...")

    # Initialize LLM
    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.llm_temperature,
        num_predict=settings.llm_max_tokens,
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
        settings=settings,
    )

    # Initialize memory
    profile_store = ProfileStore(db_path=settings.sqlite_db_path)
    session_store = SessionStore(db_path=settings.sqlite_db_path)

    print("Ready!\n")

    # Run all scenarios
    for scenario in SCENARIOS:
        try:
            await test_scenario(llm, retrieval_chain, profile_store, session_store, scenario)
        except Exception as e:
            print(f"\n[ERROR in {scenario['name']}: {e}]\n")
            continue

    print("="*80)
    print("ALL SCENARIOS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())

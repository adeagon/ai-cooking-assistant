"""Compare clarification quality: reasoning=True vs reasoning=False.

This script runs the same vague queries twice - once with reasoning enabled
and once without - to demonstrate the quality difference.
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from langchain_ollama import ChatOllama
from src.app.settings import settings
from src.retrieval.retriever import RecipeRetriever
from src.retrieval.rerank import RecipeReranker
from src.retrieval.recipe_cards import RecipeCardBuilder
from src.chains.retrieval import RetrievalRunnable
from src.chains.chat_chain import build_chat_chain
from src.memory import ProfileStore, SessionStore, FeedbackStore, HistoryStore

# Queries that should trigger clarification
TEST_QUERIES = [
    "food",
    "make me dinner",
    "I'm feeling lazy",
    "surprise me",
    "I don't know what to cook",
]


async def get_clarification_response(query: str, llm_clarification, llm, retrieval_chain,
                                      profile, session_store, feedback_store, history_store):
    """Get a clarification response for a query."""
    session_id = session_store.create()
    session = session_store.get(session_id)

    exclude_ids = (
        feedback_store.get_liked_recipe_ids(limit=20) |
        feedback_store.get_disliked_recipe_ids() |
        history_store.get_recently_cooked_ids(days=7)
    )

    chain = build_chat_chain(
        llm=llm,
        retrieval_chain=retrieval_chain,
        profile=profile,
        session=session,
        rolling_summary="",
        exclude_recipe_ids=exclude_ids,
        llm_clarification=llm_clarification,
    )

    result = await chain.ainvoke({"user_input": query})
    return result.get("response", "")


async def main():
    output_lines = []

    def log(msg):
        output_lines.append(msg)
        # Safe print for Windows
        try:
            print(msg.encode('ascii', 'replace').decode('ascii'))
        except:
            print(msg.encode('cp1252', 'replace').decode('cp1252'))

    log("=" * 80)
    log("CLARIFICATION QUALITY COMPARISON")
    log("reasoning=True (Thoughtful) vs reasoning=False (Fast)")
    log("=" * 80)
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"Model: {settings.ollama_model}")
    log("")

    # Initialize shared components
    log("Initializing components...")

    llm_fast = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.llm_temperature,
        num_predict=settings.llm_max_tokens,
        reasoning=False,
    )

    llm_thoughtful = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.llm_temperature,
        num_predict=settings.llm_max_tokens * 2,
        reasoning=True,
    )

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

    profile_store = ProfileStore(db_path=settings.sqlite_db_path)
    session_store = SessionStore(db_path=settings.sqlite_db_path)
    feedback_store = FeedbackStore(db_path=settings.sqlite_db_path)
    history_store = HistoryStore(db_path=settings.sqlite_db_path)

    profile = profile_store.load()

    log("Ready!\n")

    for i, query in enumerate(TEST_QUERIES, 1):
        log("=" * 80)
        log(f"TEST {i}/{len(TEST_QUERIES)}: \"{query}\"")
        log("=" * 80)

        # Test with reasoning=False (fast mode)
        log("\n[MODE: reasoning=False (Fast)]")
        try:
            response_fast = await get_clarification_response(
                query, llm_fast, llm_fast, retrieval_chain,
                profile, session_store, feedback_store, history_store
            )
            log(f"Response: {response_fast}")
        except Exception as e:
            log(f"Error: {e}")
            response_fast = ""

        # Test with reasoning=True (thoughtful mode)
        log("\n[MODE: reasoning=True (Thoughtful)]")
        try:
            response_thoughtful = await get_clarification_response(
                query, llm_thoughtful, llm_fast, retrieval_chain,
                profile, session_store, feedback_store, history_store
            )
            log(f"Response: {response_thoughtful}")
        except Exception as e:
            log(f"Error: {e}")
            response_thoughtful = ""

        # Quality comparison
        log("\n[COMPARISON]")
        fast_len = len(response_fast)
        thoughtful_len = len(response_thoughtful)

        fast_questions = response_fast.count("?")
        thoughtful_questions = response_thoughtful.count("?")

        fast_options = sum(1 for word in ["or", "would you", "do you", "what about", "how about"]
                          if word in response_fast.lower())
        thoughtful_options = sum(1 for word in ["or", "would you", "do you", "what about", "how about"]
                                if word in response_thoughtful.lower())

        log(f"  Length:    Fast={fast_len} chars, Thoughtful={thoughtful_len} chars")
        log(f"  Questions: Fast={fast_questions}, Thoughtful={thoughtful_questions}")
        log(f"  Options:   Fast={fast_options}, Thoughtful={thoughtful_options}")
        log("")

    log("=" * 80)
    log("COMPARISON COMPLETE")
    log("=" * 80)
    log(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Save full output with UTF-8
    output_file = "test_results/clarification_comparison.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))
    print(f"\nFull output saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())

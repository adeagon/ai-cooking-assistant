"""Application context for shared expensive resources.

These resources are initialized once at startup and shared across all requests.
"""

from dataclasses import dataclass
from pathlib import Path

from langchain_ollama import ChatOllama

from src.app.logging_config import get_logger
from src.app.settings import settings
from src.retrieval.recipe_cards import RecipeCardBuilder
from src.retrieval.rerank import RecipeReranker
from src.retrieval.retriever import RecipeRetriever
from src.chains.retrieval import RetrievalRunnable

logger = get_logger(__name__)


@dataclass
class AppContext:
    """Shared expensive resources initialized once at startup.

    These include LLM clients, retrieval components, and other resources
    that are expensive to initialize but can be safely shared across requests.
    """

    # LLM instances
    llm: ChatOllama  # Fast mode for recommendations
    llm_clarification: ChatOllama  # Thoughtful mode for clarification
    intent_llm: ChatOllama  # Intent classification

    # Retrieval components
    retriever: RecipeRetriever
    reranker: RecipeReranker
    card_builder: RecipeCardBuilder
    retrieval_chain: RetrievalRunnable

    # Database path
    db_path: Path

    @property
    def is_ready(self) -> bool:
        """Check if context is fully initialized."""
        return all([
            self.llm is not None,
            self.retriever is not None,
            self.reranker is not None,
            self.card_builder is not None,
            self.retrieval_chain is not None,
        ])


def create_app_context(db_path: Path | None = None) -> AppContext:
    """Create application context with all expensive resources.

    This should be called once at application startup.

    Args:
        db_path: Optional database path override

    Returns:
        Initialized AppContext

    Raises:
        RuntimeError: If initialization fails
    """
    db_path = db_path or settings.sqlite_db_path
    chroma_dir = Path(settings.chroma_persist_dir)

    # Verify prerequisites
    if not chroma_dir.exists():
        raise RuntimeError(
            f"Vector store not found at: {chroma_dir}. "
            "Run 'python -m src.app.cli ingest embed' to build it."
        )

    if not db_path.exists():
        raise RuntimeError(
            f"Database not found at: {db_path}. "
            "Run 'python -m src.app.cli ingest process' to build it."
        )

    logger.info("Initializing application context...")

    # Initialize LLMs
    # Main LLM for recommendations - fast, direct responses
    llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.llm_temperature,
        num_predict=settings.llm_max_tokens,
        # Note: reasoning=False is default, no thinking for recipe presentation
    )

    # LLM for clarification - thoughtful, uses reasoning for better questions
    llm_clarification = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_model,
        temperature=settings.llm_temperature,
        num_predict=settings.llm_max_tokens * 2,  # Extra budget for thinking + response
        # Note: We'll set reasoning=True when we have proper async support
    )

    # Separate LLM for intent classification
    # Lower temperature for more deterministic classification
    intent_llm = ChatOllama(
        base_url=settings.ollama_base_url,
        model=settings.ollama_intent_model,
        temperature=0.2,
        num_predict=256,  # Intent classification needs fewer tokens
    )

    # Initialize retrieval components
    retriever = RecipeRetriever(
        chroma_dir=chroma_dir,
        embedding_model=settings.embedding_model,
    )

    reranker = RecipeReranker(model_name=settings.reranker_model)

    card_builder = RecipeCardBuilder(db_path=db_path)

    retrieval_chain = RetrievalRunnable(
        retriever=retriever,
        reranker=reranker,
        card_builder=card_builder,
        settings=settings,
    )

    logger.info(
        "Application context initialized",
        model=settings.ollama_model,
        embedding_model=settings.embedding_model,
        reranker_model=settings.reranker_model,
    )

    return AppContext(
        llm=llm,
        llm_clarification=llm_clarification,
        intent_llm=intent_llm,
        retriever=retriever,
        reranker=reranker,
        card_builder=card_builder,
        retrieval_chain=retrieval_chain,
        db_path=db_path,
    )

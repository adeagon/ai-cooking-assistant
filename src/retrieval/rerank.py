"""Cross-encoder reranking for recipe search results."""

from sentence_transformers import CrossEncoder
from src.domain.models import RetrievalResult
from src.app.logging_config import get_logger

logger = get_logger(__name__)


class RecipeReranker:
    """
    Cross-encoder reranker for recipe search results.

    Uses GPU-accelerated sentence-transformers CrossEncoder
    to rerank vector search candidates for improved relevance.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize the reranker with a cross-encoder model.

        Args:
            model_name: HuggingFace model identifier for cross-encoder
        """
        logger.info(f"Loading cross-encoder model: {model_name}")
        self.model = CrossEncoder(model_name)
        self.model_name = model_name

        # Log device info
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Cross-encoder using device: {device}")

    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int = 20
    ) -> list[RetrievalResult]:
        """
        Rerank candidates using cross-encoder scoring.

        Args:
            query: Original user search query
            candidates: List of RetrievalResult from vector search
            top_k: Number of top results to return after reranking

        Returns:
            List of RetrievalResult sorted by cross-encoder score (descending),
            with score field updated to cross-encoder score
        """
        if not candidates:
            return []

        if len(candidates) < top_k:
            top_k = len(candidates)

        # Build pairs for scoring
        pairs = self._build_pairs(query, candidates)

        # Get cross-encoder scores (batch prediction for efficiency)
        logger.debug(f"Scoring {len(pairs)} query-document pairs")
        scores = self.model.predict(pairs)

        # Combine candidates with new scores
        scored_candidates = list(zip(candidates, scores))

        # Sort by cross-encoder score (descending)
        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Take top_k and update scores
        reranked = []
        for candidate, score in scored_candidates[:top_k]:
            reranked.append(
                RetrievalResult(
                    recipe_id=candidate.recipe_id,
                    title=candidate.title,
                    score=float(score),  # Cross-encoder score
                    rating_avg=candidate.rating_avg,
                    rating_count=candidate.rating_count,
                    minutes=candidate.minutes
                )
            )

        logger.info(
            f"Reranking complete: {len(candidates)} -> {len(reranked)} results "
            f"(top score: {reranked[0].score:.3f})"
        )

        return reranked

    def _build_pairs(
        self,
        query: str,
        candidates: list[RetrievalResult]
    ) -> list[tuple[str, str]]:
        """
        Build query-document pairs for cross-encoder scoring.

        Args:
            query: User search query
            candidates: List of candidates to pair with query

        Returns:
            List of (query, document_text) tuples
        """
        return [(query, candidate.title) for candidate in candidates]

"""Recipe retrieval using ChromaDB vector search."""

from pathlib import Path
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from src.domain.models import RetrievalResult
from src.app.logging_config import get_logger

logger = get_logger(__name__)


class RecipeRetriever:
    """
    Vector search retriever for recipes using ChromaDB.

    Loads embedding model once and caches it for repeated queries.
    """

    def __init__(
        self,
        chroma_dir: Path,
        embedding_model: str,
        collection_name: str = "recipes"
    ):
        """
        Initialize retriever.

        Args:
            chroma_dir: ChromaDB persistence directory
            embedding_model: Name of sentence-transformers model
            collection_name: Name of the ChromaDB collection
        """
        self.chroma_dir = chroma_dir
        self.embedding_model_name = embedding_model
        self.collection_name = collection_name

        # Load embedding model (cached for subsequent queries)
        logger.info(f"Loading embedding model: {embedding_model}")
        self.model = SentenceTransformer(embedding_model)

        # Initialize ChromaDB client
        logger.info(f"Connecting to ChromaDB at: {chroma_dir}")
        self.client = chromadb.PersistentClient(
            path=str(chroma_dir),
            settings=ChromaSettings(anonymized_telemetry=False)
        )

        # Get collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            logger.info(f"Connected to collection: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to get collection {collection_name}: {e}")
            raise ValueError(
                f"Collection '{collection_name}' not found. Run 'ingest embed' first."
            )

    def search(
        self,
        query: str,
        k: int = 30,
        filters: dict | None = None
    ) -> list[RetrievalResult]:
        """
        Search for recipes using vector similarity.

        Args:
            query: Natural language search query
            k: Number of results to return
            filters: Optional metadata filters (e.g., {"rating_avg": {"$gte": 4.0}})

        Returns:
            List of RetrievalResult objects sorted by similarity score (descending)
        """
        # Generate query embedding
        query_embedding = self.model.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()

        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=filters if filters else None
        )

        # Parse results into RetrievalResult objects
        retrieval_results = []
        for i in range(len(results['ids'][0])):
            recipe_id = results['ids'][0][i]
            metadata = results['metadatas'][0][i]
            distance = results['distances'][0][i]

            # Convert distance to similarity score (cosine distance -> similarity)
            # ChromaDB returns cosine distance (0 = identical, 2 = opposite)
            # Convert to similarity: similarity = 1 - (distance / 2)
            similarity_score = 1 - (distance / 2)

            retrieval_results.append(
                RetrievalResult(
                    recipe_id=recipe_id,
                    title=metadata.get('title', ''),
                    score=similarity_score,
                    rating_avg=metadata.get('rating_avg'),
                    rating_count=metadata.get('rating_count'),
                    minutes=metadata.get('minutes')
                )
            )

        logger.info(f"Retrieved {len(retrieval_results)} results for query: {query}")
        return retrieval_results

    def search_with_filters(
        self,
        query: str,
        k: int = 30,
        min_rating: float | None = None,
        max_minutes: int | None = None
    ) -> list[RetrievalResult]:
        """
        Search with convenient rating and time filters.

        Args:
            query: Natural language search query
            k: Number of results to return
            min_rating: Minimum average rating (e.g., 4.0)
            max_minutes: Maximum cooking time in minutes

        Returns:
            List of RetrievalResult objects sorted by similarity score
        """
        filters = {}

        if min_rating is not None:
            filters["rating_avg"] = {"$gte": min_rating}

        if max_minutes is not None:
            filters["minutes"] = {"$lte": max_minutes}

        # Combine filters if both are specified
        if len(filters) > 1:
            filters = {"$and": [
                {k: v} for k, v in filters.items()
            ]}

        return self.search(query, k, filters if filters else None)

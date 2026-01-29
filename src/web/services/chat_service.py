"""Chat service for processing user messages with SSE streaming."""

import json
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from langchain_ollama import ChatOllama

from src.app.logging_config import get_logger
from src.app.settings import settings
from src.chains.chat_chain import build_chat_chain
from src.chains.intent_classifier import classify_intent
from src.chains.extractors import ConstraintExtractor
from src.domain.models import RecipeCard, RecipeFeedback
from src.memory import RollingSummarizer
from src.memory.store_factory import StoreFactory, UserStores
from src.retrieval.recipe_cards import RecipeCardBuilder
from src.retrieval.rerank import RecipeReranker
from src.retrieval.retriever import RecipeRetriever
from src.chains.retrieval import RetrievalRunnable
from src.web.models import RecipeCardMeta, User

logger = get_logger(__name__)


class ChatService:
    """Service for processing chat messages.

    Manages LLM, retrieval components, and handles streaming responses.
    Streaming is the source of truth - non-streaming calls this and buffers.
    """

    def __init__(
        self,
        store_factory: StoreFactory,
        chroma_dir: Path | None = None,
        sqlite_db_path: Path | None = None,
    ):
        """Initialize chat service.

        Args:
            store_factory: Factory for user-scoped stores
            chroma_dir: Path to ChromaDB directory (default: from settings)
            sqlite_db_path: Path to SQLite DB (default: from settings)
        """
        self.store_factory = store_factory
        self.chroma_dir = chroma_dir or Path(settings.chroma_persist_dir)
        self.sqlite_db_path = sqlite_db_path or Path(settings.sqlite_db_path)

        # Lazy initialization - components created on first use
        self._initialized = False
        self._llm = None
        self._llm_clarification = None
        self._intent_llm = None
        self._retrieval_chain = None
        self._summarizer = None
        self._constraint_extractor = None

        # Per-user state (user_id -> state)
        self._user_state: dict[str, dict[str, Any]] = {}

    def _ensure_initialized(self) -> None:
        """Initialize LLM and retrieval components if not already done."""
        if self._initialized:
            return

        logger.info("Initializing chat service components")

        # Initialize LLMs
        # Main LLM for recommendations - fast, direct responses
        self._llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            num_predict=settings.llm_max_tokens,
            reasoning=False,  # Fast mode
        )

        # LLM for clarification - thoughtful, uses reasoning
        self._llm_clarification = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            temperature=settings.llm_temperature,
            num_predict=settings.llm_max_tokens * 2,
            reasoning=True,  # Enable thinking for better questions
        )

        # Intent classification LLM
        self._intent_llm = ChatOllama(
            base_url=settings.ollama_base_url,
            model=settings.ollama_intent_model,
            temperature=0.2,
            num_predict=256,
            reasoning=False,
        )

        # Initialize retrieval components
        retriever = RecipeRetriever(
            chroma_dir=self.chroma_dir,
            embedding_model=settings.embedding_model
        )

        reranker = RecipeReranker(model_name=settings.reranker_model)

        card_builder = RecipeCardBuilder(db_path=self.sqlite_db_path)

        self._retrieval_chain = RetrievalRunnable(
            retriever=retriever,
            reranker=reranker,
            card_builder=card_builder,
            settings=settings
        )

        self._summarizer = RollingSummarizer()
        self._constraint_extractor = ConstraintExtractor()

        self._initialized = True
        logger.info("Chat service components initialized")

    def _get_user_state(self, user_id: str, username: str) -> dict[str, Any]:
        """Get or create state for a user.

        Args:
            user_id: User UUID
            username: Username for store lookup

        Returns:
            User state dictionary
        """
        if user_id not in self._user_state:
            stores = self.store_factory.get_stores(username)
            profile = stores.profile.load()
            session_id, session = stores.session.get_or_create_current()
            rolling_summary = stores.session.get_summary(session_id)

            self._user_state[user_id] = {
                "stores": stores,
                "profile": profile,
                "session_id": session_id,
                "session": session,
                "rolling_summary": rolling_summary,
                "last_cards": [],
            }
            logger.info("Created user state", user_id=user_id[:8], username=username)

        return self._user_state[user_id]

    def _refresh_user_state(self, user_id: str, username: str) -> dict[str, Any]:
        """Force refresh user state (e.g., after login).

        Args:
            user_id: User UUID
            username: Username

        Returns:
            Fresh user state
        """
        if user_id in self._user_state:
            del self._user_state[user_id]
        return self._get_user_state(user_id, username)

    async def process_message(
        self,
        message: str,
        user: User,
    ) -> tuple[str, list[RecipeCard]]:
        """Process a chat message (non-streaming).

        Calls streaming internally and buffers the output.

        Args:
            message: User message
            user: Current user

        Returns:
            Tuple of (response_text, recipe_cards)
        """
        full_response = ""
        recipe_cards: list[RecipeCard] = []

        async for event_type, data in self.stream_message(message, user):
            if event_type == "token":
                full_response += data.get("content", "")
            elif event_type == "done":
                # Cards are returned in done event
                pass
            elif event_type == "cards":
                recipe_cards = data.get("cards", [])

        return full_response, recipe_cards

    async def stream_message(
        self,
        message: str,
        user: User,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Stream chat response as SSE events.

        This is the source of truth for chat processing.

        Args:
            message: User message
            user: Current user

        Yields:
            Tuples of (event_type, event_data)
            - ("token", {"content": str}) for streaming tokens
            - ("cards", {"cards": list[RecipeCard]}) when cards are ready
            - ("done", {"meta": dict}) when complete
            - ("error", {"error": {"message": str, "code": str}}) on error
        """
        self._ensure_initialized()

        try:
            # Get user state
            state = self._get_user_state(user.id, user.username)
            stores: UserStores = state["stores"]
            profile = state["profile"]
            session = state["session"]
            session_id = state["session_id"]
            rolling_summary = state["rolling_summary"]
            last_cards = state["last_cards"]

            # Try intent classification first (skip for very short messages)
            if len(message.strip()) > 2:
                try:
                    intent_result = classify_intent(
                        message, last_cards, self._intent_llm
                    )

                    # Handle non-conversation intents
                    if intent_result.intent != "conversation":
                        response = await self._handle_intent(
                            intent_result, state, stores
                        )
                        if response:
                            # Intent handled - yield response directly
                            yield ("token", {"content": response})
                            yield ("done", {"meta": {}})
                            return

                except Exception as e:
                    logger.warning(
                        "Intent classification failed",
                        error=str(e)
                    )

            # Build exclusion set
            exclude_ids = (
                stores.feedback.get_liked_recipe_ids(limit=20) |
                stores.feedback.get_disliked_recipe_ids() |
                stores.history.get_recently_cooked_ids(days=7)
            )

            # Build chat chain
            chain = build_chat_chain(
                llm=self._llm,
                retrieval_chain=self._retrieval_chain,
                profile=profile,
                session=session,
                rolling_summary=rolling_summary,
                exclude_recipe_ids=exclude_ids,
                llm_clarification=self._llm_clarification,
            )

            # Invoke chain (not streaming from LangChain - we stream at our level)
            # Note: For true LLM streaming, we'd need to modify the chain
            result = await chain.ainvoke({"user_input": message})

            response = result.get("response", "")
            cards = result.get("cards", [])

            # Update state with new cards
            if cards:
                state["last_cards"] = cards

            # Yield response as tokens (could be chunked for streaming effect)
            # For now, yield entire response
            if response:
                yield ("token", {"content": response})

            # Yield cards if present
            if cards:
                yield ("cards", {"cards": cards})

            # Update rolling summary
            constraints = self._constraint_extractor.extract_constraints(message)
            new_summary = self._summarizer.update_summary(
                rolling_summary, constraints, message
            )
            stores.session.update_summary(session_id, new_summary)
            state["rolling_summary"] = new_summary

            # Convert cards for meta
            card_metas = [self._card_to_meta(c) for c in cards]

            yield ("done", {"meta": {"recipe_cards": card_metas}})

            logger.info(
                "Processed message",
                user_id=user.id[:8],
                message_length=len(message),
                response_length=len(response),
                card_count=len(cards)
            )

        except Exception as e:
            logger.exception("Chat processing error", user_id=user.id[:8])
            yield ("error", {"error": {"message": str(e), "code": "llm_error"}})

    async def _handle_intent(
        self,
        intent_result: Any,
        state: dict[str, Any],
        stores: UserStores,
    ) -> str | None:
        """Handle a classified intent.

        Args:
            intent_result: Result from intent classifier
            state: User state
            stores: User stores

        Returns:
            Response string if intent handled, None to fall through to chat
        """
        intent = intent_result.intent
        last_cards = state["last_cards"]
        session_id = state["session_id"]

        # Resolve recipe reference if present
        recipe_id = None
        title = None
        if intent_result.recipe_reference:
            resolved = self._resolve_recipe_reference(
                intent_result.recipe_reference, last_cards
            )
            if resolved:
                recipe_id, title = resolved

        if intent == "like" and recipe_id:
            stores.feedback.add_feedback(RecipeFeedback(
                recipe_id=recipe_id,
                feedback_type="like",
                session_id=session_id
            ))
            return f"Liked: {title}"

        elif intent == "dislike" and recipe_id:
            stores.feedback.add_feedback(RecipeFeedback(
                recipe_id=recipe_id,
                feedback_type="dislike",
                session_id=session_id
            ))
            return f"Disliked: {title}"

        elif intent == "rate" and recipe_id and intent_result.rating_value:
            rating = intent_result.rating_value
            stores.feedback.add_feedback(RecipeFeedback(
                recipe_id=recipe_id,
                feedback_type="rate",
                rating=rating,
                session_id=session_id
            ))
            return f"Rated {title}: {rating} stars"

        elif intent == "save" and recipe_id:
            try:
                stores.recipe_box.save_recipe(recipe_id, title)
                return f"Saved to Recipe Box: {title}"
            except Exception as e:
                if "UNIQUE" in str(e):
                    return f"Already saved: {title}"
                raise

        elif intent == "cooked" and recipe_id:
            stores.history.add_cooked(recipe_id)
            return f"Marked as cooked: {title}"

        # Intent not handled - fall through to chat
        return None

    def _resolve_recipe_reference(
        self,
        ref: str,
        cards: list[RecipeCard]
    ) -> tuple[str, str] | None:
        """Resolve a recipe reference to (recipe_id, title).

        Args:
            ref: Reference string (number or name)
            cards: Recent recipe cards

        Returns:
            Tuple of (recipe_id, title) or None if not found
        """
        if not cards:
            return None

        # Try numeric reference
        try:
            idx = int(ref.strip()) - 1
            if 0 <= idx < len(cards):
                card = cards[idx]
                return (card.recipe_id, card.title)
        except ValueError:
            pass

        # Try name matching
        ref_lower = ref.lower().strip()
        for card in cards:
            if ref_lower in card.title.lower():
                return (card.recipe_id, card.title)

        return None

    def _card_to_meta(self, card: RecipeCard) -> dict[str, Any]:
        """Convert RecipeCard to metadata dict for API response.

        Args:
            card: RecipeCard from retrieval

        Returns:
            Dictionary matching RecipeCardMeta schema
        """
        return {
            "recipe_id": card.recipe_id,
            "title": card.title,
            "rating_avg": card.rating_avg,
            "time_total": card.time_total,
            "key_ingredients": card.key_ingredients,
            "one_sentence_summary": card.one_sentence_summary,
            "why_match": card.why_match,
        }

    def clear_user_state(self, user_id: str) -> None:
        """Clear cached state for a user (e.g., on logout).

        Args:
            user_id: User UUID
        """
        if user_id in self._user_state:
            del self._user_state[user_id]
            logger.info("Cleared user state", user_id=user_id[:8])

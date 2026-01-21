"""Chat service for processing messages in web and CLI contexts."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from src.app.logging_config import get_logger
from src.domain.models import PreferenceProfile, RecipeCard, RecipeFeedback, SavedRecipe

if TYPE_CHECKING:
    from src.services.app_context import AppContext

logger = get_logger(__name__)


@dataclass
class ChatResult:
    """Result from processing a chat message."""

    response: str
    cards: list[RecipeCard] = field(default_factory=list)
    rolling_summary: str | None = None
    command_executed: bool = False


@dataclass
class UserContext:
    """Per-request user-scoped stores."""

    user_id: str
    db_path: Path

    # Lazy-loaded stores (initialized on first access)
    _profile_store: "ProfileStore | None" = field(default=None, repr=False)
    _session_store: "SessionStore | None" = field(default=None, repr=False)
    _feedback_store: "FeedbackStore | None" = field(default=None, repr=False)
    _history_store: "HistoryStore | None" = field(default=None, repr=False)
    _recipe_box_store: "RecipeBoxStore | None" = field(default=None, repr=False)
    _meal_plan_store: "MealPlanStore | None" = field(default=None, repr=False)

    @property
    def profile_store(self):
        """Get ProfileStore (lazy initialization)."""
        if self._profile_store is None:
            from src.memory.profile_store import ProfileStore
            self._profile_store = ProfileStore(self.db_path, self.user_id)
        return self._profile_store

    @property
    def session_store(self):
        """Get SessionStore (lazy initialization)."""
        if self._session_store is None:
            from src.memory.session_store import SessionStore
            self._session_store = SessionStore(self.db_path, self.user_id)
        return self._session_store

    @property
    def feedback_store(self):
        """Get FeedbackStore (lazy initialization)."""
        if self._feedback_store is None:
            from src.memory.feedback_store import FeedbackStore
            self._feedback_store = FeedbackStore(self.db_path, self.user_id)
        return self._feedback_store

    @property
    def history_store(self):
        """Get HistoryStore (lazy initialization)."""
        if self._history_store is None:
            from src.memory.history_store import HistoryStore
            self._history_store = HistoryStore(self.db_path, self.user_id)
        return self._history_store

    @property
    def recipe_box_store(self):
        """Get RecipeBoxStore (lazy initialization)."""
        if self._recipe_box_store is None:
            from src.memory.recipe_box_store import RecipeBoxStore
            self._recipe_box_store = RecipeBoxStore(self.db_path, self.user_id)
        return self._recipe_box_store

    @property
    def meal_plan_store(self):
        """Get MealPlanStore (lazy initialization)."""
        if self._meal_plan_store is None:
            from src.memory.meal_plan_store import MealPlanStore
            self._meal_plan_store = MealPlanStore(self.db_path, self.user_id)
        return self._meal_plan_store


class ChatService:
    """Service for processing chat messages.

    This service can be used by both web and CLI interfaces.
    """

    def __init__(
        self,
        user_ctx: UserContext,
        last_cards: list[dict] | None = None,
        app_ctx: "AppContext | None" = None,
    ):
        """Initialize chat service.

        Args:
            user_ctx: User context with stores
            last_cards: Previous recipe cards for reference resolution
            app_ctx: Optional app context with LLM and retrieval components.
                    If None, only commands will work (no LLM chat).
        """
        self.user_ctx = user_ctx
        self.last_cards = last_cards or []
        self.app_ctx = app_ctx

    def process_message(self, message: str, rolling_summary: str | None = None) -> ChatResult:
        """Process a chat message and return result.

        Args:
            message: User's message
            rolling_summary: Current rolling summary for context

        Returns:
            ChatResult with response and updated state
        """
        message = message.strip()

        if not message:
            return ChatResult(response="Please enter a message.", command_executed=False)

        # Check for commands (start with /)
        if message.startswith("/"):
            return self._handle_command(message)

        # Check for quick reference patterns before intent classification
        # This handles "save that one", "number 1", "show me 2", etc.
        quick_result = self._try_quick_reference(message)
        if quick_result:
            return quick_result

        # Only try intent classification for short messages that might be commands
        # Skip for longer messages (likely recipe queries) to improve performance
        # Intent classification adds ~4-5 seconds and often fails for regular chat
        if self.app_ctx and len(message) < 50:
            # Quick keyword check - only classify if message contains command-like words
            msg_lower = message.lower()
            command_keywords = {
                "like", "love", "save", "show", "dislike", "hate", "cooked", "made",
                "history", "box", "preferences", "prefs", "rate", "star", "plan",
                "grocery", "meal"
            }
            if any(kw in msg_lower for kw in command_keywords):
                intent_result = self._try_intent_classification(message)
                if intent_result:
                    return intent_result

        # Regular chat message - process through LLM
        return self._handle_chat_message(message, rolling_summary)

    def _try_quick_reference(self, message: str) -> ChatResult | None:
        """Try to handle quick recipe reference patterns.

        Handles patterns like:
        - "save that", "save it", "save the first one"
        - "like that", "like it", "like number 1"
        - "show me 1", "show number 2", "recipe 3"
        - "the first one", "number 1", "#1"

        Args:
            message: User's message

        Returns:
            ChatResult if pattern matched and handled, None otherwise
        """
        if not self.last_cards:
            logger.debug("Quick reference: no cards available")
            return None

        msg_lower = message.lower().strip()
        logger.debug("Quick reference check", message=msg_lower, card_count=len(self.last_cards))

        # Patterns for different actions
        save_patterns = ["save that", "save it", "save this", "save the first", "save the second", "save the third", "save number", "save #"]
        like_patterns = ["like that", "like it", "like this", "like the first", "like the second", "love that", "love it", "loved it", "loved that", "like number", "like #"]
        dislike_patterns = ["dislike that", "dislike it", "didn't like", "don't like", "dislike the first", "dislike number"]
        show_patterns = ["show me", "show that", "show it", "show the first", "show the second", "show number", "recipe for number", "give me the recipe", "give me recipe", "what's in number"]
        cooked_patterns = ["made that", "made it", "cooked that", "cooked it", "i made the first", "cooked number"]

        # Reference patterns (numbers)
        ref = None

        # First check for ordinal words - these are most specific
        ordinal_map = {
            "first": "1", "1st": "1",
            "second": "2", "2nd": "2",
            "third": "3", "3rd": "3",
            "fourth": "4", "4th": "4",
            "fifth": "5", "5th": "5",
            "sixth": "6", "6th": "6",
        }
        for word, num in ordinal_map.items():
            if word in msg_lower:
                ref = num
                logger.debug("Quick reference: matched ordinal", word=word, ref=ref)
                break

        # Check for explicit number patterns: "number 1", "#1", "recipe 1"
        if not ref:
            num_match = re.search(r'(?:number\s*|#|recipe\s*)(\d+)', msg_lower)
            if num_match:
                ref = num_match.group(1)
                logger.debug("Quick reference: matched number pattern", ref=ref)

        # Check for standalone digit at word boundary
        # Only match in SHORT messages to avoid false positives like "5 nights. we have chicken..."
        if not ref and len(msg_lower) < 20:
            digit_match = re.search(r'\b(\d)\b', msg_lower)
            if digit_match:
                ref = digit_match.group(1)
                logger.debug("Quick reference: matched standalone digit", ref=ref)

        # Check for "that one", "it", "this" -> refers to first card
        # But avoid false positives like "that was"
        if not ref:
            # More precise patterns to avoid false positives
            if re.search(r'\bthat one\b', msg_lower):
                ref = "1"
            elif re.search(r'\bthis one\b', msg_lower):
                ref = "1"
            elif msg_lower.endswith(" it") or " it " in msg_lower or msg_lower == "it":
                ref = "1"
            elif msg_lower.endswith(" that") and "what" not in msg_lower:
                ref = "1"
            if ref:
                logger.debug("Quick reference: matched pronoun pattern", ref=ref)

        if not ref:
            logger.debug("Quick reference: no reference found")
            return None

        logger.debug("Quick reference: checking action patterns", ref=ref)

        # Determine action
        if any(p in msg_lower for p in save_patterns):
            logger.debug("Quick reference: matched save pattern")
            return self._handle_save(ref)
        elif any(p in msg_lower for p in like_patterns):
            logger.debug("Quick reference: matched like pattern")
            return self._handle_like(ref)
        elif any(p in msg_lower for p in dislike_patterns):
            logger.debug("Quick reference: matched dislike pattern")
            return self._handle_dislike(ref)
        elif any(p in msg_lower for p in cooked_patterns):
            logger.debug("Quick reference: matched cooked pattern")
            return self._handle_cooked(ref)
        elif any(p in msg_lower for p in show_patterns):
            logger.debug("Quick reference: matched show pattern")
            return self._handle_show(ref)

        # If we have a reference but no clear action verb, default to show
        # This handles "the first one", "number 1", etc.
        logger.debug("Quick reference: defaulting to show", ref=ref)
        return self._handle_show(ref)

    def _try_intent_classification(self, message: str) -> ChatResult | None:
        """Try to classify message as an intent command.

        Args:
            message: User's message

        Returns:
            ChatResult if intent was handled, None otherwise
        """
        if not self.app_ctx:
            return None

        try:
            from src.chains.intent_classifier import classify_intent

            # Convert last_cards dicts to RecipeCard objects for classifier
            cards_for_classifier = []
            for card in self.last_cards:
                cards_for_classifier.append(RecipeCard(
                    recipe_id=card.get("recipe_id") or card.get("id", ""),
                    title=card.get("title", ""),
                    tags=card.get("tags", []),
                    key_ingredients=card.get("key_ingredients", []),
                    one_sentence_summary=card.get("one_sentence_summary", ""),
                    why_match=card.get("why_match", ""),
                ))

            intent_result = classify_intent(
                message,
                cards_for_classifier,
                self.app_ctx.intent_llm,
            )

            # Skip if conversation or low confidence
            if intent_result.intent == "conversation" or intent_result.confidence == "low":
                return None

            # Execute intent
            return self._execute_intent(intent_result)

        except Exception as e:
            logger.warning("Intent classification failed", error=str(e))
            return None

    def _execute_intent(self, intent_result) -> ChatResult | None:
        """Execute a detected intent.

        Args:
            intent_result: IntentClassification object

        Returns:
            ChatResult if intent was executed, None otherwise
        """
        intent = intent_result.intent
        ref = intent_result.recipe_reference

        # Handle stateless commands
        if intent == "history":
            return ChatResult(
                response=self._get_history_text(),
                command_executed=True,
            )

        if intent == "box":
            return ChatResult(
                response=self._get_recipe_box_text(),
                command_executed=True,
            )

        if intent == "preferences":
            return ChatResult(
                response=self._get_preferences_text(),
                command_executed=True,
            )

        # Handle meal planning intents
        if intent == "mealplan":
            return self._handle_mealplan("")

        if intent == "show_plan":
            return self._handle_show_plan()

        if intent == "grocery_list":
            return self._handle_grocery()

        # Handle recipe-based commands
        if not ref:
            return None

        result = self._resolve_recipe_reference(ref)
        if not result:
            return ChatResult(
                response=f"Recipe not found: {ref}",
                command_executed=True,
            )

        recipe_id, title = result

        if intent == "like":
            self.user_ctx.feedback_store.add_feedback(RecipeFeedback(
                recipe_id=recipe_id,
                feedback_type="like",
            ))
            return ChatResult(
                response=f"Liked: **{title}**",
                command_executed=True,
            )

        if intent == "dislike":
            self.user_ctx.feedback_store.add_feedback(RecipeFeedback(
                recipe_id=recipe_id,
                feedback_type="dislike",
            ))
            return ChatResult(
                response=f"Disliked: **{title}**",
                command_executed=True,
            )

        if intent == "rate":
            rating = intent_result.rating_value or 3
            self.user_ctx.feedback_store.add_feedback(RecipeFeedback(
                recipe_id=recipe_id,
                feedback_type="rate",
                rating=rating,
            ))
            return ChatResult(
                response=f"Rated **{title}**: {rating}/5",
                command_executed=True,
            )

        if intent == "save":
            try:
                self.user_ctx.recipe_box_store.save_recipe(recipe_id, title)
                return ChatResult(
                    response=f"Saved to Recipe Box: **{title}**",
                    command_executed=True,
                )
            except Exception:
                return ChatResult(
                    response=f"Recipe already in your box: **{title}**",
                    command_executed=True,
                )

        if intent == "cooked":
            self.user_ctx.history_store.add_cooked(recipe_id)
            return ChatResult(
                response=f"Marked as cooked: **{title}**",
                command_executed=True,
            )

        if intent == "show":
            return self._handle_show(ref)

        return None

    def _handle_command(self, message: str) -> ChatResult:
        """Handle slash commands.

        Args:
            message: Command string starting with /

        Returns:
            ChatResult with command response
        """
        message_lower = message.lower().strip()

        # /commands or /help
        if message_lower in ("/commands", "/help"):
            return ChatResult(
                response=self._get_help_text(),
                command_executed=True,
            )

        # /new - start new session
        if message_lower == "/new":
            return ChatResult(
                response="Session cleared. How can I help you today?",
                cards=[],
                rolling_summary="",
                command_executed=True,
            )

        # /prefs - show preferences
        if message_lower == "/prefs":
            return ChatResult(
                response=self._get_preferences_text(),
                command_executed=True,
            )

        # /box - show recipe box
        if message_lower == "/box":
            return ChatResult(
                response=self._get_recipe_box_text(),
                command_executed=True,
            )

        # /history - show cooking history
        if message_lower == "/history":
            return ChatResult(
                response=self._get_history_text(),
                command_executed=True,
            )

        # /like <ref>
        if message_lower.startswith("/like"):
            ref = message[5:].strip()
            return self._handle_like(ref)

        # /dislike <ref>
        if message_lower.startswith("/dislike"):
            ref = message[8:].strip()
            return self._handle_dislike(ref)

        # /rate <1-5> <ref>
        if message_lower.startswith("/rate"):
            return self._handle_rate(message[5:].strip())

        # /save <ref>
        if message_lower.startswith("/save"):
            ref = message[5:].strip()
            return self._handle_save(ref)

        # /unsave <ref>
        if message_lower.startswith("/unsave"):
            ref = message[7:].strip()
            return self._handle_unsave(ref)

        # /cooked <ref>
        if message_lower.startswith("/cooked"):
            ref = message[7:].strip()
            return self._handle_cooked(ref)

        # /show <ref>
        if message_lower.startswith("/show"):
            ref = message[5:].strip()
            return self._handle_show(ref)

        # /addpref <type> <value>
        if message_lower.startswith("/addpref"):
            return self._handle_addpref(message[8:].strip())

        # /plan - show meal plan
        if message_lower == "/plan":
            return self._handle_show_plan()

        # /grocery - generate grocery list
        if message_lower == "/grocery":
            return self._handle_grocery()

        # /mealplan - start meal planning
        if message_lower.startswith("/mealplan"):
            input_text = message[9:].strip()  # Get text after /mealplan
            return self._handle_mealplan(input_text)

        # Unknown command
        return ChatResult(
            response=f"Unknown command: {message.split()[0]}. Type /commands for help.",
            command_executed=True,
        )

    def _handle_chat_message(self, message: str, rolling_summary: str | None) -> ChatResult:
        """Handle regular chat messages (non-command).

        Args:
            message: User's message
            rolling_summary: Current conversation context

        Returns:
            ChatResult with LLM response
        """
        # If no app context, return placeholder
        if not self.app_ctx:
            return ChatResult(
                response=(
                    f"I received your message: \"{message}\"\n\n"
                    "Full recipe search and LLM integration coming in a future update.\n\n"
                    "For now, try these commands:\n"
                    "- `/box` - View your saved recipes\n"
                    "- `/history` - View your cooking history\n"
                    "- `/prefs` - View your preferences\n"
                    "- `/commands` - See all available commands"
                ),
                command_executed=False,
            )

        try:
            from src.chains.chat_chain import build_chat_chain
            from src.chains.extractors import ConstraintExtractor
            from src.memory.summarizer import RollingSummarizer
            from src.domain.models import SessionState

            # Load user profile (cached per-request via lazy property)
            profile = self.user_ctx.profile_store.load()

            # Use a default SessionState - we derive context from rolling_summary instead
            # This avoids creating new CLI sessions for each web request
            session = SessionState()

            # Compute exclusion set from feedback and history
            exclude_ids = (
                self.user_ctx.feedback_store.get_liked_recipe_ids(limit=20) |
                self.user_ctx.feedback_store.get_disliked_recipe_ids() |
                self.user_ctx.history_store.get_recently_cooked_ids(days=7)
            )

            # Build the chat chain
            chain = build_chat_chain(
                llm=self.app_ctx.llm,
                retrieval_chain=self.app_ctx.retrieval_chain,
                profile=profile,
                session=session,
                rolling_summary=rolling_summary or "",
                exclude_recipe_ids=exclude_ids,
                llm_clarification=self.app_ctx.llm_clarification,
            )

            # Invoke chain (synchronous)
            result = chain.invoke({"user_input": message})

            # Extract response and cards
            response = result.get("response", "")
            cards = result.get("cards", [])

            # Update rolling summary based on extracted constraints
            summarizer = RollingSummarizer()
            extractor = ConstraintExtractor()
            constraints = extractor.extract_constraints(message)
            new_summary = summarizer.update_summary(rolling_summary or "", constraints, message)

            logger.info(
                "Chat message processed via LLM",
                user_id=self.user_ctx.user_id,
                message_length=len(message),
                response_length=len(response),
                num_cards=len(cards),
            )

            return ChatResult(
                response=response,
                cards=cards,
                rolling_summary=new_summary,
                command_executed=False,
            )

        except Exception as e:
            logger.exception("Error processing chat message", error=str(e))

            # Check for common errors
            error_msg = str(e).lower()
            if "connection" in error_msg or "connect" in error_msg:
                return ChatResult(
                    response="Unable to connect to the LLM service (Ollama). Please ensure Ollama is running.",
                    command_executed=False,
                )

            return ChatResult(
                response=f"Sorry, an error occurred while processing your request: {e}",
                command_executed=False,
            )

    def _resolve_recipe_reference(self, ref: str) -> tuple[str, str] | None:
        """Resolve recipe reference to (recipe_id, title).

        Args:
            ref: Reference (number like "1" or partial name)

        Returns:
            Tuple of (recipe_id, title) or None if not found
        """
        if not ref or not self.last_cards:
            return None

        ref = ref.strip().strip('"\'')

        # Try by number
        try:
            idx = int(ref) - 1
            if 0 <= idx < len(self.last_cards):
                card = self.last_cards[idx]
                return (card.get("recipe_id") or card.get("id"), card.get("title", ""))
        except ValueError:
            pass

        # Try by name (fuzzy match)
        ref_lower = ref.lower()

        # Strip articles for matching
        def strip_articles(text):
            words = text.split()
            if words and words[0].lower() in ("the", "a", "an"):
                return " ".join(words[1:])
            return text

        ref_normalized = strip_articles(ref_lower)

        for card in self.last_cards:
            title = card.get("title", "").lower()
            title_normalized = strip_articles(title)

            # Check if ref is in title OR title is in ref
            if ref_normalized in title_normalized or title_normalized in ref_normalized:
                return (card.get("recipe_id") or card.get("id"), card.get("title", ""))

            # Also check word-subset matching
            ref_words = set(ref_normalized.split())
            title_words = set(title_normalized.split())
            if ref_words and ref_words.issubset(title_words):
                return (card.get("recipe_id") or card.get("id"), card.get("title", ""))

        return None

    def _get_help_text(self) -> str:
        """Get help text for commands."""
        return (
            "**Available Commands**\n\n"
            "**Session:**\n"
            "- `/new` - Start a new session\n"
            "- `/prefs` - Show your preferences\n"
            "- `/addpref <type> <value>` - Add a preference\n"
            "- `/commands` - Show this help\n\n"
            "**Recipe Feedback:**\n"
            "- `/like <ref>` - Like a recipe (by number or name)\n"
            "- `/dislike <ref>` - Dislike a recipe\n"
            "- `/rate <1-5> <ref>` - Rate a recipe 1-5 stars\n"
            "- `/cooked <ref>` - Mark recipe as cooked\n\n"
            "**Recipe Box:**\n"
            "- `/save <ref>` - Save recipe to Recipe Box\n"
            "- `/unsave <ref>` - Remove from Recipe Box\n"
            "- `/box` - View saved recipes\n"
            "- `/show <ref>` - Show full recipe details\n\n"
            "**Meal Planning:**\n"
            "- `/mealplan` - Plan meals for the week\n"
            "- `/plan` - View current meal plan\n"
            "- `/grocery` - Generate grocery list\n\n"
            "**History:**\n"
            "- `/history` - Show cooking history\n\n"
            "**Preference Types:**\n"
            "- `cuisine <name>` - Add preferred cuisine\n"
            "- `avoid <ingredient>` - Avoid an ingredient\n"
            "- `diet <type>` - Set diet (vegetarian, vegan, etc.)\n"
            "- `spice <level>` - Set spice level (none, mild, medium, hot)\n"
            "- `time <minutes>` - Set default cooking time"
        )

    def _get_preferences_text(self) -> str:
        """Get formatted preferences text."""
        profile = self.user_ctx.profile_store.load()

        lines = ["**Your Preferences:**\n"]
        lines.append(f"- Spice level: {profile.spice_level}")
        lines.append(f"- Diet: {profile.diet}")

        if profile.avoid_ingredients:
            lines.append(f"- Avoid: {', '.join(profile.avoid_ingredients)}")

        if profile.preferred_cuisines:
            lines.append(f"- Cuisines: {', '.join(profile.preferred_cuisines)}")

        if profile.time_limit_default_minutes:
            lines.append(f"- Default time: {profile.time_limit_default_minutes} minutes")

        # Learned preferences
        learned = self.user_ctx.feedback_store.get_preferred_cuisines_from_likes(min_count=3)
        if learned:
            lines.append(f"\n*Learned from your likes:* {', '.join(learned)}")

        return "\n".join(lines)

    def _get_recipe_box_text(self) -> str:
        """Get formatted recipe box text."""
        saved = self.user_ctx.recipe_box_store.get_saved_recipes(limit=20)

        if not saved:
            return "Your Recipe Box is empty. Use `/save <ref>` to save recipes."

        lines = ["**Your Recipe Box:**\n"]
        for i, recipe in enumerate(saved, 1):
            date_str = recipe.saved_at.strftime("%m/%d") if recipe.saved_at else ""
            lines.append(f"{i}. {recipe.title} ({date_str})")

        return "\n".join(lines)

    def _get_history_text(self) -> str:
        """Get formatted cooking history text."""
        history = self.user_ctx.history_store.get_cooking_history(limit=10)

        if not history:
            return "No cooking history yet. Use `/cooked <ref>` to track what you make."

        lines = ["**Recent Cooking History:**\n"]

        # Try to get recipe titles
        for entry in history:
            date_str = entry.cooked_at.strftime("%m/%d") if entry.cooked_at else ""
            title = entry.recipe_id  # Default to ID

            # Try to look up title if we have app context
            if self.app_ctx:
                try:
                    from src.ingest.build_db import get_recipe_by_id
                    recipe = get_recipe_by_id(self.app_ctx.db_path, entry.recipe_id)
                    if recipe:
                        title = recipe.title
                except Exception:
                    pass

            lines.append(f"- {title} ({date_str})")

        return "\n".join(lines)

    def _handle_like(self, ref: str) -> ChatResult:
        """Handle /like command."""
        if not ref:
            return ChatResult(
                response="Usage: `/like <number or recipe name>`",
                command_executed=True,
            )

        result = self._resolve_recipe_reference(ref)
        if not result:
            return ChatResult(
                response=f"Recipe not found: {ref}",
                command_executed=True,
            )

        recipe_id, title = result
        self.user_ctx.feedback_store.add_feedback(RecipeFeedback(
            recipe_id=recipe_id,
            feedback_type="like",
        ))

        return ChatResult(
            response=f"Liked: **{title}**",
            command_executed=True,
        )

    def _handle_dislike(self, ref: str) -> ChatResult:
        """Handle /dislike command."""
        if not ref:
            return ChatResult(
                response="Usage: `/dislike <number or recipe name>`",
                command_executed=True,
            )

        result = self._resolve_recipe_reference(ref)
        if not result:
            return ChatResult(
                response=f"Recipe not found: {ref}",
                command_executed=True,
            )

        recipe_id, title = result
        self.user_ctx.feedback_store.add_feedback(RecipeFeedback(
            recipe_id=recipe_id,
            feedback_type="dislike",
        ))

        return ChatResult(
            response=f"Disliked: **{title}**",
            command_executed=True,
        )

    def _handle_rate(self, args: str) -> ChatResult:
        """Handle /rate command."""
        parts = args.split(maxsplit=1)
        if len(parts) < 2:
            return ChatResult(
                response="Usage: `/rate <1-5> <number or recipe name>`",
                command_executed=True,
            )

        try:
            rating = int(parts[0])
            if not 1 <= rating <= 5:
                return ChatResult(
                    response="Rating must be between 1 and 5.",
                    command_executed=True,
                )
        except ValueError:
            return ChatResult(
                response="Rating must be a number 1-5.",
                command_executed=True,
            )

        result = self._resolve_recipe_reference(parts[1])
        if not result:
            return ChatResult(
                response=f"Recipe not found: {parts[1]}",
                command_executed=True,
            )

        recipe_id, title = result
        self.user_ctx.feedback_store.add_feedback(RecipeFeedback(
            recipe_id=recipe_id,
            feedback_type="rate",
            rating=rating,
        ))

        return ChatResult(
            response=f"Rated **{title}**: {rating}/5",
            command_executed=True,
        )

    def _handle_save(self, ref: str) -> ChatResult:
        """Handle /save command."""
        if not ref:
            return ChatResult(
                response="Usage: `/save <number or recipe name>`",
                command_executed=True,
            )

        result = self._resolve_recipe_reference(ref)
        if not result:
            return ChatResult(
                response=f"Recipe not found: {ref}",
                command_executed=True,
            )

        recipe_id, title = result
        try:
            self.user_ctx.recipe_box_store.save_recipe(recipe_id, title)
            return ChatResult(
                response=f"Saved to Recipe Box: **{title}**",
                command_executed=True,
            )
        except Exception:
            return ChatResult(
                response=f"Recipe already in your box: **{title}**",
                command_executed=True,
            )

    def _handle_unsave(self, ref: str) -> ChatResult:
        """Handle /unsave command."""
        if not ref:
            return ChatResult(
                response="Usage: `/unsave <number or recipe name>`",
                command_executed=True,
            )

        result = self._resolve_recipe_reference(ref)
        if not result:
            return ChatResult(
                response=f"Recipe not found: {ref}",
                command_executed=True,
            )

        recipe_id, title = result
        if self.user_ctx.recipe_box_store.remove_recipe(recipe_id):
            return ChatResult(
                response=f"Removed from Recipe Box: **{title}**",
                command_executed=True,
            )
        else:
            return ChatResult(
                response=f"Recipe not in your box: **{title}**",
                command_executed=True,
            )

    def _handle_cooked(self, ref: str) -> ChatResult:
        """Handle /cooked command."""
        if not ref:
            return ChatResult(
                response="Usage: `/cooked <number or recipe name>`",
                command_executed=True,
            )

        result = self._resolve_recipe_reference(ref)
        if not result:
            return ChatResult(
                response=f"Recipe not found: {ref}",
                command_executed=True,
            )

        recipe_id, title = result
        self.user_ctx.history_store.add_cooked(recipe_id)

        return ChatResult(
            response=f"Marked as cooked: **{title}**",
            command_executed=True,
        )

    def _handle_show(self, ref: str) -> ChatResult:
        """Handle /show command."""
        if not ref:
            return ChatResult(
                response="Usage: `/show <number or recipe name>`",
                command_executed=True,
            )

        result = self._resolve_recipe_reference(ref)
        if not result:
            return ChatResult(
                response=f"Recipe not found: {ref}",
                command_executed=True,
            )

        recipe_id, title = result

        # Try to get full recipe from database
        if self.app_ctx:
            try:
                from src.ingest.build_db import get_recipe_by_id
                recipe = get_recipe_by_id(self.app_ctx.db_path, recipe_id)
                if recipe:
                    return ChatResult(
                        response=self._format_full_recipe(recipe),
                        command_executed=True,
                    )
            except Exception as e:
                logger.warning("Failed to load recipe", error=str(e))

        return ChatResult(
            response=f"Recipe: **{title}** (ID: {recipe_id})\n\nFull recipe display requires database lookup.",
            command_executed=True,
        )

    def _format_full_recipe(self, recipe) -> str:
        """Format a recipe for display.

        Args:
            recipe: Recipe object from database

        Returns:
            Formatted recipe string
        """
        lines = [f"**{recipe.title}**\n"]

        if recipe.rating_avg and recipe.rating_count:
            lines.append(f"Rating: {recipe.rating_avg:.1f}/5 ({recipe.rating_count} reviews)")

        if recipe.minutes:
            lines.append(f"Time: {recipe.minutes} minutes")

        lines.append("")

        if recipe.ingredients:
            lines.append("**Ingredients:**")
            for ing in recipe.ingredients:
                lines.append(f"- {ing}")
            lines.append("")

        if recipe.instructions:
            lines.append("**Instructions:**")
            for i, step in enumerate(recipe.instructions, 1):
                lines.append(f"{i}. {step}")

        return "\n".join(lines)

    def _handle_addpref(self, args: str) -> ChatResult:
        """Handle /addpref command."""
        parts = args.split(maxsplit=1)
        if len(parts) < 2:
            return ChatResult(
                response="Usage: `/addpref <type> <value>`\nTypes: cuisine, avoid, diet, spice, time",
                command_executed=True,
            )

        pref_type, value = parts[0].lower(), parts[1].strip()
        profile = self.user_ctx.profile_store.load()

        if pref_type == "cuisine":
            current = list(profile.preferred_cuisines)
            if value.lower() not in [c.lower() for c in current]:
                current.append(value.lower())
                self.user_ctx.profile_store.update(preferred_cuisines=current)
                return ChatResult(
                    response=f"Added cuisine preference: **{value}**",
                    command_executed=True,
                )
            return ChatResult(
                response=f"Already in preferences: {value}",
                command_executed=True,
            )

        elif pref_type == "avoid":
            current = list(profile.avoid_ingredients)
            if value.lower() not in [i.lower() for i in current]:
                current.append(value.lower())
                self.user_ctx.profile_store.update(avoid_ingredients=current)
                return ChatResult(
                    response=f"Will avoid: **{value}**",
                    command_executed=True,
                )
            return ChatResult(
                response=f"Already avoiding: {value}",
                command_executed=True,
            )

        elif pref_type == "diet":
            valid = ["none", "vegetarian", "vegan", "pescatarian", "keto", "gluten_free"]
            if value.lower() in valid:
                self.user_ctx.profile_store.update(diet=value.lower())
                return ChatResult(
                    response=f"Set diet: **{value}**",
                    command_executed=True,
                )
            return ChatResult(
                response=f"Invalid diet. Options: {', '.join(valid)}",
                command_executed=True,
            )

        elif pref_type == "spice":
            valid = ["none", "mild", "medium", "hot"]
            if value.lower() in valid:
                self.user_ctx.profile_store.update(spice_level=value.lower())
                return ChatResult(
                    response=f"Set spice level: **{value}**",
                    command_executed=True,
                )
            return ChatResult(
                response=f"Invalid spice level. Options: {', '.join(valid)}",
                command_executed=True,
            )

        elif pref_type == "time":
            try:
                minutes = int(value)
                if minutes > 0:
                    self.user_ctx.profile_store.update(time_limit_default_minutes=minutes)
                    return ChatResult(
                        response=f"Set default time limit: **{minutes} minutes**",
                        command_executed=True,
                    )
                return ChatResult(response="Time must be positive.", command_executed=True)
            except ValueError:
                return ChatResult(response="Time must be a number (minutes).", command_executed=True)

        return ChatResult(
            response=f"Unknown preference type: {pref_type}\nTypes: cuisine, avoid, diet, spice, time",
            command_executed=True,
        )

    def _handle_mealplan(self, input_text: str) -> ChatResult:
        """Handle /mealplan command to generate a meal plan.

        Args:
            input_text: Optional constraints (e.g., "5 vegetarian dinners")

        Returns:
            ChatResult with generated meal plan
        """
        from collections import defaultdict
        from datetime import date, timedelta

        try:
            from src.planning.constraint_extractor import MealPlanConstraintExtractor
            from src.planning.meal_planner import MealPlanner
            from src.domain.models import MealPlan
            from src.ingest.build_db import get_all_recipes

            # Get database path
            db_path = self.user_ctx.db_path
            if self.app_ctx:
                db_path = self.app_ctx.db_path

            # Load user profile
            profile = self.user_ctx.profile_store.load()

            # Extract constraints from input
            extractor = MealPlanConstraintExtractor(db_path=db_path)
            constraints = extractor.extract(
                input_text if input_text else "plan dinners",
                profile=profile
            )

            # Build response with extracted constraints
            lines = ["**Meal Planning**\n"]
            lines.append(f"Planning {constraints.days} {constraints.meal_types[0] if constraints.meal_types else 'dinner'}(s)")

            if constraints.dietary.value != "none":
                lines.append(f"- Diet: {constraints.dietary.value}")
            if constraints.max_prep_time:
                lines.append(f"- Max time: {constraints.max_prep_time} minutes")
            if constraints.excluded_categories:
                lines.append(f"- Excluding: {', '.join(c.value for c in constraints.excluded_categories)}")
            if constraints.excluded_tags:
                lines.append(f"- No: {', '.join(constraints.excluded_tags)}")

            lines.append("\nGenerating plan...")

            # Get saved recipes from Recipe Box
            saved_recipes = self.user_ctx.recipe_box_store.get_saved_recipes(limit=100)
            box_recipe_ids = {r.recipe_id for r in saved_recipes}

            # Fetch candidate recipes from database
            all_recipes = get_all_recipes(db_path, limit=500)

            if not all_recipes:
                return ChatResult(
                    response="No recipes found in database. Please ensure the database is set up.",
                    command_executed=True,
                )

            # Initialize planner and generate plan
            planner = MealPlanner()
            meals, metrics = planner.generate_plan(
                all_recipes, constraints, profile, box_recipe_ids=box_recipe_ids
            )

            if not meals:
                return ChatResult(
                    response="No recipes found matching your criteria. Try relaxing some constraints.",
                    command_executed=True,
                )

            # Create meal plan object
            start_date = constraints.start_date or date.today()
            end_date = start_date + timedelta(days=constraints.days - 1)

            plan = MealPlan(
                start_date=start_date,
                end_date=end_date,
                meal_types=constraints.meal_types or ["dinner"],
                status="draft",
                constraints=constraints.model_dump(exclude={"extraction_sources"}),
                metrics=metrics,
                meals=meals
            )

            # Save plan
            plan_id = self.user_ctx.meal_plan_store.create_plan(plan)
            plan.id = plan_id

            # Build response
            lines = [f"**Meal Plan Generated** (ID: {plan_id})\n"]

            # Group meals by day
            meals_by_day = defaultdict(list)
            for meal in meals:
                meals_by_day[meal.day].append(meal)

            for day in sorted(meals_by_day.keys()):
                day_meals = meals_by_day[day]
                day_str = day.strftime("%A, %b %d")
                lines.append(f"**{day_str}**")
                for meal in day_meals:
                    source_icon = "📦" if meal.source == "box" else "🔍"
                    lines.append(f"  {source_icon} {meal.title}")

            # Show metrics
            lines.append(f"\n*Metrics:*")
            lines.append(f"- Unique ingredients: {metrics.unique_ingredients}")
            lines.append(f"- Ingredient overlap: {metrics.overlap_ratio:.0%}")
            lines.append(f"- From Recipe Box: {metrics.box_recipe_count}, Discovery: {metrics.discovery_recipe_count}")

            if metrics.top_shared_ingredients:
                shared = ", ".join(f"{ing}" for ing, _ in metrics.top_shared_ingredients[:5])
                lines.append(f"- Common ingredients: {shared}")

            lines.append(f"\n*Use `/plan` to view, `/grocery` for shopping list*")

            return ChatResult(
                response="\n".join(lines),
                command_executed=True,
            )

        except Exception as e:
            logger.exception("Meal plan generation error", error=str(e))
            return ChatResult(
                response=f"Error generating meal plan: {e}",
                command_executed=True,
            )

    def _handle_show_plan(self) -> ChatResult:
        """Handle /plan command."""
        from collections import defaultdict

        # Get most recent plan
        plans = self.user_ctx.meal_plan_store.get_recent_plans(limit=1)
        if not plans:
            return ChatResult(
                response="No meal plans found. Use `/mealplan` to create one.",
                command_executed=True,
            )

        plan = plans[0]

        lines = [f"**Meal Plan** (ID: {plan.id})"]
        lines.append(f"Status: {plan.status}")
        lines.append(f"Period: {plan.start_date} to {plan.end_date}\n")

        if not plan.meals:
            lines.append("*Plan has no meals yet.*")
            return ChatResult(
                response="\n".join(lines),
                command_executed=True,
            )

        # Group meals by day
        meals_by_day = defaultdict(list)
        for meal in plan.meals:
            meals_by_day[meal.day].append(meal)

        for day in sorted(meals_by_day.keys()):
            day_meals = meals_by_day[day]
            day_str = day.strftime("%A, %b %d")
            lines.append(f"**{day_str}**")
            for meal in day_meals:
                source_icon = "📦" if meal.source == "box" else "🔍"
                lines.append(f"  {source_icon} {meal.title}")

        if plan.metrics:
            lines.append(f"\n*Metrics: {plan.metrics.unique_ingredients} ingredients, "
                        f"{plan.metrics.overlap_ratio:.0%} overlap*")

        return ChatResult(
            response="\n".join(lines),
            command_executed=True,
        )

    def _handle_grocery(self) -> ChatResult:
        """Handle /grocery command."""
        try:
            from src.planning.grocery_list import GroceryListGenerator
            from src.ingest.build_db import get_recipe_by_id

            # Get database path
            db_path = self.user_ctx.db_path
            if self.app_ctx:
                db_path = self.app_ctx.db_path

            # Get most recent plan
            plans = self.user_ctx.meal_plan_store.get_recent_plans(limit=1)
            if not plans:
                return ChatResult(
                    response="No meal plans found. Use `/mealplan` to create one first.",
                    command_executed=True,
                )

            plan = plans[0]
            if not plan.meals:
                return ChatResult(
                    response="Your meal plan has no meals yet.",
                    command_executed=True,
                )

            # Fetch full recipes
            recipes = {}
            for meal in plan.meals:
                recipe = get_recipe_by_id(db_path, meal.recipe_id)
                if recipe:
                    recipes[meal.recipe_id] = recipe

            if not recipes:
                return ChatResult(
                    response="Could not load recipes for the meal plan.",
                    command_executed=True,
                )

            # Generate grocery list
            generator = GroceryListGenerator()
            grocery_list = generator.generate(plan, recipes, exclude_pantry_staples=True)

            if not grocery_list.items:
                return ChatResult(
                    response="No items in grocery list (all ingredients may be pantry staples).",
                    command_executed=True,
                )

            # Format for display
            formatted = generator.format_for_display(
                grocery_list, show_recipes=True, group_by_category=True
            )

            lines = [f"**Grocery List**"]
            lines.append(f"For meal plan {plan.start_date} to {plan.end_date}\n")
            lines.append(formatted)

            # Add summary
            summary = generator.get_summary(grocery_list)
            lines.append(f"\n*{summary['total_items']} items across {summary['category_count']} categories*")

            return ChatResult(
                response="\n".join(lines),
                command_executed=True,
            )

        except Exception as e:
            logger.exception("Grocery list generation error", error=str(e))
            return ChatResult(
                response=f"Error generating grocery list: {e}",
                command_executed=True,
            )

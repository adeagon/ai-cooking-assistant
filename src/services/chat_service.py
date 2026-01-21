"""Chat service for processing messages in web and CLI contexts."""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from src.app.logging_config import get_logger
from src.domain.models import PreferenceProfile, RecipeCard, RecipeFeedback, SavedRecipe

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

    def __init__(self, user_ctx: UserContext, last_cards: list[dict] | None = None):
        """Initialize chat service.

        Args:
            user_ctx: User context with stores
            last_cards: Previous recipe cards for reference resolution
        """
        self.user_ctx = user_ctx
        self.last_cards = last_cards or []

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

        # Regular chat message - for now return a placeholder
        # TODO: Integrate with LLM retrieval chain
        return self._handle_chat_message(message, rolling_summary)

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
            return ChatResult(
                response="Meal planning requires LLM integration. Coming soon!",
                command_executed=True,
            )

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
        # TODO: Integrate with LLM retrieval chain
        # For now, return a helpful placeholder
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
        for card in self.last_cards:
            title = card.get("title", "").lower()
            if ref_lower in title or title in ref_lower:
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
        for entry in history:
            date_str = entry.cooked_at.strftime("%m/%d") if entry.cooked_at else ""
            lines.append(f"- {entry.recipe_id} ({date_str})")

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
                response=f"Recipe not found: {ref}. Full recipe display requires database lookup.",
                command_executed=True,
            )

        recipe_id, title = result
        # TODO: Look up full recipe from database
        return ChatResult(
            response=f"Recipe: **{title}** (ID: {recipe_id})\n\nFull recipe display coming soon.",
            command_executed=True,
        )

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

    def _handle_show_plan(self) -> ChatResult:
        """Handle /plan command."""
        plan = self.user_ctx.meal_plan_store.get_active_plan()

        if not plan:
            return ChatResult(
                response="No active meal plan. Use `/mealplan` to create one.",
                command_executed=True,
            )

        lines = [f"**Meal Plan: {plan.name or 'Untitled'}**\n"]
        lines.append(f"*{plan.start_date} to {plan.end_date}*\n")

        for meal in sorted(plan.meals, key=lambda m: (m.day, m.position)):
            lines.append(f"- {meal.day.strftime('%a %m/%d')}: {meal.title}")

        return ChatResult(
            response="\n".join(lines),
            command_executed=True,
        )

    def _handle_grocery(self) -> ChatResult:
        """Handle /grocery command."""
        plan = self.user_ctx.meal_plan_store.get_active_plan()

        if not plan:
            return ChatResult(
                response="No active meal plan. Use `/mealplan` to create one first.",
                command_executed=True,
            )

        # TODO: Generate actual grocery list from plan
        return ChatResult(
            response=f"Grocery list for meal plan '{plan.name or 'Untitled'}' coming soon.",
            command_executed=True,
        )

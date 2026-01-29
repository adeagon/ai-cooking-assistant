"""Conversation service for managing chat threads and messages."""

import json
import uuid
from datetime import datetime
from pathlib import Path

from src.app.logging_config import get_logger
from src.web.db import get_db_connection
from src.web.models import (
    ConversationSummary,
    Message,
    MessageMeta,
    RecipeCardMeta,
)

logger = get_logger(__name__)


class ConversationService:
    """Service for managing conversations and messages.

    Handles conversation CRUD and message persistence with
    transactional updates to conversation timestamps.
    """

    def __init__(self, db_path: Path):
        """Initialize conversation service.

        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path

    def create(self, user_id: str, title: str | None = None) -> str:
        """Create a new conversation.

        Args:
            user_id: User UUID
            title: Optional conversation title

        Returns:
            Created conversation ID
        """
        conv_id = str(uuid.uuid4())
        now = datetime.now()

        with get_db_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, user_id, title, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (conv_id, user_id, title, now.isoformat())
            )
            conn.commit()
            logger.info(
                "Created conversation",
                conversation_id=conv_id[:8],
                user_id=user_id[:8]
            )

        return conv_id

    def get(self, conversation_id: str, user_id: str) -> ConversationSummary | None:
        """Get a conversation by ID, verifying user ownership.

        Args:
            conversation_id: Conversation ID
            user_id: User UUID (for ownership verification)

        Returns:
            Conversation if found and owned by user, None otherwise
        """
        with get_db_connection(self.db_path) as conn:
            row = conn.execute(
                """
                SELECT id, title, created_at, last_message_at
                FROM conversations
                WHERE id = ? AND user_id = ? AND archived_at IS NULL
                """,
                (conversation_id, user_id)
            ).fetchone()

            if row:
                return self._row_to_summary(row)
            return None

    def list_for_user(
        self,
        user_id: str,
        limit: int = 50
    ) -> list[ConversationSummary]:
        """List conversations for a user, sorted by last message time.

        Args:
            user_id: User UUID
            limit: Maximum number of conversations to return

        Returns:
            List of conversation summaries
        """
        with get_db_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, title, created_at, last_message_at
                FROM conversations
                WHERE user_id = ? AND archived_at IS NULL
                ORDER BY COALESCE(last_message_at, created_at) DESC
                LIMIT ?
                """,
                (user_id, limit)
            ).fetchall()

            return [self._row_to_summary(row) for row in rows]

    def update_title(
        self,
        conversation_id: str,
        user_id: str,
        title: str
    ) -> bool:
        """Update conversation title.

        Args:
            conversation_id: Conversation ID
            user_id: User UUID (for ownership verification)
            title: New title

        Returns:
            True if updated, False if not found
        """
        with get_db_connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE conversations
                SET title = ?
                WHERE id = ? AND user_id = ? AND archived_at IS NULL
                """,
                (title, conversation_id, user_id)
            )
            conn.commit()
            return cursor.rowcount > 0

    def archive(self, conversation_id: str, user_id: str) -> bool:
        """Archive (soft-delete) a conversation.

        Args:
            conversation_id: Conversation ID
            user_id: User UUID (for ownership verification)

        Returns:
            True if archived, False if not found
        """
        now = datetime.now()
        with get_db_connection(self.db_path) as conn:
            cursor = conn.execute(
                """
                UPDATE conversations
                SET archived_at = ?
                WHERE id = ? AND user_id = ? AND archived_at IS NULL
                """,
                (now.isoformat(), conversation_id, user_id)
            )
            conn.commit()
            if cursor.rowcount > 0:
                logger.info(
                    "Archived conversation",
                    conversation_id=conversation_id[:8]
                )
                return True
            return False

    def add_message(
        self,
        conversation_id: str,
        user_id: str,
        role: str,
        content: str,
        meta: dict | None = None
    ) -> str:
        """Add a message to a conversation.

        Updates last_message_at in the same transaction.

        Args:
            conversation_id: Conversation ID
            user_id: User UUID
            role: Message role ('user' or 'assistant')
            content: Message content
            meta: Optional metadata (recipe_cards, error, etc.)

        Returns:
            Created message ID
        """
        msg_id = str(uuid.uuid4())
        now = datetime.now()
        meta_json = json.dumps(meta or {})

        with get_db_connection(self.db_path) as conn:
            # Insert message
            conn.execute(
                """
                INSERT INTO messages
                    (id, conversation_id, user_id, role, content, meta_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (msg_id, conversation_id, user_id, role, content, meta_json, now.isoformat())
            )

            # Update conversation timestamp
            conn.execute(
                """
                UPDATE conversations
                SET last_message_at = ?
                WHERE id = ?
                """,
                (now.isoformat(), conversation_id)
            )

            conn.commit()
            logger.debug(
                "Added message",
                message_id=msg_id[:8],
                conversation_id=conversation_id[:8],
                role=role
            )

        return msg_id

    def get_messages(
        self,
        conversation_id: str,
        user_id: str,
        limit: int = 100,
        before_id: str | None = None
    ) -> list[Message]:
        """Get messages for a conversation.

        Args:
            conversation_id: Conversation ID
            user_id: User UUID (for ownership verification)
            limit: Maximum number of messages
            before_id: Optional message ID for pagination

        Returns:
            List of messages, oldest first
        """
        with get_db_connection(self.db_path) as conn:
            # First verify user owns the conversation
            conv = conn.execute(
                "SELECT id FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, user_id)
            ).fetchone()

            if not conv:
                return []

            if before_id:
                rows = conn.execute(
                    """
                    SELECT id, conversation_id, user_id, role, content,
                           meta_json, created_at
                    FROM messages
                    WHERE conversation_id = ?
                      AND created_at < (
                          SELECT created_at FROM messages WHERE id = ?
                      )
                    ORDER BY created_at ASC
                    LIMIT ?
                    """,
                    (conversation_id, before_id, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT id, conversation_id, user_id, role, content,
                           meta_json, created_at
                    FROM messages
                    WHERE conversation_id = ?
                    ORDER BY created_at ASC
                    LIMIT ?
                    """,
                    (conversation_id, limit)
                ).fetchall()

            return [self._row_to_message(row) for row in rows]

    def get_recent_messages(
        self,
        conversation_id: str,
        limit: int = 10
    ) -> list[Message]:
        """Get recent messages for building chat context.

        Does not verify user ownership - use for internal service calls.

        Args:
            conversation_id: Conversation ID
            limit: Number of recent messages

        Returns:
            List of messages, oldest first
        """
        with get_db_connection(self.db_path) as conn:
            rows = conn.execute(
                """
                SELECT id, conversation_id, user_id, role, content,
                       meta_json, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (conversation_id, limit)
            ).fetchall()

            # Reverse to get oldest first
            messages = [self._row_to_message(row) for row in rows]
            return list(reversed(messages))

    def _row_to_summary(self, row) -> ConversationSummary:
        """Convert database row to ConversationSummary.

        Args:
            row: SQLite Row object

        Returns:
            ConversationSummary model
        """
        def parse_datetime(val):
            if val is None:
                return None
            if isinstance(val, str):
                return datetime.fromisoformat(val)
            return val

        return ConversationSummary(
            id=row["id"],
            title=row["title"],
            created_at=parse_datetime(row["created_at"]),
            last_message_at=parse_datetime(row["last_message_at"])
        )

    def _row_to_message(self, row) -> Message:
        """Convert database row to Message model.

        Args:
            row: SQLite Row object

        Returns:
            Message model
        """
        created_at = row["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)

        # Parse meta_json
        meta_dict = json.loads(row["meta_json"] or "{}")
        recipe_cards = [
            RecipeCardMeta(**card)
            for card in meta_dict.get("recipe_cards", [])
        ]
        meta = MessageMeta(
            recipe_cards=recipe_cards,
            intent=meta_dict.get("intent"),
            error=meta_dict.get("error")
        )

        return Message(
            id=row["id"],
            conversation_id=row["conversation_id"],
            user_id=row["user_id"],
            role=row["role"],
            content=row["content"],
            meta=meta,
            created_at=created_at
        )

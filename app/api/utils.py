"""
Utilities for session management and API functionality.
"""
from typing import Dict, Any, Optional, List

from app.graph.data_types import Book
from app.api.database import DatabaseService
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SessionManager:
    """Manages user sessions and their data using SQLite database."""

    def __init__(self, max_sessions: int = 1000, timeout_hours: int = 24, db_path: str = "sessions.db"):
        self.max_sessions = max_sessions
        self.timeout_hours = timeout_hours
        self.db = DatabaseService(db_path)
        logger.info(f"SessionManager initialized with max_sessions={max_sessions}, timeout_hours={timeout_hours}")

    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new session."""
        # Clean up old sessions if at limit
        if self._get_session_count() >= self.max_sessions:
            logger.warning(f"Session limit reached ({self.max_sessions}), cleaning up old sessions")
            self._cleanup_old_sessions()

        session_id = self.db.create_session(session_id)
        logger.info(f"New session created: {session_id}")
        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        session_data = self.db.get_session(session_id)
        if session_data:
            logger.debug(f"Session data retrieved for: {session_id}")
        else:
            logger.warning(f"Session not found: {session_id}")
        return session_data

    def update_session(self, session_id: str, **kwargs):
        """Update session data."""
        logger.debug(f"Updating session {session_id} with data: {list(kwargs.keys())}")
        self.db.update_session(session_id, **kwargs)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        success = self.db.delete_session(session_id)
        if success:
            logger.info(f"Session deleted: {session_id}")
        else:
            logger.warning(f"Failed to delete session: {session_id}")
        return success

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions."""
        sessions = self.db.list_sessions()
        logger.debug(f"Listed {len(sessions)} active sessions")
        return sessions

    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to session."""
        logger.debug(f"Adding message to session {session_id}: role={role}")
        self.db.add_message(session_id, role, content)

    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        stats = self.db.get_session_stats()
        logger.debug(f"Session statistics retrieved: {stats}")
        return stats

    def _cleanup_old_sessions(self):
        """Remove sessions older than timeout."""
        deleted_count = self.db.cleanup_old_sessions(self.timeout_hours)
        logger.info(f"Cleaned up {deleted_count} old sessions")
        return deleted_count

    def _get_session_count(self) -> int:
        """Get current session count."""
        count = self.db.get_session_count()
        return count


def format_response(assistant_response: str, recommended_books: List[Book]) -> str:
    """
    Format the assistant response with book recommendations.

    Args:
        assistant_response: The base response from the assistant
        recommended_books: List of recommended books to include

    Returns:
        Formatted response string with book recommendations
    """
    if not recommended_books:
        logger.debug("No books to format in response")
        return assistant_response

    logger.debug(f"Formatting response with {len(recommended_books)} book recommendations")

    # Add book recommendations to the response
    books_section = "\n\n📚 **Recommended Books:**\n"
    for i, book in enumerate(recommended_books[:5], 1):  # Limit to 5 books
        books_section += f"{i}. **{book.name}** by {book.author}\n"
        if book.description:
            books_section += f"   _{book.description[:100]}..._\n"
        books_section += "\n"

    return assistant_response + books_section


def sanitize_input(text: str, max_length: int = 1000) -> str:
    """
    Sanitize user input by removing potential harmful content and limiting length.

    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized text
    """
    if not text:
        logger.warning("Empty text provided for sanitization")
        return ""

    # Remove potential harmful patterns
    import re

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Remove potential SQL injection patterns
    dangerous_patterns = [
        r'(DROP|DELETE|INSERT|UPDATE|SELECT|UNION|ALTER|CREATE)\s+',
        r'(--|\#|\/\*|\*\/)',
        r'(\bor\b|\band\b)\s+\d+\s*=\s*\d+'
    ]

    for pattern in dangerous_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Limit length
    if len(text) > max_length:
        logger.warning(f"Input text truncated from {len(text)} to {max_length} characters")
        text = text[:max_length]

    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    logger.debug(f"Input sanitized: length={len(text)}")
    return text

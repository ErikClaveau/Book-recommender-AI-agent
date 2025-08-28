"""
Utilities for session management and API functionality.
"""
from typing import Dict, Any, Optional, List

from app.graph.data_types import Book
from app.api.database import DatabaseService


class SessionManager:
    """Manages user sessions and their data using SQLite database."""

    def __init__(self, max_sessions: int = 1000, timeout_hours: int = 24, db_path: str = "sessions.db"):
        self.max_sessions = max_sessions
        self.timeout_hours = timeout_hours
        self.db = DatabaseService(db_path)

    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new session."""
        # Clean up old sessions if at limit
        if self._get_session_count() >= self.max_sessions:
            self._cleanup_old_sessions()

        return self.db.create_session(session_id)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        return self.db.get_session(session_id)

    def update_session(self, session_id: str, **kwargs):
        """Update session data."""
        self.db.update_session(session_id, **kwargs)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        return self.db.delete_session(session_id)

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions."""
        return self.db.list_sessions()

    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to session."""
        self.db.add_message(session_id, role, content)

    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        return self.db.get_session_stats()

    def _cleanup_old_sessions(self):
        """Remove sessions older than timeout."""
        return self.db.cleanup_old_sessions(self.timeout_hours)

    def _get_session_count(self) -> int:
        """Get total number of sessions."""
        stats = self.db.get_session_stats()
        return stats["total_sessions"]


def format_response(message: str, books: List[Book] = None) -> str:
    """Format API response message with optional book list."""
    if not books:
        return message

    book_list = "\n\n📚 Recommended Books:\n"
    for i, book in enumerate(books, 1):
        book_list += f"{i}. **{book.name}** by {book.author}\n"

    return f"{message}{book_list}"


def sanitize_input(text: str, max_length: int = 1000) -> str:
    """Sanitize user input."""
    if not isinstance(text, str):
        return ""

    # Remove potentially harmful characters
    sanitized = text.strip()

    # Truncate if too long
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]

    return sanitized

"""
Utilities for session management and API functionality.
"""
import json
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
from uuid import uuid4

from app.graph.data_types import Book


class SessionManager:
    """Manages user sessions and their data."""

    def __init__(self, max_sessions: int = 1000, timeout_hours: int = 24):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.max_sessions = max_sessions
        self.timeout_hours = timeout_hours

    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new session."""
        if session_id is None:
            session_id = str(uuid4())

        # Clean up old sessions if at limit
        if len(self.sessions) >= self.max_sessions:
            self._cleanup_old_sessions()

        self.sessions[session_id] = {
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "messages": [],
            "recommended_books": [],
            "read_books": [],
            "preferences": [],
            "message_count": 0,
            "recommendation_count": 0
        }

        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data."""
        if session_id not in self.sessions:
            return None

        # Update last activity
        self.sessions[session_id]["last_activity"] = datetime.utcnow().isoformat()
        return self.sessions[session_id]

    def update_session(self, session_id: str, **kwargs):
        """Update session data."""
        if session_id in self.sessions:
            self.sessions[session_id].update(kwargs)
            self.sessions[session_id]["last_activity"] = datetime.utcnow().isoformat()

    def delete_session(self, session_id: str) -> bool:
        """Delete a session."""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions."""
        sessions_info = []
        for session_id, data in self.sessions.items():
            sessions_info.append({
                "session_id": session_id,
                "created_at": data["created_at"],
                "last_activity": data["last_activity"],
                "message_count": data.get("message_count", 0),
                "recommendation_count": data.get("recommendation_count", 0)
            })
        return sessions_info

    def _cleanup_old_sessions(self):
        """Remove sessions older than timeout."""
        cutoff = datetime.utcnow() - timedelta(hours=self.timeout_hours)
        expired_sessions = []

        for session_id, data in self.sessions.items():
            last_activity = datetime.fromisoformat(data["last_activity"])
            if last_activity < cutoff:
                expired_sessions.append(session_id)

        for session_id in expired_sessions:
            del self.sessions[session_id]


def format_response(message: str, books: List[Book] = None) -> str:
    """Format API response message with optional book list."""
    if not books:
        return message

    book_list = "\n\n📚 Recommended Books:\n"
    for i, book in enumerate(books, 1):
        book_list += f"{i}. **{book.name}** by {book.author}\n"

    return f"{message}{book_list}"


def parse_books_from_text(text: str) -> List[Book]:
    """Parse book references from text."""
    # This is a simple implementation - could be enhanced with NLP
    books = []
    lines = text.split('\n')

    for line in lines:
        # Look for patterns like "Title by Author" or "Title - Author"
        if ' by ' in line:
            parts = line.split(' by ')
            if len(parts) == 2:
                title = parts[0].strip()
                author = parts[1].strip()
                books.append(Book(name=title, author=author))
        elif ' - ' in line and len(line.split(' - ')) == 2:
            parts = line.split(' - ')
            title = parts[0].strip()
            author = parts[1].strip()
            books.append(Book(name=title, author=author))

    return books


def sanitize_input(text: str, max_length: int = 1000) -> str:
    """Sanitize user input."""
    if not text:
        return ""

    # Remove excessive whitespace
    text = " ".join(text.split())

    # Truncate if too long
    if len(text) > max_length:
        text = text[:max_length]

    return text.strip()


def extract_preferences_from_text(text: str) -> List[str]:
    """Extract reading preferences from user text."""
    text_lower = text.lower()
    preferences = []

    # Genre keywords
    genres = [
        "fiction", "non-fiction", "mystery", "thriller", "romance", "fantasy",
        "science fiction", "sci-fi", "biography", "history", "horror", "comedy",
        "drama", "adventure", "crime", "historical", "contemporary", "classic"
    ]

    for genre in genres:
        if genre in text_lower:
            preferences.append(genre.title())

    # Other preference keywords
    if "short" in text_lower:
        preferences.append("Short books")
    if "long" in text_lower:
        preferences.append("Long books")
    if "series" in text_lower:
        preferences.append("Book series")
    if "standalone" in text_lower:
        preferences.append("Standalone books")

    return list(set(preferences))  # Remove duplicates

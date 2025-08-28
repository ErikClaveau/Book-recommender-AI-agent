"""
Database service for managing session data with SQLite.
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List
from uuid import uuid4

from app.graph.data_types import Book


class DatabaseService:
    """Service for managing SQLite database operations."""

    def __init__(self, db_path: str = "sessions.db"):
        """Initialize database service."""
        self.db_path = db_path
        self.init_database()

    def init_database(self):
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Create sessions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    last_activity TEXT NOT NULL,
                    message_count INTEGER DEFAULT 0,
                    recommendation_count INTEGER DEFAULT 0
                )
            """)

            # Create session_messages table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                )
            """)

            # Create session_books table for recommended and read books
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_books (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    book_type TEXT NOT NULL CHECK (book_type IN ('recommended', 'read')),
                    book_name TEXT NOT NULL,
                    author TEXT NOT NULL,
                    description TEXT,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                )
            """)

            # Create session_preferences table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    preference TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                )
            """)

            # Create indexes for better performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions (last_activity)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_messages_session_id ON session_messages (session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_books_session_id ON session_books (session_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_preferences_session_id ON session_preferences (session_id)")

            conn.commit()

    @contextmanager
    def _get_connection(self):
        """Get database connection with context manager."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        try:
            yield conn
        finally:
            conn.close()

    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new session."""
        if session_id is None:
            session_id = str(uuid4())

        timestamp = datetime.now(timezone.utc).isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO sessions 
                (session_id, created_at, last_activity, message_count, recommendation_count)
                VALUES (?, ?, ?, 0, 0)
            """, (session_id, timestamp, timestamp))
            conn.commit()

        return session_id

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data with all related information."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Get basic session info
            cursor.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,))
            session_row = cursor.fetchone()

            if not session_row:
                return None

            # Update last activity
            cursor.execute(
                "UPDATE sessions SET last_activity = ? WHERE session_id = ?",
                (datetime.now(timezone.utc).isoformat(), session_id)
            )
            conn.commit()

            # Get messages
            cursor.execute("""
                SELECT role, content, timestamp FROM session_messages 
                WHERE session_id = ? ORDER BY timestamp ASC
            """, (session_id,))
            messages = [{"role": row["role"], "content": row["content"], "timestamp": row["timestamp"]}
                       for row in cursor.fetchall()]

            # Get books
            cursor.execute("""
                SELECT book_type, book_name, author, description FROM session_books 
                WHERE session_id = ?
            """, (session_id,))
            books_data = cursor.fetchall()

            recommended_books = []
            read_books = []
            for book in books_data:
                book_obj = Book(
                    name=book["book_name"],
                    author=book["author"],
                    description=book["description"] if book["description"] else ""
                )
                if book["book_type"] == "recommended":
                    recommended_books.append(book_obj)
                else:
                    read_books.append(book_obj)

            # Get preferences
            cursor.execute("""
                SELECT preference FROM session_preferences 
                WHERE session_id = ? ORDER BY timestamp ASC
            """, (session_id,))
            preferences = [row["preference"] for row in cursor.fetchall()]

            return {
                "session_id": session_row["session_id"],
                "created_at": session_row["created_at"],
                "last_activity": session_row["last_activity"],
                "message_count": session_row["message_count"],
                "recommendation_count": session_row["recommendation_count"],
                "messages": messages,
                "recommended_books": recommended_books,
                "read_books": read_books,
                "preferences": preferences
            }

    def update_session(self, session_id: str, **kwargs):
        """Update session data."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Update basic session info
            update_fields = []
            params = []

            if "message_count" in kwargs:
                update_fields.append("message_count = ?")
                params.append(kwargs["message_count"])

            if "recommendation_count" in kwargs:
                update_fields.append("recommendation_count = ?")
                params.append(kwargs["recommendation_count"])

            update_fields.append("last_activity = ?")
            params.append(datetime.now(timezone.utc).isoformat())
            params.append(session_id)

            if update_fields[:-1]:  # If there are fields to update besides last_activity
                cursor.execute(f"""
                    UPDATE sessions SET {', '.join(update_fields)}
                    WHERE session_id = ?
                """, params)

            # Handle books
            if "recommended_books" in kwargs:
                # Remove old recommended books
                cursor.execute(
                    "DELETE FROM session_books WHERE session_id = ? AND book_type = 'recommended'",
                    (session_id,)
                )
                # Add new recommended books
                timestamp = datetime.now(timezone.utc).isoformat()
                for book in kwargs["recommended_books"]:
                    cursor.execute("""
                        INSERT INTO session_books 
                        (session_id, book_type, book_name, author, description, timestamp)
                        VALUES (?, 'recommended', ?, ?, ?, ?)
                    """, (session_id, book.name, book.author, book.description, timestamp))

            if "read_books" in kwargs:
                # Remove old read books
                cursor.execute(
                    "DELETE FROM session_books WHERE session_id = ? AND book_type = 'read'",
                    (session_id,)
                )
                # Add new read books
                timestamp = datetime.now(timezone.utc).isoformat()
                for book in kwargs["read_books"]:
                    cursor.execute("""
                        INSERT INTO session_books 
                        (session_id, book_type, book_name, author, description, timestamp)
                        VALUES (?, 'read', ?, ?, ?, ?)
                    """, (session_id, book.name, book.author, book.description, timestamp))

            # Handle preferences
            if "preferences" in kwargs:
                # Remove old preferences
                cursor.execute("DELETE FROM session_preferences WHERE session_id = ?", (session_id,))
                # Add new preferences
                timestamp = datetime.now(timezone.utc).isoformat()
                for pref in kwargs["preferences"]:
                    cursor.execute("""
                        INSERT INTO session_preferences (session_id, preference, timestamp)
                        VALUES (?, ?, ?)
                    """, (session_id, pref, timestamp))

            conn.commit()

    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO session_messages (session_id, role, content, timestamp)
                VALUES (?, ?, ?, ?)
            """, (session_id, role, content, datetime.now(timezone.utc).isoformat()))
            conn.commit()

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all related data."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            deleted = cursor.rowcount > 0
            conn.commit()
            return deleted

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT session_id, created_at, last_activity, message_count, recommendation_count
                FROM sessions ORDER BY last_activity DESC
            """)
            return [dict(row) for row in cursor.fetchall()]

    def cleanup_old_sessions(self, timeout_hours: int = 24):
        """Remove sessions older than timeout."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=timeout_hours)).isoformat()

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM sessions WHERE last_activity < ?", (cutoff,))
            deleted_count = cursor.rowcount
            conn.commit()
            return deleted_count

    def get_session_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Total sessions
            cursor.execute("SELECT COUNT(*) as total FROM sessions")
            total_sessions = cursor.fetchone()["total"]

            # Active sessions (last 24 hours)
            cutoff = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
            cursor.execute("SELECT COUNT(*) as active FROM sessions WHERE last_activity > ?", (cutoff,))
            active_sessions = cursor.fetchone()["active"]

            # Total messages
            cursor.execute("SELECT COUNT(*) as total FROM session_messages")
            total_messages = cursor.fetchone()["total"]

            # Total books
            cursor.execute("SELECT COUNT(*) as total FROM session_books")
            total_books = cursor.fetchone()["total"]

            return {
                "total_sessions": total_sessions,
                "active_sessions": active_sessions,
                "total_messages": total_messages,
                "total_books": total_books
            }

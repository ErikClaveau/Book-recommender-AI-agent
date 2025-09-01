"""
Database service for managing session data with SQLite.
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Dict, Any, Optional, List
from uuid import uuid4

from app.graph.data_types import Book
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseService:
    """Service for managing SQLite database operations."""

    def __init__(self, db_path: str = "sessions.db"):
        """Initialize database service."""
        self.db_path = db_path
        logger.info(f"Initializing database service with path: {db_path}")
        self.init_database()

    def init_database(self):
        """Initialize database schema."""
        logger.debug("Initializing database schema")
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
                    book_type TEXT NOT NULL, -- 'recommended' or 'read'
                    name TEXT NOT NULL,
                    author TEXT NOT NULL,
                    description TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                )
            """)

            # Create session_preferences table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_preferences (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    preference TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (session_id) REFERENCES sessions (session_id) ON DELETE CASCADE
                )
            """)

            conn.commit()
            logger.debug("Database schema initialized successfully")

    @contextmanager
    def _get_connection(self):
        """Get database connection with proper error handling."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            yield conn
        except sqlite3.Error as e:
            logger.error(f"Database error: {e}")
            if conn:
                conn.rollback()
            raise
        finally:
            if conn:
                conn.close()

    def create_session(self, session_id: Optional[str] = None) -> str:
        """Create a new session."""
        if not session_id:
            session_id = str(uuid4())

        current_time = datetime.now(timezone.utc).isoformat()

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO sessions (session_id, created_at, last_activity)
                    VALUES (?, ?, ?)
                """, (session_id, current_time, current_time))
                conn.commit()
                logger.info(f"Database session created: {session_id}")
                return session_id
        except sqlite3.IntegrityError:
            logger.warning(f"Session already exists: {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"Error creating session {session_id}: {e}")
            raise

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data with all related information."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Get session basic info
                cursor.execute("""
                    SELECT * FROM sessions WHERE session_id = ?
                """, (session_id,))
                session_row = cursor.fetchone()

                if not session_row:
                    logger.debug(f"Session not found in database: {session_id}")
                    return None

                session_data = dict(session_row)

                # Get messages
                cursor.execute("""
                    SELECT role, content, timestamp FROM session_messages
                    WHERE session_id = ? ORDER BY timestamp
                """, (session_id,))
                session_data['messages'] = [dict(row) for row in cursor.fetchall()]

                # Get books
                cursor.execute("""
                    SELECT book_type, name, author, description FROM session_books
                    WHERE session_id = ?
                """, (session_id,))
                books = cursor.fetchall()

                recommended_books = []
                read_books = []

                for book in books:
                    book_obj = Book(
                        name=book['name'],
                        author=book['author'],
                        description=book['description']
                    )
                    if book['book_type'] == 'recommended':
                        recommended_books.append(book_obj)
                    else:
                        read_books.append(book_obj)

                session_data['recommended_books'] = recommended_books
                session_data['read_books'] = read_books

                # Get preferences
                cursor.execute("""
                    SELECT preference FROM session_preferences
                    WHERE session_id = ? ORDER BY created_at
                """, (session_id,))
                session_data['preferences'] = [row['preference'] for row in cursor.fetchall()]

                logger.debug(f"Session data retrieved from database: {session_id}")
                return session_data

        except Exception as e:
            logger.error(f"Error getting session {session_id}: {e}")
            return None

    def update_session(self, session_id: str, **kwargs):
        """Update session data."""
        try:
            current_time = datetime.now(timezone.utc).isoformat()

            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Update basic session info
                cursor.execute("""
                    UPDATE sessions 
                    SET last_activity = ?, message_count = ?, recommendation_count = ?
                    WHERE session_id = ?
                """, (
                    current_time,
                    kwargs.get('message_count', 0),
                    kwargs.get('recommendation_count', 0),
                    session_id
                ))

                # Update recommended books
                if 'recommended_books' in kwargs:
                    # Clear existing recommended books
                    cursor.execute("""
                        DELETE FROM session_books 
                        WHERE session_id = ? AND book_type = 'recommended'
                    """, (session_id,))

                    # Insert new recommended books
                    for book in kwargs['recommended_books']:
                        cursor.execute("""
                            INSERT INTO session_books 
                            (session_id, book_type, name, author, description, created_at)
                            VALUES (?, 'recommended', ?, ?, ?, ?)
                        """, (session_id, book.name, book.author, book.description, current_time))

                # Update read books
                if 'read_books' in kwargs:
                    # Clear existing read books
                    cursor.execute("""
                        DELETE FROM session_books 
                        WHERE session_id = ? AND book_type = 'read'
                    """, (session_id,))

                    # Insert new read books
                    for book in kwargs['read_books']:
                        cursor.execute("""
                            INSERT INTO session_books 
                            (session_id, book_type, name, author, description, created_at)
                            VALUES (?, 'read', ?, ?, ?, ?)
                        """, (session_id, book.name, book.author, book.description, current_time))

                # Update preferences
                if 'preferences' in kwargs:
                    # Clear existing preferences
                    cursor.execute("""
                        DELETE FROM session_preferences WHERE session_id = ?
                    """, (session_id,))

                    # Insert new preferences
                    for preference in kwargs['preferences']:
                        cursor.execute("""
                            INSERT INTO session_preferences 
                            (session_id, preference, created_at)
                            VALUES (?, ?, ?)
                        """, (session_id, preference, current_time))

                conn.commit()
                logger.debug(f"Session updated in database: {session_id}")

        except Exception as e:
            logger.error(f"Error updating session {session_id}: {e}")
            raise

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its data."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
                deleted = cursor.rowcount > 0
                conn.commit()

                if deleted:
                    logger.info(f"Session deleted from database: {session_id}")
                else:
                    logger.warning(f"No session found to delete: {session_id}")

                return deleted

        except Exception as e:
            logger.error(f"Error deleting session {session_id}: {e}")
            return False

    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT session_id, created_at, last_activity, message_count, recommendation_count
                    FROM sessions ORDER BY last_activity DESC
                """)
                sessions = [dict(row) for row in cursor.fetchall()]
                logger.debug(f"Listed {len(sessions)} sessions from database")
                return sessions

        except Exception as e:
            logger.error(f"Error listing sessions: {e}")
            return []

    def add_message(self, session_id: str, role: str, content: str):
        """Add a message to session."""
        try:
            current_time = datetime.now(timezone.utc).isoformat()

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO session_messages (session_id, role, content, timestamp)
                    VALUES (?, ?, ?, ?)
                """, (session_id, role, content, current_time))
                conn.commit()
                logger.debug(f"Message added to database for session {session_id}: role={role}")

        except Exception as e:
            logger.error(f"Error adding message to session {session_id}: {e}")
            raise

    def get_session_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Total sessions
                cursor.execute("SELECT COUNT(*) as total FROM sessions")
                total_sessions = cursor.fetchone()['total']

                # Active sessions (last 24 hours)
                cutoff_time = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
                cursor.execute("""
                    SELECT COUNT(*) as active FROM sessions 
                    WHERE last_activity > ?
                """, (cutoff_time,))
                active_sessions = cursor.fetchone()['active']

                # Total messages
                cursor.execute("SELECT COUNT(*) as total FROM session_messages")
                total_messages = cursor.fetchone()['total']

                # Total books
                cursor.execute("SELECT COUNT(*) as total FROM session_books")
                total_books = cursor.fetchone()['total']

                stats = {
                    'total_sessions': total_sessions,
                    'active_sessions': active_sessions,
                    'total_messages': total_messages,
                    'total_books': total_books
                }

                logger.debug(f"Database statistics retrieved: {stats}")
                return stats

        except Exception as e:
            logger.error(f"Error getting session stats: {e}")
            return {}

    def cleanup_old_sessions(self, timeout_hours: int) -> int:
        """Remove sessions older than timeout."""
        try:
            cutoff_time = (datetime.now(timezone.utc) - timedelta(hours=timeout_hours)).isoformat()

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM sessions WHERE last_activity < ?
                """, (cutoff_time,))
                deleted_count = cursor.rowcount
                conn.commit()

                logger.info(f"Cleaned up {deleted_count} old sessions from database")
                return deleted_count

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return 0

    def get_session_count(self) -> int:
        """Get current session count."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) as count FROM sessions")
                count = cursor.fetchone()['count']
                return count

        except Exception as e:
            logger.error(f"Error getting session count: {e}")
            return 0

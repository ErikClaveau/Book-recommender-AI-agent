"""
Database operations for session management in the Book Recommendation API.

Handles SQLite database operations for storing and retrieving user sessions,
including conversation history, preferences, and recommendations.
"""
import sqlite3
import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
from contextlib import contextmanager
from app.utils.logger import get_logger

logger = get_logger(__name__)


class DatabaseService:
    """Service class for database operations."""

    def __init__(self, db_path: str):
        """
        Initialize database service with SQLite database.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        self._init_database()
        logger.info(f"Database service initialized with path: {db_path}")

    def _init_database(self):
        """Initialize database tables if they don't exist."""
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Create sessions table with updated schema
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS sessions (
                        id TEXT PRIMARY KEY,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        data TEXT NOT NULL DEFAULT '{}',
                        message_count INTEGER DEFAULT 0,
                        recommendation_count INTEGER DEFAULT 0,
                        active INTEGER DEFAULT 1
                    )
                """)

                # Check if we need to migrate old schema
                cursor.execute("PRAGMA table_info(sessions)")
                columns = [column[1] for column in cursor.fetchall()]

                # Add missing columns if they don't exist (for migration)
                if 'message_count' not in columns:
                    cursor.execute("ALTER TABLE sessions ADD COLUMN message_count INTEGER DEFAULT 0")
                    logger.info("Added message_count column to sessions table")

                if 'recommendation_count' not in columns:
                    cursor.execute("ALTER TABLE sessions ADD COLUMN recommendation_count INTEGER DEFAULT 0")
                    logger.info("Added recommendation_count column to sessions table")

                if 'active' not in columns:
                    cursor.execute("ALTER TABLE sessions ADD COLUMN active INTEGER DEFAULT 1")
                    logger.info("Added active column to sessions table")

                conn.commit()
                logger.debug("Database tables initialized successfully")

        except sqlite3.Error as e:
            logger.error(f"Database initialization error: {e}")
            raise

    @contextmanager
    def _get_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path, timeout=30.0)
            conn.row_factory = sqlite3.Row
            yield conn
        except sqlite3.Error as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Create a new session in the database.

        Args:
            session_id: Optional session ID, if None a new UUID will be generated

        Returns:
            str: The session ID
        """
        if not session_id:
            session_id = str(uuid.uuid4())

        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR REPLACE INTO sessions 
                    (id, created_at, updated_at, data, message_count, recommendation_count, active)
                    VALUES (?, ?, ?, ?, 0, 0, 1)
                """, (
                    session_id,
                    datetime.now(timezone.utc).isoformat(),
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps({
                        "recommended_books": [],
                        "read_books": [],
                        "preferences": [],
                        "messages": []
                    })
                ))
                conn.commit()
                logger.info(f"Database session created: {session_id}")
                return session_id

        except sqlite3.Error as e:
            logger.error(f"Error creating session {session_id}: {e}")
            raise

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session data from the database.

        Args:
            session_id: The session ID to retrieve

        Returns:
            Dict containing session data or None if not found
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, created_at, updated_at, data, message_count, recommendation_count, active
                    FROM sessions 
                    WHERE id = ? AND active = 1
                """, (session_id,))

                row = cursor.fetchone()
                if row:
                    session_data = json.loads(row['data'])
                    session_data.update({
                        'session_id': row['id'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at'],
                        'message_count': row['message_count'],
                        'recommendation_count': row['recommendation_count'],
                        'active': bool(row['active'])
                    })
                    logger.debug(f"Session retrieved: {session_id}")
                    return session_data
                else:
                    logger.warning(f"Session not found: {session_id}")
                    return None

        except sqlite3.Error as e:
            logger.error(f"Error getting session {session_id}: {e}")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing session data for {session_id}: {e}")
            return None

    def update_session(self, session_id: str, **kwargs) -> bool:
        """
        Update session data in the database.

        Args:
            session_id: The session ID to update
            **kwargs: Key-value pairs to update in the session

        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            # First get current session data
            current_session = self.get_session(session_id)
            if not current_session:
                logger.warning(f"Cannot update non-existent session: {session_id}")
                return False

            # Update the data dictionary with new values
            updated_data = {
                'recommended_books': current_session.get('recommended_books', []),
                'read_books': current_session.get('read_books', []),
                'preferences': current_session.get('preferences', []),
                'messages': current_session.get('messages', [])
            }

            # Update with new values, handling both direct data updates and metadata
            message_count = current_session.get('message_count', 0)
            recommendation_count = current_session.get('recommendation_count', 0)

            for key, value in kwargs.items():
                if key in ['message_count', 'recommendation_count']:
                    if key == 'message_count':
                        message_count = value
                    elif key == 'recommendation_count':
                        recommendation_count = value
                elif key in updated_data:
                    updated_data[key] = value

            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE sessions 
                    SET updated_at = ?, data = ?, message_count = ?, recommendation_count = ?
                    WHERE id = ?
                """, (
                    datetime.now(timezone.utc).isoformat(),
                    json.dumps(updated_data),
                    message_count,
                    recommendation_count,
                    session_id
                ))
                conn.commit()

                if cursor.rowcount > 0:
                    logger.debug(f"Session updated: {session_id}")
                    return True
                else:
                    logger.warning(f"No rows updated for session: {session_id}")
                    return False

        except sqlite3.Error as e:
            logger.error(f"Error updating session {session_id}: {e}")
            return False
        except json.JSONDecodeError as e:
            logger.error(f"Error encoding session data for {session_id}: {e}")
            return False

    def list_sessions(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        List all sessions in the database.

        Args:
            active_only: Whether to return only active sessions

        Returns:
            List of session dictionaries
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                query = """
                    SELECT id, created_at, updated_at, message_count, recommendation_count, active
                    FROM sessions
                """
                if active_only:
                    query += " WHERE active = 1"

                query += " ORDER BY updated_at DESC"

                cursor.execute(query)
                sessions = []

                for row in cursor.fetchall():
                    sessions.append({
                        'session_id': row['id'],
                        'created_at': row['created_at'],
                        'updated_at': row['updated_at'],
                        'message_count': row['message_count'],
                        'recommendation_count': row['recommendation_count'],
                        'active': bool(row['active'])
                    })

                logger.debug(f"Listed {len(sessions)} sessions")
                return sessions

        except sqlite3.Error as e:
            logger.error(f"Error listing sessions: {e}")
            return []

    def delete_session(self, session_id: str) -> bool:
        """
        Mark a session as inactive (soft delete).

        Args:
            session_id: The session ID to deactivate

        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE sessions 
                    SET active = 0, updated_at = ?
                    WHERE id = ?
                """, (datetime.now(timezone.utc).isoformat(), session_id))
                conn.commit()

                if cursor.rowcount > 0:
                    logger.info(f"Session deactivated: {session_id}")
                    return True
                else:
                    logger.warning(f"No session found to deactivate: {session_id}")
                    return False

        except sqlite3.Error as e:
            logger.error(f"Error deactivating session {session_id}: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """
        Get database statistics.

        Returns:
            Dictionary containing database statistics
        """
        try:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Get total sessions
                cursor.execute("SELECT COUNT(*) as total FROM sessions")
                total_sessions = cursor.fetchone()['total']

                # Get active sessions
                cursor.execute("SELECT COUNT(*) as active FROM sessions WHERE active = 1")
                active_sessions = cursor.fetchone()['active']

                # Get total messages
                cursor.execute("SELECT SUM(message_count) as total_messages FROM sessions WHERE active = 1")
                total_messages = cursor.fetchone()['total_messages'] or 0

                # Get total recommendations
                cursor.execute("SELECT SUM(recommendation_count) as total_recs FROM sessions WHERE active = 1")
                total_recommendations = cursor.fetchone()['total_recs'] or 0

                stats = {
                    'total_sessions': total_sessions,
                    'active_sessions': active_sessions,
                    'total_messages': total_messages,
                    'total_recommendations': total_recommendations,
                    'database_path': self.db_path
                }

                logger.debug(f"Database stats retrieved: {stats}")
                return stats

        except sqlite3.Error as e:
            logger.error(f"Error getting database stats: {e}")
            return {
                'total_sessions': 0,
                'active_sessions': 0,
                'total_messages': 0,
                'total_recommendations': 0,
                'database_path': self.db_path,
                'error': str(e)
            }

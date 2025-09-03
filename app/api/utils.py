"""
Utility functions and classes for the Book Recommendation API.

Contains session management, input sanitization, and response formatting utilities.
"""
import re
import uuid
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
from app.api.database import DatabaseService
from app.utils.logger import get_logger

logger = get_logger(__name__)


class SessionManager:
    """
    Manager class for handling user sessions with database persistence.
    
    Handles session creation, retrieval, updates, and cleanup operations
    using SQLite database for persistence.
    """
    
    def __init__(self, max_sessions: int = 1000, timeout_hours: int = 24, db_path: str = "sessions.db"):
        """
        Initialize session manager with database backend.
        
        Args:
            max_sessions: Maximum number of active sessions to maintain
            timeout_hours: Session timeout in hours
            db_path: Path to SQLite database file
        """
        self.max_sessions = max_sessions
        self.timeout_hours = timeout_hours
        self.db = DatabaseService(db_path)
        logger.info(f"SessionManager initialized with max_sessions={max_sessions}, timeout_hours={timeout_hours}")
    
    def create_session(self, session_id: Optional[str] = None) -> str:
        """
        Create a new session.
        
        Args:
            session_id: Optional session ID, if None a new UUID will be generated
            
        Returns:
            str: The created session ID
        """
        if session_id and self.get_session(session_id):
            logger.debug(f"Session already exists: {session_id}")
            return session_id
        
        # Generate new session ID if not provided
        if not session_id:
            session_id = str(uuid.uuid4())
        
        # Clean up old sessions if we're at the limit
        self._cleanup_old_sessions()
        
        # Create session in database
        created_id = self.db.create_session(session_id)
        logger.info(f"New session created: {created_id}")
        return created_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve session data.
        
        Args:
            session_id: The session ID to retrieve
            
        Returns:
            Dictionary containing session data or None if not found
        """
        if not session_id:
            logger.warning("Attempted to get session with empty ID")
            return None
        
        session_data = self.db.get_session(session_id)
        
        if session_data:
            # Check if session has expired
            if self._is_session_expired(session_data):
                logger.info(f"Session expired: {session_id}")
                self.delete_session(session_id)
                return None
            
            logger.debug(f"Session retrieved: {session_id}")
            return session_data
        else:
            logger.warning(f"Session not found: {session_id}")
            return None
    
    def update_session(self, session_id: str, **kwargs) -> bool:
        """
        Update session data.
        
        Args:
            session_id: The session ID to update
            **kwargs: Key-value pairs to update
            
        Returns:
            bool: True if update was successful
        """
        if not session_id:
            logger.warning("Attempted to update session with empty ID")
            return False
        
        success = self.db.update_session(session_id, **kwargs)
        if success:
            logger.debug(f"Session updated: {session_id}")
        else:
            logger.warning(f"Failed to update session: {session_id}")
        
        return success
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session (mark as inactive).
        
        Args:
            session_id: The session ID to delete
            
        Returns:
            bool: True if deletion was successful
        """
        success = self.db.delete_session(session_id)
        if success:
            logger.info(f"Session deleted: {session_id}")
        return success
    
    def list_sessions(self, active_only: bool = True) -> List[Dict[str, Any]]:
        """
        List all sessions.
        
        Args:
            active_only: Whether to return only active sessions
            
        Returns:
            List of session dictionaries
        """
        sessions = self.db.list_sessions(active_only)
        logger.debug(f"Listed {len(sessions)} sessions")
        return sessions
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get session statistics.
        
        Returns:
            Dictionary containing session statistics
        """
        return self.db.get_stats()
    
    def _is_session_expired(self, session_data: Dict[str, Any]) -> bool:
        """
        Check if a session has expired.
        
        Args:
            session_data: Session data dictionary
            
        Returns:
            bool: True if session has expired
        """
        try:
            updated_at_str = session_data.get('updated_at')
            if not updated_at_str:
                return True
            
            updated_at = datetime.fromisoformat(updated_at_str.replace('Z', '+00:00'))
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)
            
            expiry_time = updated_at + timedelta(hours=self.timeout_hours)
            is_expired = datetime.now(timezone.utc) > expiry_time
            
            if is_expired:
                logger.debug(f"Session expired: updated_at={updated_at}, expiry_time={expiry_time}")
            
            return is_expired
            
        except (ValueError, TypeError) as e:
            logger.error(f"Error checking session expiry: {e}")
            return True  # Consider expired if we can't parse the date
    
    def _cleanup_old_sessions(self):
        """Clean up old or expired sessions to maintain the session limit."""
        try:
            sessions = self.list_sessions(active_only=True)
            
            # Remove expired sessions
            expired_count = 0
            for session in sessions:
                if self._is_session_expired(session):
                    self.delete_session(session['session_id'])
                    expired_count += 1
            
            if expired_count > 0:
                logger.info(f"Cleaned up {expired_count} expired sessions")
            
            # If still over limit, remove oldest sessions
            remaining_sessions = self.list_sessions(active_only=True)
            if len(remaining_sessions) > self.max_sessions:
                # Sort by updated_at and remove oldest
                remaining_sessions.sort(key=lambda x: x.get('updated_at', ''))
                sessions_to_remove = len(remaining_sessions) - self.max_sessions
                
                for i in range(sessions_to_remove):
                    session_id = remaining_sessions[i]['session_id']
                    self.delete_session(session_id)
                
                logger.info(f"Cleaned up {sessions_to_remove} old sessions to maintain limit")
                
        except Exception as e:
            logger.error(f"Error during session cleanup: {e}")


def sanitize_input(text: str, max_length: int = 2000) -> str:
    """
    Sanitize user input to prevent injection attacks and limit length.
    
    Args:
        text: The input text to sanitize
        max_length: Maximum allowed length
        
    Returns:
        str: Sanitized text
    """
    if not text or not isinstance(text, str):
        logger.warning("Invalid input for sanitization")
        return ""
    
    # Remove potentially dangerous characters
    sanitized = re.sub(r'[<>"\'\\\x00-\x1f\x7f-\x9f]', '', text)
    
    # Limit length
    if len(sanitized) > max_length:
        sanitized = sanitized[:max_length]
        logger.debug(f"Input truncated to {max_length} characters")
    
    # Remove excessive whitespace
    sanitized = ' '.join(sanitized.split())
    
    logger.debug(f"Input sanitized: original length {len(text)}, sanitized length {len(sanitized)}")
    return sanitized


def format_response(response_text: str, session_id: str, 
                   recommended_books: Optional[List] = None,
                   preferences: Optional[List[str]] = None,
                   read_books: Optional[List] = None) -> Dict[str, Any]:
    """
    Format API response with consistent structure and proper serialization.

    Args:
        response_text: The main response text
        session_id: Session identifier
        recommended_books: List of recommended books (Book objects or dicts)
        preferences: List of user preferences
        read_books: List of books user has read (Book objects or dicts)

    Returns:
        Dict: Formatted response dictionary with serialized book data
    """
    # Convert Book objects to dictionaries for JSON serialization
    def serialize_books(books: Optional[List]) -> List[Dict[str, Any]]:
        if not books:
            return []

        serialized_books = []
        for book in books:
            if hasattr(book, 'model_dump'):  # Pydantic v2
                serialized_books.append(book.model_dump())
            elif hasattr(book, 'dict'):  # Pydantic v1
                serialized_books.append(book.dict())
            elif hasattr(book, '__dict__'):  # Regular object with attributes
                book_dict = {}
                for attr in ['name', 'author', 'title', 'genre', 'year', 'description', 'rating']:
                    if hasattr(book, attr):
                        value = getattr(book, attr)
                        if value is not None:
                            book_dict[attr] = value
                # Ensure we have at least name/title and author
                if 'name' in book_dict and 'title' not in book_dict:
                    book_dict['title'] = book_dict['name']
                elif 'title' in book_dict and 'name' not in book_dict:
                    book_dict['name'] = book_dict['title']
                serialized_books.append(book_dict)
            elif isinstance(book, dict):  # Already a dictionary
                serialized_books.append(book)
            else:
                logger.warning(f"Unknown book format: {type(book)}, converting to string")
                serialized_books.append({
                    'name': str(book),
                    'author': 'Unknown',
                    'title': str(book)
                })

        return serialized_books

    formatted_response = {
        "response": response_text or "",
        "session_id": session_id,
        "recommended_books": serialize_books(recommended_books),
        "preferences": preferences or [],
        "read_books": serialize_books(read_books),
        "timestamp": datetime.now(timezone.utc).isoformat()
    }
    
    logger.debug(f"Response formatted for session {session_id} with {len(formatted_response['recommended_books'])} books")
    return formatted_response


def validate_uuid(uuid_string: str) -> bool:
    """
    Validate UUID format.
    
    Args:
        uuid_string: String to validate as UUID
        
    Returns:
        bool: True if valid UUID format
    """
    try:
        uuid.UUID(uuid_string)
        return True
    except (ValueError, TypeError):
        return False


def extract_book_data(book_objects: List[Any]) -> List[Dict[str, Any]]:
    """
    Extract book data from Book objects to dictionary format.
    
    Args:
        book_objects: List of Book objects or dictionaries
        
    Returns:
        List of book dictionaries
    """
    books_data = []
    
    for book in book_objects:
        if hasattr(book, '__dict__'):
            # Handle Book objects
            book_dict = {
                'title': getattr(book, 'title', ''),
                'author': getattr(book, 'author', ''),
                'genre': getattr(book, 'genre', ''),
                'year': getattr(book, 'year', None),
                'description': getattr(book, 'description', ''),
                'rating': getattr(book, 'rating', None)
            }
        elif isinstance(book, dict):
            # Handle dictionary objects
            book_dict = {
                'title': book.get('title', ''),
                'author': book.get('author', ''),
                'genre': book.get('genre', ''),
                'year': book.get('year', None),
                'description': book.get('description', ''),
                'rating': book.get('rating', None)
            }
        else:
            logger.warning(f"Unknown book format: {type(book)}")
            continue
        
        books_data.append(book_dict)
    
    logger.debug(f"Extracted data for {len(books_data)} books")
    return books_data

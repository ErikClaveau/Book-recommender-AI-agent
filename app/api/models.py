"""
Data models and validation functions for the Book Recommendation API.

Contains Pydantic models and validation utilities for API request/response handling.
"""
import re
from typing import Optional
from app.utils.logger import get_logger

logger = get_logger(__name__)


def validate_message_content(message: str, max_length: int = 2000) -> bool:
    """
    Validate message content for chat requests.

    Args:
        message: The message content to validate
        max_length: Maximum allowed message length

    Returns:
        bool: True if message is valid, False otherwise
    """
    if not message or not isinstance(message, str):
        logger.warning("Message validation failed: empty or invalid type")
        return False

    # Check length
    if len(message.strip()) == 0:
        logger.warning("Message validation failed: empty message after strip")
        return False

    if len(message) > max_length:
        logger.warning(f"Message validation failed: length {len(message)} exceeds max {max_length}")
        return False

    # Check for potentially malicious content
    suspicious_patterns = [
        r'<script[^>]*>.*?</script>',  # Script tags
        r'javascript:',                # JavaScript URLs
        r'on\w+\s*=',                 # Event handlers
        r'data:text/html',            # Data URLs with HTML
    ]

    message_lower = message.lower()
    for pattern in suspicious_patterns:
        if re.search(pattern, message_lower, re.IGNORECASE | re.DOTALL):
            logger.warning(f"Message validation failed: suspicious pattern detected: {pattern}")
            return False

    logger.debug(f"Message validation passed for message of length {len(message)}")
    return True


def sanitize_session_id(session_id: Optional[str]) -> Optional[str]:
    """
    Sanitize session ID to prevent injection attacks.

    Args:
        session_id: The session ID to sanitize

    Returns:
        str: Sanitized session ID or None if invalid
    """
    if not session_id:
        return None

    # Only allow alphanumeric characters and hyphens (UUID format)
    if not re.match(r'^[a-fA-F0-9\-]+$', session_id):
        logger.warning(f"Invalid session ID format: {session_id}")
        return None

    # Check length (UUID should be 36 characters with hyphens)
    if len(session_id) not in [32, 36]:  # With or without hyphens
        logger.warning(f"Invalid session ID length: {len(session_id)}")
        return None

    return session_id

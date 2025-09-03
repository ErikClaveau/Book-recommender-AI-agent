"""
Configuration settings for the Book Recommendation API.

Contains all configuration parameters for the FastAPI application,
including database settings, CORS configuration, and API limits.
"""
import os
from typing import List
from app.utils.logger import get_logger

logger = get_logger(__name__)


class APIConfig:
    """Configuration class for API settings."""
    
    def __init__(self):
        """Initialize API configuration with default and environment values."""
        # Database configuration
        self.database_path = os.getenv("DATABASE_PATH", "sessions.db")
        
        # Server configuration
        self.host = os.getenv("API_HOST", "0.0.0.0")
        self.port = int(os.getenv("API_PORT", "8000"))
        self.debug = os.getenv("DEBUG", "false").lower() == "true"
        
        # CORS configuration
        self.cors_origins = self._parse_list(os.getenv("CORS_ORIGINS", "*"))
        self.cors_allow_credentials = os.getenv("CORS_ALLOW_CREDENTIALS", "true").lower() == "true"
        self.cors_allow_methods = self._parse_list(os.getenv("CORS_ALLOW_METHODS", "GET,POST,PUT,DELETE,OPTIONS"))
        self.cors_allow_headers = self._parse_list(os.getenv("CORS_ALLOW_HEADERS", "*"))
        
        # Session management
        self.max_sessions = int(os.getenv("MAX_SESSIONS", "1000"))
        self.session_timeout_hours = int(os.getenv("SESSION_TIMEOUT_HOURS", "24"))
        
        # API limits
        self.max_message_length = int(os.getenv("MAX_MESSAGE_LENGTH", "2000"))
        
        logger.info(f"Configuration loaded: host={self.host}, port={self.port}, debug={self.debug}")
    
    def _parse_list(self, value: str) -> List[str]:
        """Parse comma-separated string into list."""
        if value == "*":
            return ["*"]
        return [item.strip() for item in value.split(",") if item.strip()]


# Global config instance
config = APIConfig()

logger.info("API configuration initialized")

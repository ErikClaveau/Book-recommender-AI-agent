"""
Configuration settings for the Book Recommendation API.
"""
from pydantic import BaseModel
from typing import List
import os
from app.utils.logger import get_logger

logger = get_logger(__name__)


class APIConfig(BaseModel):
    """Configuration settings for the API."""

    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # CORS settings
    cors_origins: List[str] = ["*"]
    cors_allow_credentials: bool = True
    cors_allow_methods: List[str] = ["*"]
    cors_allow_headers: List[str] = ["*"]

    # Session settings
    session_timeout_hours: int = 24
    max_sessions: int = 1000

    # Database settings
    database_path: str = "sessions.db"
    auto_cleanup_interval_minutes: int = 60

    # API settings
    max_message_length: int = 1000
    max_recommendations: int = 10

    @classmethod
    def from_env(cls) -> "APIConfig":
        """Create configuration from environment variables."""
        logger.debug("Loading configuration from environment variables")
        config = cls(
            host=os.getenv("API_HOST", "0.0.0.0"),
            port=int(os.getenv("API_PORT", "8000")),
            debug=os.getenv("API_DEBUG", "false").lower() == "true",
            cors_origins=os.getenv("CORS_ORIGINS", "*").split(","),
            session_timeout_hours=int(os.getenv("SESSION_TIMEOUT_HOURS", "24")),
            max_sessions=int(os.getenv("MAX_SESSIONS", "1000")),
            database_path=os.getenv("DATABASE_PATH", "sessions.db"),
            auto_cleanup_interval_minutes=int(os.getenv("AUTO_CLEANUP_INTERVAL_MINUTES", "60")),
            max_message_length=int(os.getenv("MAX_MESSAGE_LENGTH", "1000")),
            max_recommendations=int(os.getenv("MAX_RECOMMENDATIONS", "10"))
        )
        logger.info(f"Configuration loaded: host={config.host}, port={config.port}, debug={config.debug}")
        return config


# Global configuration instance
config = APIConfig.from_env()
logger.info("API configuration initialized")

"""
Additional models and utilities for the Book Recommendation API.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime
from enum import Enum

from app.graph.data_types import Book, IntentEnum


class APIStatus(str, Enum):
    """API status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


class SessionInfo(BaseModel):
    """Information about a user session."""
    session_id: str
    created_at: str
    last_activity: Optional[str] = None
    message_count: int = 0
    recommendation_count: int = 0


class BulkRecommendationRequest(BaseModel):
    """Request model for bulk recommendations."""
    requests: List[Dict[str, Any]] = Field(..., description="List of recommendation requests")

    @validator('requests')
    def validate_requests(cls, v):
        if len(v) > 10:
            raise ValueError("Maximum 10 requests allowed in bulk")
        return v


class BookSearchRequest(BaseModel):
    """Request model for searching books."""
    query: str = Field(..., min_length=1, max_length=200)
    limit: int = Field(10, ge=1, le=50)


class UserPreferences(BaseModel):
    """Model for user reading preferences."""
    genres: List[str] = []
    authors: List[str] = []
    length_preference: Optional[str] = None  # "short", "medium", "long"
    difficulty_level: Optional[str] = None   # "easy", "moderate", "challenging"
    themes: List[str] = []


class RecommendationStats(BaseModel):
    """Statistics about recommendations."""
    total_recommendations: int = 0
    unique_books_recommended: int = 0
    user_satisfaction_score: Optional[float] = None
    most_recommended_genre: Optional[str] = None


class APIError(BaseModel):
    """Standard error response model."""
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ConversationHistory(BaseModel):
    """Model for conversation history."""
    messages: List[Dict[str, str]]
    session_id: str
    started_at: str
    updated_at: str


def validate_message_content(content: str, max_length: int = 1000) -> bool:
    """Validate message content."""
    if not content or not content.strip():
        return False
    if len(content) > max_length:
        return False
    return True


def format_book_list(books: List[Book]) -> str:
    """Format a list of books for display."""
    if not books:
        return "No books available."

    formatted = []
    for i, book in enumerate(books, 1):
        formatted.append(f"{i}. {book.name} by {book.author}")

    return "\n".join(formatted)


def extract_intent_from_message(message: str) -> List[str]:
    """Extract potential intents from a user message."""
    message_lower = message.lower()
    intents = []

    # Recommendation keywords
    if any(word in message_lower for word in ["recommend", "suggestion", "suggest", "book", "read"]):
        intents.append("recommendation")

    # Preference keywords
    if any(word in message_lower for word in ["like", "prefer", "enjoy", "favorite", "genre"]):
        intents.append("preferences")

    # Read books keywords
    if any(word in message_lower for word in ["read", "finished", "completed"]):
        intents.append("read")

    # General conversation keywords
    if any(word in message_lower for word in ["hello", "hi", "chat", "talk", "discuss"]):
        intents.append("talk")

    return intents or ["talk"]

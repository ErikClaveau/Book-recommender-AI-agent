"""
FastAPI application for the Book Recommendation AI Agent.

Provides REST API endpoints to interact with the recommendation graph,
allowing users to get book recommendations, save preferences, and manage reading history.
"""
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
from datetime import datetime

from app.graph.graph import graph
from app.graph.states import InternalState
from app.graph.data_types import Book, IntentEnum
from app.api.config import config
from app.api.models import (
    APIError, SessionInfo, RecommendationStats,
    format_book_list, validate_message_content
)
from app.api.utils import SessionManager, format_response, sanitize_input


# Request/Response Models
class ChatRequest(BaseModel):
    """Request model for chat interactions."""
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat interactions."""
    response: str
    session_id: str
    recommended_books: List[Book] = []
    preferences: List[str] = []
    read_books: List[Book] = []


class RecommendationRequest(BaseModel):
    """Request model for getting recommendations."""
    preferences: Optional[List[str]] = None
    read_books: Optional[List[Book]] = None
    session_id: Optional[str] = None


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    recommended_books: List[Book]
    session_id: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    timestamp: str
    version: str = "1.0.0"


# Initialize FastAPI app
app = FastAPI(
    title="Book Recommendation API",
    description="AI-powered book recommendation system using LangGraph",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.cors_origins,
    allow_credentials=config.cors_allow_credentials,
    allow_methods=config.cors_allow_methods,
    allow_headers=config.cors_allow_headers,
)

# Initialize session manager
session_manager = SessionManager(
    max_sessions=config.max_sessions,
    timeout_hours=config.session_timeout_hours
)


def create_initial_state(session_id: str, message: str) -> InternalState:
    """Create initial state for graph execution."""
    session_data = session_manager.get_session(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    return InternalState(
        messages=[{"role": "user", "content": message}],
        recommended_books=session_data.get("recommended_books", []),
        read_books=session_data.get("read_books", []),
        preferences=session_data.get("preferences", []),
        intents=[]
    )


def update_session_from_state(session_id: str, final_state: Dict[str, Any]):
    """Update session data from graph execution results."""
    session_data = session_manager.get_session(session_id)
    if not session_data:
        return

    # Update accumulated data - accessing as dictionary
    session_manager.update_session(
        session_id,
        recommended_books=final_state.get("recommended_books", []),
        read_books=final_state.get("read_books", []),
        preferences=final_state.get("preferences", []),
        message_count=session_data.get("message_count", 0) + 1,
        recommendation_count=session_data.get("recommendation_count", 0) + len(final_state.get("recommended_books", []))
    )

    # Store conversation history
    session_data = session_manager.get_session(session_id)
    if "messages" not in session_data:
        session_data["messages"] = []

    # Add the latest messages to session
    messages = final_state.get("messages", [])
    for message in messages:
        if message not in session_data["messages"]:
            session_data["messages"].append(message)


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with basic health check."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version="1.0.0"
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main chat endpoint for natural language interaction with the book recommendation agent.

    Processes user messages through the recommendation graph and returns responses
    along with any updated recommendations, preferences, or reading history.
    """
    try:
        # Validate and sanitize input
        sanitized_message = sanitize_input(request.message, config.max_message_length)
        if not validate_message_content(sanitized_message, config.max_message_length):
            raise HTTPException(status_code=400, detail="Invalid message content")

        # Get or create session
        session_id = request.session_id
        if not session_id or not session_manager.get_session(session_id):
            session_id = session_manager.create_session(session_id)

        # Create initial state
        initial_state = create_initial_state(session_id, sanitized_message)

        # Execute graph
        result = graph.invoke(initial_state)

        # Update session with results
        update_session_from_state(session_id, result)

        # Extract assistant response from messages
        assistant_response = ""
        messages = result.get("messages", [])
        for message in reversed(messages):
            if hasattr(message, 'content') and message.content:
                if hasattr(message, 'role') and message.role == 'assistant':
                    assistant_response = message.content
                    break
                elif not hasattr(message, 'role'):
                    # Handle different message formats
                    assistant_response = str(message.content) if hasattr(message, 'content') else str(message)
                    break

        if not assistant_response:
            assistant_response = "I'm here to help you find great books! What are you looking for?"

        # Format response with book recommendations if any
        recommended_books = result.get("recommended_books", [])
        formatted_response = format_response(assistant_response, recommended_books)

        return ChatResponse(
            response=formatted_response,
            session_id=session_id,
            recommended_books=result.get("recommended_books", []),
            preferences=result.get("preferences", []),
            read_books=result.get("read_books", [])
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing chat request: {str(e)}")


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get book recommendations based on preferences and reading history.

    This endpoint directly requests recommendations without natural language processing.
    """
    try:
        # Get or create session
        session_id = request.session_id
        if not session_id or not session_manager.get_session(session_id):
            session_id = session_manager.create_session(session_id)

        # Create recommendation request message
        message_parts = ["I'd like some book recommendations."]

        if request.preferences:
            message_parts.append(f"My preferences are: {', '.join(request.preferences)}")

        if request.read_books:
            book_names = [f"{book.name} by {book.author}" for book in request.read_books]
            message_parts.append(f"I have read: {', '.join(book_names)}")

        recommendation_message = " ".join(message_parts)

        # Create initial state
        initial_state = create_initial_state(session_id, recommendation_message)

        # Add any provided data to the state
        if request.preferences:
            initial_state.preferences.extend(request.preferences)
        if request.read_books:
            initial_state.read_books.extend(request.read_books)

        # Execute graph
        result = graph.invoke(initial_state)

        # Update session with results
        update_session_from_state(session_id, result)

        return RecommendationResponse(
            recommended_books=result.get("recommended_books", [])[:config.max_recommendations],
            session_id=session_id
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recommendations: {str(e)}")


@app.get("/sessions/{session_id}", response_model=Dict[str, Any])
async def get_session(session_id: str):
    """Get session data including conversation history and accumulated data."""
    session_data = session_manager.get_session(session_id)
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")

    return session_data


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and all its data."""
    if not session_manager.delete_session(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    return {"message": f"Session {session_id} deleted successfully"}


@app.get("/sessions", response_model=Dict[str, Any])
async def list_sessions():
    """List all active sessions."""
    sessions_info = session_manager.list_sessions()
    return {
        "sessions": sessions_info,
        "count": len(sessions_info)
    }


@app.get("/stats", response_model=RecommendationStats)
async def get_stats():
    """Get API usage statistics."""
    sessions_info = session_manager.list_sessions()
    total_recommendations = sum(s.get("recommendation_count", 0) for s in sessions_info)

    return RecommendationStats(
        total_recommendations=total_recommendations,
        unique_books_recommended=0,  # Could be calculated from session data
        user_satisfaction_score=None,  # Could be implemented with user feedback
        most_recommended_genre=None   # Could be calculated from preferences
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        debug=config.debug
    )

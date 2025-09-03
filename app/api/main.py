"""
FastAPI application for the Book Recommendation AI Agent.

Provides REST API endpoints to interact with the recommendation graph,
allowing users to get book recommendations, save preferences, and manage reading history.
"""
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timezone

from app.graph.graph import graph
from app.graph.states import InternalState
from app.graph.data_types import Book
from app.api.config import config
from app.api.models import validate_message_content
from app.api.utils import SessionManager, format_response, sanitize_input
from app.utils.logger import get_logger

logger = get_logger(__name__)

# Request/Response Models
class ChatRequest(BaseModel):
    """Request model for chat interactions."""
    message: str
    session_id: Optional[str] = None


class BookDict(BaseModel):
    """Dictionary representation of a book for API responses."""
    name: Optional[str] = None
    title: Optional[str] = None
    author: str
    genre: Optional[str] = None
    year: Optional[int] = None
    description: Optional[str] = None
    rating: Optional[float] = None


class ChatResponse(BaseModel):
    """Response model for chat interactions."""
    response: str
    session_id: str
    recommended_books: List[BookDict] = []
    preferences: List[str] = []
    read_books: List[BookDict] = []
    timestamp: Optional[str] = None


class RecommendationRequest(BaseModel):
    """Request model for getting recommendations."""
    preferences: Optional[List[str]] = None
    read_books: Optional[List[Book]] = None
    session_id: Optional[str] = None


class RecommendationResponse(BaseModel):
    """Response model for recommendations."""
    recommended_books: List[BookDict]
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
    timeout_hours=config.session_timeout_hours,
    db_path=config.database_path
)

logger.info("Book Recommendation API initialized")


def create_initial_state(session_id: str, message: str) -> InternalState:
    """Create initial state for graph execution."""
    session_data = session_manager.get_session(session_id)
    if not session_data:
        logger.error(f"Session not found: {session_id}")
        raise HTTPException(status_code=404, detail="Session not found")

    logger.debug(f"Creating initial state for session {session_id}")
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
        logger.warning(f"Attempted to update non-existent session: {session_id}")
        return

    logger.debug(f"Updating session {session_id} with new state data")

    # Serialize Book objects before storing in database
    def serialize_books_for_storage(books):
        """Convert Book objects to dictionaries for database storage."""
        if not books:
            return []

        serialized = []
        for book in books:
            if hasattr(book, 'model_dump'):  # Pydantic v2
                serialized.append(book.model_dump())
            elif hasattr(book, 'dict'):  # Pydantic v1
                serialized.append(book.dict())
            elif hasattr(book, '__dict__'):  # Object with attributes
                book_dict = {}
                for attr in ['name', 'author', 'title', 'genre', 'year', 'description', 'rating']:
                    if hasattr(book, attr):
                        value = getattr(book, attr)
                        if value is not None:
                            book_dict[attr] = value
                # Ensure consistency between name and title
                if 'name' in book_dict and 'title' not in book_dict:
                    book_dict['title'] = book_dict['name']
                elif 'title' in book_dict and 'name' not in book_dict:
                    book_dict['name'] = book_dict['title']
                serialized.append(book_dict)
            elif isinstance(book, dict):
                serialized.append(book)
            else:
                logger.warning(f"Unknown book type {type(book)}, converting to string")
                serialized.append({'name': str(book), 'author': 'Unknown'})

        return serialized

    # Serialize the book data before updating session
    serialized_recommended_books = serialize_books_for_storage(final_state.get("recommended_books", []))
    serialized_read_books = serialize_books_for_storage(final_state.get("read_books", []))

    # Update accumulated data with serialized book objects
    session_manager.update_session(
        session_id,
        recommended_books=serialized_recommended_books,
        read_books=serialized_read_books,
        preferences=final_state.get("preferences", []),
        message_count=session_data.get("message_count", 0) + 1,
        recommendation_count=session_data.get("recommendation_count", 0) + len(serialized_recommended_books)
    )

    # Store conversation history
    session_data = session_manager.get_session(session_id)
    if session_data and "messages" not in session_data:
        session_data["messages"] = []

    # Add the latest messages to session
    messages = final_state.get("messages", [])
    if session_data and messages:
        for message in messages:
            if message not in session_data["messages"]:
                session_data["messages"].append(message)


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint with basic health check."""
    logger.debug("Health check requested at root endpoint")
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="1.0.0"
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    logger.debug("Health check requested")
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
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
        logger.info(f"Chat request received for session: {request.session_id}")

        # Validate and sanitize input
        sanitized_message = sanitize_input(request.message, config.max_message_length)
        if not validate_message_content(sanitized_message, config.max_message_length):
            logger.warning(f"Invalid message content from session {request.session_id}")
            raise HTTPException(status_code=400, detail="Invalid message content")

        # Get or create session
        session_id = request.session_id
        if not session_id or not session_manager.get_session(session_id):
            session_id = session_manager.create_session(session_id)
            logger.info(f"Created new session: {session_id}")
        else:
            logger.debug(f"Using existing session: {session_id}")

        # Create initial state
        initial_state = create_initial_state(session_id, sanitized_message)

        # Execute graph
        logger.debug(f"Executing graph for session {session_id}")
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
            elif hasattr(message, 'content') and not hasattr(message, 'role'):
                # Handle AIMessage or other LangChain message types
                assistant_response = str(message.content)
                break
            elif isinstance(message, dict):
                # Handle dictionary messages
                if message.get('role') == 'assistant' and message.get('content'):
                    assistant_response = message.get('content', '')
                    break
                elif message.get('content') and not message.get('role'):
                    assistant_response = str(message.get('content', ''))
                    break
            elif hasattr(message, '__str__'):
                # Fallback to string representation
                assistant_response = str(message)
                break

        if not assistant_response:
            assistant_response = "I understand. How can I help you with book recommendations?"

        # Format response
        response_data = format_response(
            response_text=assistant_response,
            session_id=session_id,
            recommended_books=result.get("recommended_books", []),
            preferences=result.get("preferences", []),
            read_books=result.get("read_books", [])
        )

        logger.info(f"Chat response generated for session {session_id}")
        return ChatResponse(**response_data)

    except Exception as e:
        logger.error(f"Error processing chat request: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing request: {str(e)}")


@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get book recommendations based on preferences and reading history.

    This endpoint allows direct recommendation requests without chat interaction.
    """
    try:
        logger.info(f"Recommendation request received for session: {request.session_id}")

        # Get or create session
        session_id = request.session_id
        if not session_id or not session_manager.get_session(session_id):
            session_id = session_manager.create_session(session_id)
            logger.info(f"Created new session for recommendations: {session_id}")

        # Create state for recommendations
        session_data = session_manager.get_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")

        # Use provided data or session data
        preferences = request.preferences or session_data.get("preferences", [])
        read_books = request.read_books or session_data.get("read_books", [])

        # Create message requesting recommendations
        recommendation_message = "Please give me some book recommendations"
        if preferences:
            recommendation_message += f" based on my preferences: {', '.join(preferences)}"

        initial_state = InternalState(
            messages=[{"role": "user", "content": recommendation_message}],
            recommended_books=session_data.get("recommended_books", []),
            read_books=read_books,
            preferences=preferences,
            intents=["recommend_books"]
        )

        # Execute graph
        result = graph.invoke(initial_state)

        # Update session
        update_session_from_state(session_id, result)

        logger.info(f"Recommendations generated for session {session_id}")
        return RecommendationResponse(
            recommended_books=result.get("recommended_books", []),
            session_id=session_id
        )

    except Exception as e:
        logger.error(f"Error generating recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")


@app.get("/sessions")
async def list_sessions():
    """List all active sessions."""
    try:
        sessions = session_manager.list_sessions(active_only=True)
        logger.info(f"Listed {len(sessions)} active sessions")
        return {"sessions": sessions}
    except Exception as e:
        logger.error(f"Error listing sessions: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error listing sessions: {str(e)}")


@app.get("/sessions/{session_id}")
async def get_session_details(session_id: str):
    """Get details for a specific session."""
    try:
        session_data = session_manager.get_session(session_id)
        if not session_data:
            logger.warning(f"Session not found: {session_id}")
            raise HTTPException(status_code=404, detail="Session not found")

        logger.debug(f"Session details retrieved: {session_id}")
        return {"session": session_data}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session details: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting session details: {str(e)}")


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a specific session."""
    try:
        success = session_manager.delete_session(session_id)
        if success:
            logger.info(f"Session deleted: {session_id}")
            return {"message": "Session deleted successfully", "session_id": session_id}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error deleting session: {str(e)}")


@app.get("/stats")
async def get_api_stats():
    """Get API statistics and health information."""
    try:
        stats = session_manager.get_stats()
        stats.update({
            "api_version": "1.0.0",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "healthy"
        })
        logger.debug(f"API stats retrieved: {stats}")
        return stats
    except Exception as e:
        logger.error(f"Error getting API stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting API stats: {str(e)}")


# Admin endpoints for debugging and maintenance
@app.post("/admin/cleanup")
async def cleanup_old_sessions():
    """Manually trigger cleanup of old sessions."""
    try:
        session_manager._cleanup_old_sessions()
        logger.info("Manual cleanup triggered")
        return {"message": "Cleanup completed successfully"}
    except Exception as e:
        logger.error(f"Error during cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during cleanup: {str(e)}")


@app.get("/admin/database-info")
async def get_database_info():
    """Get database information and statistics."""
    try:
        stats = session_manager.get_stats()
        logger.debug(f"Database info requested: {stats}")
        return {
            "database_path": config.database_path,
            "stats": stats,
            "config": {
                "max_sessions": config.max_sessions,
                "session_timeout_hours": config.session_timeout_hours
            }
        }
    except Exception as e:
        logger.error(f"Error getting database info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting database info: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        debug=config.debug
    )

"""
Streamlit Interface for the Book Recommendation Agent.

Provides an intuitive web interface to interact with the AI
book recommendation agent through the FastAPI.
"""

import streamlit as st
import requests
import json
from typing import List, Dict, Any, Optional
import uuid
from datetime import datetime
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="📚 Book Recommendation Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API URLs
API_BASE_URL = "http://localhost:8000"

class BookRecommenderUI:
    """Main class for the book recommender user interface."""

    def __init__(self):
        """Initialize the user interface."""
        self.session_id = self._get_or_create_session_id()

    def _get_or_create_session_id(self) -> str:
        """Get or create a unique session ID."""
        if 'session_id' not in st.session_state:
            st.session_state.session_id = str(uuid.uuid4())
        return st.session_state.session_id

    def _make_api_request(self, endpoint: str, method: str = "GET", data: Dict = None) -> Optional[Dict]:
        """Make a request to the API."""
        try:
            url = f"{API_BASE_URL}/{endpoint.lstrip('/')}"

            if method == "GET":
                response = requests.get(url, timeout=30)
            elif method == "POST":
                response = requests.post(url, json=data, timeout=30)
            elif method == "DELETE":
                response = requests.delete(url, timeout=30)
            else:
                st.error(f"Unsupported HTTP method: {method}")
                return None

            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"API Error ({response.status_code}): {response.text}")
                return None

        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot connect to API. Is the server running on http://localhost:8000?")
            return None
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timeout. The API took too long to respond.")
            return None
        except Exception as e:
            st.error(f"Unexpected error: {str(e)}")
            return None

    def _send_chat_message(self, message: str) -> Optional[Dict]:
        """Send a message to the chat endpoint."""
        data = {
            "message": message,
            "session_id": self.session_id
        }
        return self._make_api_request("/chat", "POST", data)

    def _get_direct_recommendations(self, preferences: List[str] = None, read_books: List[Dict] = None) -> Optional[Dict]:
        """Get direct recommendations."""
        data = {
            "session_id": self.session_id,
            "preferences": preferences or [],
            "read_books": read_books or []
        }
        return self._make_api_request("/recommend", "POST", data)

    def _get_session_info(self) -> Optional[Dict]:
        """Get current session information."""
        return self._make_api_request(f"/sessions/{self.session_id}")

    def _get_all_sessions(self) -> Optional[Dict]:
        """Get all active sessions."""
        return self._make_api_request("/sessions")

    def _delete_session(self, session_id: str) -> bool:
        """Delete a specific session."""
        result = self._make_api_request(f"/sessions/{session_id}", "DELETE")
        return result is not None

    def _switch_to_session(self, session_id: str):
        """Switch to a different session."""
        st.session_state.session_id = session_id
        st.session_state.messages = []  # Clear current chat history
        self.session_id = session_id
        st.rerun()

    def _get_api_stats(self) -> Optional[Dict]:
        """Get API statistics."""
        return self._make_api_request("/stats")

    def _format_book_display(self, book: Dict) -> str:
        """Format a book for display."""
        return f"📖 **{book.get('name', 'Unknown Title')}** by {book.get('author', 'Unknown Author')}"

    def render_sidebar(self):
        """Render the sidebar with session information and controls."""
        st.sidebar.title("📚 Session Control")

        # Previous Sessions List
        st.sidebar.subheader("📋 Previous Sessions")
        all_sessions_data = self._get_all_sessions()
        if all_sessions_data and all_sessions_data.get('sessions'):
            sessions = all_sessions_data['sessions']

            # Show current session first
            current_session = None
            other_sessions = []

            for session in sessions:
                if session['session_id'] == self.session_id:
                    current_session = session
                else:
                    other_sessions.append(session)

            # Display current session
            if current_session:
                st.sidebar.markdown(f"**🟢 Current Session**")
                st.sidebar.text(f"ID: {current_session['session_id'][:8]}...")
                st.sidebar.text(f"Messages: {current_session.get('message_count', 0)}")
                st.sidebar.text(f"Recommendations: {current_session.get('recommendation_count', 0)}")

                # Delete current session button
                if st.sidebar.button("🗑️ Delete Current Session", help="Delete this session permanently"):
                    if self._delete_session(self.session_id):
                        st.sidebar.success("Session deleted")
                        st.session_state.clear()
                        st.rerun()

                st.sidebar.divider()

            # Display other sessions
            if other_sessions:
                st.sidebar.markdown("**🔄 Switch to Session:**")
                for session in other_sessions[:5]:  # Show max 5 previous sessions
                    session_id = session['session_id']
                    short_id = session_id[:8]
                    msg_count = session.get('message_count', 0)

                    col1, col2 = st.sidebar.columns([3, 1])

                    with col1:
                        if st.button(f"📁 {short_id}... ({msg_count} msgs)",
                                   key=f"switch_{session_id}",
                                   help=f"Switch to session {short_id}"):
                            self._switch_to_session(session_id)

                    with col2:
                        if st.button("🗑️",
                                   key=f"delete_{session_id}",
                                   help="Delete session"):
                            if self._delete_session(session_id):
                                st.rerun()

                if len(other_sessions) > 5:
                    st.sidebar.text(f"... and {len(other_sessions) - 5} more sessions")
            else:
                st.sidebar.text("No other sessions available")
        else:
            st.sidebar.text("No sessions found")

        # New session button
        if st.sidebar.button("➕ Create New Session", help="Start a fresh clean session"):
            st.session_state.clear()
            st.rerun()

        st.sidebar.divider()

        # Current session detailed information
        session_info = self._get_session_info()
        if session_info:
            st.sidebar.subheader("📊 Session Details")

            # Show current preferences
            preferences = session_info.get('preferences', [])
            if preferences:
                st.sidebar.markdown("**🎯 Your Preferences:**")
                for pref in preferences[:3]:  # Show first 3
                    st.sidebar.write(f"• {pref}")
                if len(preferences) > 3:
                    st.sidebar.write(f"• ... and {len(preferences) - 3} more")

            # Show read books
            read_books = session_info.get('read_books', [])
            if read_books:
                st.sidebar.markdown("**📖 Books You've Read:**")
                for book in read_books[:3]:  # Show first 3
                    st.sidebar.write(f"• {book.get('name', 'No title')}")
                if len(read_books) > 3:
                    st.sidebar.write(f"• ... and {len(read_books) - 3} more")

        # API statistics
        st.sidebar.subheader("🌐 System Statistics")
        stats = self._get_api_stats()
        if stats:
            st.sidebar.metric("Total Recommendations", stats.get('total_recommendations', 0))

    def render_chat_interface(self):
        """Render the main chat interface."""
        st.title("📚 Book Recommendation Agent")
        st.markdown("Hello! I'm your personal assistant for finding incredible books. You can ask me for recommendations, tell me about your tastes, or simply chat about literature.")

        # Initialize chat history in session_state
        if 'messages' not in st.session_state:
            st.session_state.messages = []

        # Show chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

                # Show recommended books if any
                if message.get("books"):
                    st.markdown("**📚 Recommended books:**")
                    for book in message["books"]:
                        st.markdown(self._format_book_display(book))

        # Chat input
        if prompt := st.chat_input("Type your message here..."):
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Show user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Get API response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response_data = self._send_chat_message(prompt)

                    if response_data:
                        response_text = response_data.get("response", "Sorry, I couldn't process your message.")
                        recommended_books = response_data.get("recommended_books", [])

                        st.markdown(response_text)

                        # Show recommended books if any
                        if recommended_books:
                            st.markdown("**📚 Recommended books:**")
                            for book in recommended_books:
                                st.markdown(self._format_book_display(book))

                        # Add response to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text,
                            "books": recommended_books
                        })
                    else:
                        error_msg = "Sorry, there was a problem processing your message. Please try again."
                        st.markdown(error_msg)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg
                        })

    def render_quick_recommendations(self):
        """Render the quick recommendations section."""
        st.subheader("🎯 Get Quick Recommendations")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Genre/Theme Preferences:**")
            preferences_input = st.text_area(
                "Write your favorite genres or themes (one per line):",
                placeholder="Science fiction\nMystery\nHistorical romance\nSelf-help",
                height=100
            )

        with col2:
            st.markdown("**Books you've read (optional):**")
            books_input = st.text_area(
                "Write books you've read (format: 'Title - Author', one per line):",
                placeholder="1984 - George Orwell\nOne Hundred Years of Solitude - Gabriel García Márquez",
                height=100
            )

        if st.button("🔍 Get Recommendations"):
            preferences = [p.strip() for p in preferences_input.split('\n') if p.strip()]

            # Process read books
            read_books = []
            if books_input.strip():
                for line in books_input.split('\n'):
                    if ' - ' in line:
                        title, author = line.split(' - ', 1)
                        read_books.append({"name": title.strip(), "author": author.strip()})

            with st.spinner("Finding recommendations..."):
                response_data = self._get_direct_recommendations(preferences, read_books)

                if response_data:
                    recommended_books = response_data.get("recommended_books", [])

                    if recommended_books:
                        st.success(f"✅ Found {len(recommended_books)} recommendations for you:")

                        for i, book in enumerate(recommended_books, 1):
                            st.markdown(f"{i}. {self._format_book_display(book)}")
                    else:
                        st.info("No specific recommendations found. Try the chat for more personalized suggestions.")

    def render_session_management(self):
        """Render the session management section."""
        st.subheader("⚙️ Session Management")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("📊 View Session Data"):
                session_info = self._get_session_info()
                if session_info:
                    st.json(session_info)

        with col2:
            if st.button("🗑️ Clear History"):
                st.session_state.messages = []
                st.success("History cleared")
                st.rerun()

        with col3:
            if st.button("📈 View Statistics"):
                stats = self._get_api_stats()
                if stats:
                    st.json(stats)

    def run(self):
        """Run the main application."""
        # Render sidebar
        self.render_sidebar()

        # Main tabs
        tab1, tab2, tab3 = st.tabs(["💬 Chat", "🎯 Quick Recommendations", "⚙️ Settings"])

        with tab1:
            self.render_chat_interface()

        with tab2:
            self.render_quick_recommendations()

        with tab3:
            self.render_session_management()

def main():
    """Main application function."""
    app = BookRecommenderUI()
    app.run()

if __name__ == "__main__":
    main()

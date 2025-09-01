# 📚 Book Recommender AI Agent

An intelligent AI agent that recommends books based on your previously read books, wishlist, and personal preferences. Built with LangGraph, FastAPI, and Streamlit to offer a complete book recommendation experience.

## 🚀 Key Features

- **Personalized Recommendations**: Get book suggestions based on your reading history and preferences
- **Conversational Interface**: Chat with the agent to get more specific recommendations
- **Session Management**: Maintains conversation context through persistent sessions
- **REST API**: Complete endpoints for integration with other applications
- **Web Interface**: Intuitive frontend built with Streamlit
- **Evaluation System**: Complete framework to evaluate recommendation quality

## 🏗️ Project Architecture

The project is divided into four main components:

### 1. 🤖 AI Agent (LangGraph)
**Location**: `app/graph/`

The heart of the system is an agent built with LangGraph that handles the conversation flow:

- **States** (`states.py`): Defines the agent's internal state
- **Nodes** (`nodes.py`): Processing logic for each type of action
- **Graph** (`graph.py`): Orchestrates the flow between nodes
- **Data Types** (`data_types.py`): Pydantic models for data structuring
- **Prompts** (`prompts.py`): Templates for LLM interactions

#### Agent Flow:
1. **Router**: Classifies user intent
2. **Thinking Node**: Generates recommendations using LLM
3. **Save Nodes**: Persists recommendations, preferences, or read books
4. **Summary**: Generates final interaction summary

### 2. 🧪 Evaluation System
**Location**: `tests/`

Complete framework to evaluate agent performance:

- **Node Evaluations** (`tests/nodes_evals/`):
  - `router.py`: Evaluates intent classification
  - `recommend_books.py`: Evaluates recommendation quality
  - `save_preferences.py`: Evaluates preference extraction
  - `save_read_books.py`: Evaluates reading history processing
  - `summary.py`: Evaluates summary generation
  - `talk_with_data.py`: Evaluates contextual conversations

- **Ground Truth Data** (`tests/files/`): Evaluation datasets for each component
- **Utilities** (`tests/utils/`): Evaluation tools and metrics

### 3. 🌐 REST API (FastAPI)
**Location**: `app/api/`

Robust API to interact with the agent:

- **Main Endpoints**:
  - `POST /chat`: Free conversation with the agent
  - `POST /recommend`: Get direct recommendations
  - `GET /sessions/{session_id}`: Session information
  - `DELETE /sessions/{session_id}`: Clear session

- **Features**:
  - Persistent session management
  - Pydantic data validation
  - Robust error handling
  - CORS middleware for frontend
  - SQLite database for sessions

### 4. 🖥️ Frontend (Streamlit)
**Location**: `ui/streamlit_app.py`

Intuitive web interface that includes:

- **Chat Interface**: Natural conversation with the agent
- **Session Management**: Create, switch, and delete sessions
- **Data Visualization**: Display recommendations, preferences, and history
- **Direct Input**: Forms to add preferences and read books
- **Persistent State**: Maintains context during session

## 📋 System Requirements

### Minimum Requirements
- Python 3.8+
- 4GB RAM
- Internet connection (for OpenAI API)

### Main Dependencies
- **LangGraph**: Framework for AI agents
- **LangChain**: Tools for LLM
- **FastAPI**: Web framework for API
- **Streamlit**: Framework for web interfaces
- **OpenAI**: Language model
- **ChromaDB**: Vector database
- **SQLite**: Session persistence

## 🛠️ Installation and Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd Book-recommender-AI-agent
```

### 2. Setup Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and fill in the variables:

```env
# OpenAI API Key (REQUIRED)
OPENAI_API_KEY=your_openai_api_key_here

# LangSmith (Optional - for traceability)
LANGSMITH_TRACING=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=Book-recommender-AI-agent

# Database Configuration
DATABASE_URL=sqlite:///sessions.db

# API Configuration
API_HOST=127.0.0.1
API_PORT=8000

# Streamlit Configuration
STREAMLIT_PORT=8501
```

## 🚀 Using the System

### Method 1: Quick Start (Windows)
```bash
setup.bat
```

### Method 2: Manual Start

#### 1. Start API
```bash
# Using FastAPI directly
uvicorn app.api.main:app --host 0.0.0.0 --port 8000

# Or using included script
python start_api.py
```

#### 2. Start Frontend (new terminal)
```bash
streamlit run ui/streamlit_app.py --server.port 8501
```

#### 3. Start with LangGraph Studio (Optional)
```bash
langgraph up
```

### Application Access
- **Streamlit Frontend**: http://localhost:8501
- **REST API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **LangGraph Studio**: http://localhost:8080

## 💬 Usage Examples

### Conversational Chat
```
User: "Hello, I like science fiction and I've read Dune"
Agent: "Excellent choice! Based on your love for science fiction and having read Dune, I recommend..."
```

### REST API
```bash
# Get recommendations
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "preferences": ["science fiction", "space opera"],
    "read_books": [{"title": "Dune", "author": "Frank Herbert"}]
  }'
```

## 🧪 Run Evaluations

### Individual Evaluation
```bash
# Evaluate router
python -m tests.nodes_evals.router

# Evaluate recommendations
python -m tests.nodes_evals.recommend_books
```

### Complete Evaluation
```bash
# Run all evaluations
python -m pytest tests/ -v
```

## 📊 Data Structure

### Book
```json
{
  "title": "Dune",
  "author": "Frank Herbert",
}
```

### Preferences
```json
{
  "preferences": ["science fiction", "fantasy"]
}
```

## 🐛 Troubleshooting

### Common Issues

1. **API Key Error**
   ```
   Error: OpenAI API key not found
   Solution: Verify that OPENAI_API_KEY is configured in .env
   ```

2. **API Connection Error**
   ```
   Error: Cannot connect to API
   Solution: Make sure the API is running on port 8000
   ```

3. **Database Error**
   ```
   Error: Database locked
   Solution: Close all instances and delete sessions.db.lock if it exists
   ```

### Logs and Debugging
- Logs are displayed in console during execution
- Use LangSmith for detailed traceability
- Check API documentation at `/docs`

## 🤝 Contributing

1. Fork the project
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License. See `LICENSE` for more details.

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the API documentation

---

Enjoy discovering your next great read! 📚✨

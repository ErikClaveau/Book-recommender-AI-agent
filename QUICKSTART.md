# Quick Start Guide - Book Recommender AI Agent

## 🚀 Quick Start (5 minutes)

### Step 1: Initial Setup
```bash
# 1. Clone and navigate to project
git clone <repository-url>
cd Book-recommender-AI-agent

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt
```

### Step 2: Configure Environment Variables
1. Copy `.env.example` to `.env`
2. Edit `.env` and add your OpenAI API key:
```env
OPENAI_API_KEY=your_api_key_here
```

### Step 3: Run the Application
```bash
# Windows - Use automatic script
setup.bat

# Or manually:
# Terminal 1 - API
python start_api.py

# Terminal 2 - Frontend
streamlit run ui/streamlit_app.py
```

### Step 4: Access the Application
- **Frontend**: http://localhost:8501
- **API**: http://localhost:8000

## 🐳 With Docker (Alternative)
```bash
# 1. Configure environment variables
cp .env.example .env
# Edit .env with your API key

# 2. Run with Docker Compose
docker-compose up --build
```

## 💬 First Use
1. Open http://localhost:8501
2. Type: "Hello, I like science fiction"
3. Get your first recommendations!

## 🆘 Common Issues
- **API Key Error**: Verify that OPENAI_API_KEY is in .env
- **Port occupied**: Change ports in docker-compose.yml
- **Dependencies**: Run `pip install -r requirements.txt` again

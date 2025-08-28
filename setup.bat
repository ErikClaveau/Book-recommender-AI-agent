@echo off
echo 📚 Book Recommender AI Agent - Setup
echo ==========================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo ✅ Python found
echo.

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
    echo ✅ Virtual environment created
) else (
    echo ✅ Virtual environment already exists
)

echo.
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo 📥 Installing main dependencies...
pip install -r requirements.txt

echo.
echo 🔐 Verifying configuration...
if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env"
        echo ⚠️  .env file created from .env.example
        echo ⚠️  IMPORTANT: Edit .env and add your OPENAI_API_KEY
        echo.
    ) else (
        echo ❌ Error: .env.example not found
    )
) else (
    echo ✅ .env file already exists
)

echo.
echo 🚀 Starting services...
echo.
echo Starting API on port 8000...
start "Book Recommender API" cmd /k "venv\Scripts\activate.bat && python start_api.py"

echo Waiting for API to start...
timeout /t 3 /nobreak >nul

echo.
echo Starting Streamlit interface on port 8501...
start "Book Recommender UI" cmd /k "venv\Scripts\activate.bat && streamlit run ui/streamlit_app.py --server.port 8501"

echo.
echo ==========================================
echo ✅ Setup completed!
echo.
echo 📱 Access the application at:
echo   Frontend: http://localhost:8501
echo   API: http://localhost:8000
echo   Documentation: http://localhost:8000/docs
echo.
echo ⚠️  If this is your first time, make sure to:
echo   1. Configure OPENAI_API_KEY in the .env file
echo   2. Have internet connection
echo.
echo Press any key to close...
pause >nul

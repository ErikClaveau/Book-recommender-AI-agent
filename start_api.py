#!/usr/bin/env python3
"""
Simple startup script for the Book Recommendation API.

Run this file to start the API server with default settings.
"""

if __name__ == "__main__":
    import uvicorn
    from app.api.main import app
    from app.api.config import config

    print("🚀 Starting Book Recommendation API...")
    print(f"📍 Server will be available at: http://{config.host}:{config.port}")
    print(f"📖 API Documentation: http://{config.host}:{config.port}/docs")
    print(f"🔧 ReDoc Documentation: http://{config.host}:{config.port}/redoc")

    uvicorn.run(
        "app.api.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info"
    )

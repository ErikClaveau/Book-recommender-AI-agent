#!/usr/bin/env python3
"""
Simple startup script for the Book Recommendation API.

Run this file to start the API server with default settings.
"""

if __name__ == "__main__":
    import uvicorn
    from app.api.main import app
    from app.api.config import config
    from app.utils.logger import get_logger

    logger = get_logger(__name__)

    logger.info("🚀 Starting Book Recommendation API...")
    logger.info(f"📍 Server will be available at: http://{config.host}:{config.port}")
    logger.info(f"📖 API Documentation: http://{config.host}:{config.port}/docs")
    logger.info(f"🔧 ReDoc Documentation: http://{config.host}:{config.port}/redoc")

    uvicorn.run(
        "app.api.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info"
    )

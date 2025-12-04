"""Main entry point for the search and recommendation service"""
import logging
from app.app import create_app
from app.config import Config

# Configure logging BEFORE importing anything else
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Download BGE-M3 model if needed (first run)
    logger.info("=== Starting Search and Recommendation Service ===")
    try:
        from init_model import ensure_model_downloaded
        ensure_model_downloaded()
    except Exception as e:
        logger.warning(f"Model pre-download failed: {e}")
        logger.warning("Will retry when Milvus service initializes...")
    
    # Create and run Flask app
    logger.info("Creating Flask application...")
    app = create_app()
    
    logger.info(f"Starting Flask server on {Config.FLASK_HOST}:{Config.FLASK_PORT}")
    app.run(
        host=Config.FLASK_HOST,
        port=Config.FLASK_PORT,
        debug=Config.FLASK_DEBUG,
    )
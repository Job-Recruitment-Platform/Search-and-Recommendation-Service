"""Flask application factory"""
import logging
from flask import Flask
from services.search_service import SearchService
from services.recommend import RecommendationService
from services.milvus_service import MilvusService
from app.routes import create_routes

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Create and configure Flask application"""
    app = Flask(__name__)

    # Initialize services
    milvus_service = MilvusService()
    search_service = SearchService(milvus_service)
    recommend_service = RecommendationService(milvus_service)

    # Register routes
    create_routes(app, search_service, recommend_service)

    logger.info("Flask application created successfully")
    return app


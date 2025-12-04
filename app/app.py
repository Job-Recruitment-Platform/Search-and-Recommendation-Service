"""Flask application factory"""
import logging
import threading
from flask import Flask
from services.search_service import SearchService
from services.recommend import RecommendationService
from services.milvus_service import MilvusService
from app.routes import create_routes
from sync_service.outbox_consumer import OutboxEventConsumer
from sync_service.interaction_consumer import InteractionConsumer

logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Create and configure Flask application"""
    app = Flask(__name__)

    # Initialize services
    logger.info("Initializing services...")
    try:
        milvus_service = MilvusService()
        search_service = SearchService(milvus_service)
        recommend_service = RecommendationService(milvus_service)
        logger.info("✓ Services initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {e}", exc_info=True)
        raise

    # Register routes
    create_routes(app, search_service, recommend_service)
    logger.info("✓ Routes registered")

    # ============================================
    # Start Consumer Thread 1: Outbox Events
    # ============================================
    def start_outbox_consumer():
        """Start outbox-events consumer (Job sync to Milvus)"""
        try:
            logger.info("Starting Outbox Event Consumer thread...")
            consumer = OutboxEventConsumer()
            logger.info(
                "✓ Outbox consumer initialized, processing messages...")
            consumer.run()
        except KeyboardInterrupt:
            logger.info("⚠️  Outbox consumer interrupted by user")
        except Exception as e:
            logger.error(f"❌ Outbox consumer error: {e}", exc_info=True)
            logger.warning(
                "⚠️  Outbox consumer stopped, but Flask app continues")

    outbox_thread = threading.Thread(
        target=start_outbox_consumer,
        daemon=True,
        name="OutboxEventConsumer"
    )
    outbox_thread.start()
    logger.info("✓ Outbox Event Consumer thread started")

    # ============================================
    # Start Consumer Thread 2: User Interactions
    # ============================================
    def start_interaction_consumer():
        """Start user-interactions consumer (Recommendation signals)"""
        try:
            logger.info("Starting Interaction Consumer thread...")
            consumer = InteractionConsumer()
            logger.info(
                "✓ Interaction consumer initialized, processing messages...")
            consumer.run()
        except KeyboardInterrupt:
            logger.info("⚠️  Interaction consumer interrupted by user")
        except Exception as e:
            logger.error(f"❌ Interaction consumer error: {e}", exc_info=True)
            logger.warning(
                "⚠️  Interaction consumer stopped, but Flask app continues")

    interaction_thread = threading.Thread(
        target=start_interaction_consumer,
        daemon=True,
        name="InteractionConsumer"
    )
    interaction_thread.start()
    logger.info("✓ Interaction Consumer thread started")

    logger.info(
        "✓ Flask application created successfully with 2 consumer threads")
    return app

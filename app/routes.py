"""Flask routes for the search service"""
import logging
from flask import Flask, request, jsonify
from services.search_service import SearchService
from app.config import Config
from models.search import SearchWeights

logger = logging.getLogger(__name__)


def create_routes(app: Flask, search_service: SearchService):
    """Register routes for the Flask application"""

    @app.route("/health", methods=["GET"])
    def health():
        """Health check endpoint"""
        return jsonify({"status": "healthy"}), 200

    @app.route("/search", methods=["POST"])
    def search():
        """Job search endpoint (hybrid: dense + sparse with weights)."""
        data = request.json
        query = data.get("query")

        if not query:
            return jsonify({"error": "Query is required"}), 400

        limit = int(data.get("limit", Config.SEARCH_DEFAULT_LIMIT))
        offset = int(data.get("offset", Config.SEARCH_DEFAULT_OFFSET))
        weights = SearchWeights.from_dict(data.get("weights", {"dense": 1.0, "sparse": 1.0}))

        try:
            results, pagination = search_service.search(
                query=query,
                limit=limit,
                offset=offset,
                dense_weight=weights.dense,
                sparse_weight=weights.sparse,
            )
            return jsonify({
                "results": results,
                "pagination": pagination.to_dict()
            }), 200
        except Exception as e:
            logger.exception("Search failed")
            return jsonify({"error": str(e)}), 500


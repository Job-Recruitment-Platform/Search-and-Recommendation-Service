"""Flask routes for the search service"""
import logging
import json
from flask import Flask, request, jsonify
from services.search_service import SearchService
from services.recommend import RecommendationService
from app.config import Config
from models.search import SearchWeights

logger = logging.getLogger(__name__)


def create_routes(app: Flask, search_service: SearchService, recommend_service: RecommendationService):
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
        threshold = float(data.get("threshold", Config.SEARCH_THRESHOLD))
        filters = data.get("filters", None)
        try:
            job_ids, pagination = search_service.search(
                query=query,
                limit=limit,
                offset=offset,
                filters=filters,
                threshold=threshold,
            )
            return jsonify({
                "jobIds": job_ids,
                "pagination": pagination.to_dict()
            }), 200
        except Exception as e:
            logger.exception("Search failed")
            return jsonify({"error": str(e)}), 500

    @app.route("/recommend", methods=["GET"])
    def recommend():
        """Recommendation endpoint (hybrid CF + content-based với exploration)."""
        try:
            top_k = int(request.args.get("top_k", 20))
            user_id = int(request.args.get("user_id", 0))
            
            recommendations = recommend_service.recommend(
                user_id=user_id,
                top_k=top_k
            )
            
            return jsonify({
                "code": 1000,
                "data": {
                    "user_id": user_id,
                    "recommendations": recommendations,
                    "count": len(recommendations)
                }
            }), 200
            
        except Exception as e:
            logger.exception("Recommendation failed")
            return jsonify({
                "code": 5000,
                "error": str(e)
            }), 500

    @app.route("/internal/reload-model", methods=["POST"])
    def reload_model():
        token = request.headers.get("X-Internal-Token")
        expected_token = getattr(Config, "INTERNAL_API_TOKEN", None)
        
        if expected_token and token != expected_token:
             return jsonify({"error": "Unauthorized"}), 403
        
        success = recommend_service.reload_model()
        if success:
            return jsonify({"status": "success", "message": "Model reloaded"}), 200
        else:
            return jsonify({"status": "error", "message": "Failed to reload model"}), 500

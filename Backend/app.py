import logging
from flask import Flask, jsonify
from flask_cors import CORS
from datetime import datetime

from config import get_config
from db import get_db
from Service.training_service import TrainingService
from Service.prediction_service import PredictionService
from Service.retention_training_service import RetentionTrainingService
from Service.retention_prediction_service import RetentionPredictionService
from Service.schedule_service import ScheduleService
from Service.performance_service import PerformanceService

from Blueprint.practice import practice_bp
from Blueprint.real_exam import real_exam_bp
from Blueprint.analysis import analysis_bp
from Blueprint.dashboard import dashboard_bp
from Blueprint.retention import retention_bp
from Blueprint.internal import internal_bp
from Blueprint.schedule import schedule_bp
from Blueprint.performance import performance_bp

from Utils.logging import setup_logging


def create_app(config_class=None):
    """Application Factory"""

    if config_class is None:
        config_class = get_config()

    app = Flask(__name__)
    app.config.from_object(config_class)

    # Logging
    setup_logging(app)
    logger = logging.getLogger(__name__)

    # CORS
    CORS(app, origins=app.config.get("CORS_ORIGINS", "*"))

    # Database
    db = get_db()
    logger.info("Database initialized")

    # Services
    app.training_service = TrainingService(app.config)
    app.prediction_service = PredictionService(app.config)

    app.retention_training_service = RetentionTrainingService(app.config)
    app.retention_prediction_service = RetentionPredictionService(app.config)

    app.schedule_service = ScheduleService(app.config)
    app.performance_service = PerformanceService(app.config)

    # Blueprints
    app.register_blueprint(practice_bp, url_prefix="/api/practice")
    app.register_blueprint(real_exam_bp, url_prefix="/api/real-exam")
    app.register_blueprint(analysis_bp, url_prefix="/api/analysis")
    app.register_blueprint(dashboard_bp, url_prefix="/api/dashboard")

    app.register_blueprint(retention_bp, url_prefix="/api/retention")
    app.register_blueprint(schedule_bp, url_prefix="/api/retention/schedule")
    app.register_blueprint(internal_bp, url_prefix="/api/retention/internal")
    app.register_blueprint(performance_bp, url_prefix="/api/retention/performance")

    @app.route("/api/health")
    def health():
        return jsonify({
            "success": True,
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "database": "connected" if get_db().client else "disconnected",
            "models": [
                "learning_velocity",
                "burnout_risk",
                "adaptive_scheduling"
            ]
        })

    @app.errorhandler(404)
    def not_found(error):
        return jsonify({
            "success": False,
            "error": "Resource not found"
        }), 404

    @app.errorhandler(500)
    def internal_error(error):
        logger.exception(error)
        return jsonify({
            "success": False,
            "error": "Internal server error"
        }), 500

    logger.info("Application initialized successfully")
    return app


# Gunicorn entrypoint
app = create_app()

# Local development only
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5500,
        debug=True
    )
    
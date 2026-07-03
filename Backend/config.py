import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Base configuration for the application."""

    # MongoDB Configuration
    MONGODB_URI = os.getenv('MONGODB_URI')
    MONGODB_DB_NAME = os.getenv('MONGODB_DB_NAME', 'studystream')

    # Application Configuration
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

    # CORS Configuration
    CORS_ORIGINS = os.getenv('CORS_ORIGINS', '*').split(',')

    # Feature definitions
    PRACTICE_FEATURES = [
        'accuracy', 'normalized_response_time', 'rolling_time_variance',
        'answer_change_count', 'stress_score', 'confidence_index',
        'concept_mastery_score', 'current_question_difficulty',
        'consecutive_correct_streak', 'fatigue_indicator',
        'focus_loss_frequency', 'preferred_difficulty_offset'
    ]
    PRACTICE_TARGET = 'next_difficulty'
    SEQUENCE_LENGTH_PRACTICE = 10
    PRACTICE_FEATURES_COUNT = len(PRACTICE_FEATURES)

    GLOBAL_FEATURES = [
        'session_accuracy_avg', 'avg_solved_difficulty', 'max_difficulty_sustained',
        'performance_trend_slope', 'retention_score', 'burnout_risk_index',
        'stress_trend_slope', 'concept_coverage_ratio', 'high_difficulty_accuracy',
        'consistency_index', 'avg_response_time_trend', 'serious_test_performance_score'
    ]
    GLOBAL_TARGET = 'readiness_difficulty_score'
    SEQUENCE_LENGTH_GLOBAL = 5
    GLOBAL_FEATURES_COUNT = len(GLOBAL_FEATURES)

    EXAM_FEATURES = 8
    SEQUENCE_LENGTH_EXAM = 5

    LEARNING_VELOCITY_FEATURES = 9
    SEQUENCE_LENGTH_DAILY = 30

    BURNOUT_RISK_FEATURES = 11
    SEQUENCE_LENGTH_SESSION = 14

    # Training parameters
    MIN_PRACTICE_SAMPLES = 100
    MODEL_RETRAIN_INTERVAL_ROWS = 100
    PRACTICE_RETRAIN_INTERVAL = 100
    MIN_PRACTICE_SAMPLES_FOR_GLOBAL = 40
    MIN_EXAM_SAMPLES = 5
    EPOCHS = 100
    BATCH_SIZE = 32
    SEQUENCE_LENGTH_PRACTICE=50
    
    # Retention Model Configuration
    RETRAIN_COOLDOWN_SECONDS = 120
    EXPORT_TFLITE_MODELS = False

    MODEL_CONFIG = {
        'micro': {
            'name': 'micro_lstm',
            'sequence_length': 20,
            'n_features': 15,
            'epochs': 100,
            'batch_size': 32,
            'learning_rate': 0.001,
            'min_samples': 20,
            'retrain_interval': 5
        },
        'meso': {
            'name': 'meso_lstm',
            'sequence_length': 30,
            'n_temporal_features': 10,
            'n_metadata_features': 18,
            'epochs': 80,
            'batch_size': 16,
            'learning_rate': 0.001,
            'min_samples': 7,
            'retrain_interval': 5
        },
        'macro': {
            'name': 'macro_lstm',
            'encoder_units': 256,
            'decoder_units': 256,
            'n_topics': 100,
            'epochs': 60,
            'batch_size': 16,
            'learning_rate': 0.001,
            'min_samples': 30,
            'retrain_interval': 5
        }
    }

    SUBJECTS = {
        'Data Structires and Algorithms': ['DSA', 'Data Structures', 'Algorithms'],
        'Operating Systems': ['OS', 'Operating Systems'],
        'Programming Languages': ['Python', 'Java', 'C++'],
        'Subject':['Topic1', 'Topic2', 'Topic3'],
        'Database Management Systems': ['DBMS', 'Databases', 'Database Management'],
        'Computer Networks': ['Networking', 'Computer Networks'],
    }

    RETENTION_THRESHOLDS = {
        'critical': 0.3,
        'warning': 0.5,
        'moderate': 0.7,
        'good': 0.85,
        'excellent': 0.95
    }

    REPETITION_SCHEDULES = {
        'immediate': {
            'retention_range': (0, 0.3),
            'batch_size': 3,
            'schedule_type': 'immediate_review',
            'questions_per_topic': 3,
            'description': 'Review now - Critical retention'
        },
        'short_term': {
            'retention_range': (0.3, 0.5),
            'batch_size': 5,
            'schedule_type': 'next_session',
            'questions_per_topic': 4,
            'description': 'Review in next session'
        },
        'medium_term': {
            'retention_range': (0.5, 0.7),
            'batch_size': 8,
            'schedule_type': 'next_day',
            'questions_per_topic': 3,
            'description': 'Review tomorrow'
        },
        'long_term': {
            'retention_range': (0.7, 0.85),
            'batch_size': 10,
            'schedule_type': 'in_3_days',
            'questions_per_topic': 2,
            'description': 'Review in 3 days'
        },
        'mastered': {
            'retention_range': (0.85, 1.0),
            'batch_size': 15,
            'schedule_type': 'in_week',
            'questions_per_topic': 1,
            'description': 'Review weekly'
        }
    }

    FORGETTING_CURVE = {
        'time_points': [1, 3, 7, 14, 30, 60, 90],
        'decay_factor_range': (0.1, 0.3),
        'reinforcement_boost': 0.15
    }

    NODE_API = {
        'base_url': 'http://localhost:5000',
        'endpoints': {
            'initial_predictions': '/api/ml/initial-predictions',
            'retention_update': '/api/ml/retention-update',
            'batch_complete': '/api/ml/batch-complete',
            'performance_metrics': '/api/ml/performance-metrics',
            'schedule_update': '/api/ml/schedule-update',
            'question_sequence': '/api/ml/question-sequence',
            'stress_fatigue_update': '/api/ml/stress-fatigue-update',
            'health_check': '/api/ml/health'
        },
        'timeout': 5,
        'retry_attempts': 3
    }


class TestConfig(Config):
    """Test configuration."""
    TESTING = True
    MONGODB_DB_NAME = 'studystream_test'


class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False


# Configuration mapping
config_map = {
    'development': Config,
    'testing': TestConfig,
    'production': ProductionConfig,
    'default': Config
}

def get_config():
    """Get configuration based on environment."""
    env = os.getenv('FLASK_ENV', 'development')
    return config_map.get("default", Config)

"""Constants used throughout the application."""

# Model configuration
MODEL_CONFIG = {
    'micro': {
        'sequence_length': 20,
        'n_features': 15,
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001
    },
    'meso': {
        'sequence_length': 30,
        'n_temporal_features': 10,
        'n_metadata_features': 18,
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001
    },
    'macro': {
        'encoder_units': 256,
        'decoder_units': 256,
        'n_topics': 100,
        'epochs': 50,
        'batch_size': 32,
        'learning_rate': 0.001
    }
}

# Retention thresholds
RETENTION_THRESHOLDS = {
    'critical': 0.3,
    'warning': 0.5,
    'moderate': 0.7,
    'good': 0.85,
    'excellent': 0.95
}

# Review intervals in days
REVIEW_INTERVALS = {
    'immediate': 0,
    'next_day': 1,
    'three_days': 3,
    'one_week': 7,
    'two_weeks': 14,
    'one_month': 30,
    'three_months': 90
}

# Difficulty levels
DIFFICULTY_LEVELS = {
    1: 'very_easy',
    2: 'easy',
    3: 'medium',
    4: 'hard',
    5: 'very_hard'
}

# HTTP status codes
HTTP_STATUS = {
    'ok': 200,
    'created': 201,
    'bad_request': 400,
    'unauthorized': 401,
    'forbidden': 403,
    'not_found': 404,
    'server_error': 500
}

# Error messages
ERROR_MESSAGES = {
    'missing_user_id': 'User ID is required',
    'missing_subject': 'Subject is required',
    'missing_topic': 'Topic ID is required',
    'invalid_data': 'Invalid data format',
    'model_not_found': 'Model not found for this user',
    'training_failed': 'Model training failed',
    'prediction_failed': 'Prediction generation failed',
    'unauthorized': 'Unauthorized access',
    'server_error': 'Internal server error'
}

# Stress and fatigue thresholds
STRESS_THRESHOLDS = {
    'low': 0.3,
    'moderate': 0.6,
    'high': 0.8
}

FATIGUE_THRESHOLDS = {
    'low': 0.3,
    'moderate': 0.6,
    'high': 0.8
}

FOCUS_THRESHOLDS = {
    'low': 0.4,
    'moderate': 0.7,
    'high': 0.9
}

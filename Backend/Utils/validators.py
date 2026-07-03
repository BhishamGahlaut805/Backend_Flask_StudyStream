"""Data validation utilities."""

import re
import logging
from typing import List, Dict, Any, Union

logger = logging.getLogger(__name__)


def validate_student_id(student_id: str) -> bool:
    """Validate student ID format."""
    if not student_id or not isinstance(student_id, str):
        raise ValueError("Student ID is required and must be a string")

    # Allow alphanumeric, underscore, hyphen, and course_ prefix
    pattern = r'^[a-zA-Z0-9_\-]+$|^course_[a-zA-Z0-9_\-]+$'
    if not re.match(pattern, student_id):
        raise ValueError(f"Invalid student ID format: {student_id}")

    return True


def validate_subject(subject: str) -> bool:
    """Validate subject name."""
    if not subject or not isinstance(subject, str):
        raise ValueError("Subject is required and must be a string")

    valid_subjects = {'default1', 'default2'}
    normalized = subject.strip().lower()

    # Allow course IDs
    if re.match(r'^[0-9a-f]{24}$', normalized) or normalized.startswith('course_'):
        return True

    if normalized not in valid_subjects:
        raise ValueError(f"Invalid subject: {subject}. Must be one of: {', '.join(valid_subjects)}")

    return True


def validate_features(features: List[float], expected_count: int) -> bool:
    """Validate feature list."""
    if not isinstance(features, list):
        raise ValueError("Features must be a list")

    if len(features) < expected_count:
        raise ValueError(f"Features must have at least {expected_count} elements, got {len(features)}")

    for f in features:
        if not isinstance(f, (int, float)):
            raise ValueError(f"All features must be numbers, got {type(f)}")
        if f < 0 or f > 1:
            raise ValueError(f"All features must be between 0 and 1, got {f}")

    return True


def validate_json_request(data: Dict, required_fields: List[str]) -> Dict:
    """Validate JSON request has required fields."""
    if not data or not isinstance(data, dict):
        raise ValueError("Request data must be a JSON object")

    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
        if data[field] is None:
            raise ValueError(f"Field '{field}' cannot be null")

    return data


def validate_practice_attempts(attempts: List[Dict]) -> bool:
    """Validate practice attempts data."""
    if not isinstance(attempts, list):
        raise ValueError("Attempts must be a list")

    required_fields = ['timestamp', 'correct', 'time_spent', 'difficulty']

    for i, attempt in enumerate(attempts):
        for field in required_fields:
            if field not in attempt:
                raise ValueError(f"Attempt {i} missing required field: {field}")

        # Validate field types
        if not isinstance(attempt.get('correct'), bool):
            raise ValueError(f"Attempt {i}: 'correct' must be boolean")

        if not isinstance(attempt.get('time_spent'), (int, float)):
            raise ValueError(f"Attempt {i}: 'time_spent' must be a number")

        if attempt.get('time_spent', 0) < 0:
            raise ValueError(f"Attempt {i}: 'time_spent' cannot be negative")

        if not isinstance(attempt.get('difficulty'), (int, float)):
            raise ValueError(f"Attempt {i}: 'difficulty' must be a number")

        if attempt.get('difficulty', 0) < 0 or attempt.get('difficulty', 1) > 1:
            raise ValueError(f"Attempt {i}: 'difficulty' must be between 0 and 1")

    return True


def validate_retention_session(data: Dict) -> bool:
    """Validate retention session data."""
    required_fields = ['student_id', 'subject']

    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    validate_student_id(data['student_id'])
    validate_subject(data['subject'])

    return True


def sanitize_string(value: str) -> str:
    """Sanitize a string for storage."""
    if not value:
        return ''
    # Remove any control characters
    return ''.join(char for char in str(value) if ord(char) >= 32 or char in '\n\r\t')


def validate_email(email: str) -> bool:
    """Validate email format."""
    if not email:
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

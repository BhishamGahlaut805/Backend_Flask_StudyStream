"""Helper utility functions."""

import hashlib
import re
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


def generate_user_id(email: str) -> str:
    """Generate a unique user ID from email."""
    hash_obj = hashlib.md5(email.encode())
    return f"usr_{hash_obj.hexdigest()[:8]}"


def difficulty_level_from_value(value: float) -> str:
    """Map numeric difficulty to difficulty level string."""
    score = float(max(0.0, min(1.0, value)))
    if score < 0.3:
        return 'easy'
    elif score < 0.5:
        return 'medium-easy'
    elif score < 0.7:
        return 'medium-hard'
    else:
        return 'hard'


def bounded_smooth_difficulty(current_diff: float, predicted_diff: float, max_step: float = 0.08) -> float:
    """Smooth difficulty shift with bounded per-step delta."""
    current = float(np.clip(current_diff, 0.2, 0.95))
    predicted = float(np.clip(predicted_diff, 0.2, 0.95))

    if predicted > current + max_step:
        return current + max_step
    elif predicted < current - max_step:
        return current - max_step
    return predicted


def extract_last_training_info(metadata_history: List[Dict]) -> tuple:
    """Extract last training rows and timestamp from metadata history."""
    last_rows = None
    last_at = None

    if metadata_history:
        for item in reversed(metadata_history):
            if not isinstance(item, dict):
                continue
            if last_rows is None and item.get('feature_rows_at_training') is not None:
                try:
                    last_rows = int(item.get('feature_rows_at_training'))
                except Exception:
                    pass
            if last_at is None and item.get('timestamp'):
                last_at = item.get('timestamp')
            if last_rows is not None and last_at is not None:
                break

    return last_rows, last_at


def normalize_subject(subject: str) -> str:
    """Normalize subject string."""
    if not subject:
        return ''
    subject_str = str(subject).strip().lower()
    # If it's a valid ObjectId format (24 hex chars), treat as course ID
    if re.match(r'^[0-9a-f]{24}$', subject_str):
        return 'course_' + subject_str
    return subject_str


def get_display_subject(subject: str) -> str:
    """Get display subject for internal use."""
    normalized = normalize_subject(subject)
    if normalized.startswith('course_'):
        return 'default1'
    return normalized if normalized in {'default1', 'default2'} else 'default1'


def default_topics_for_subject(subject: str) -> List[str]:
    """Get default topics for a subject."""
    subject = normalize_subject(subject)
    if subject == 'default1':
        return ['vocabulary', 'idioms', 'phrases', 'synonyms', 'antonyms', 'one_word_substitution']
    elif subject == 'default2':
        return ['history', 'geography', 'science', 'current_affairs']
    return ['general']


def get_datetime_series(df: pd.DataFrame) -> Optional[pd.Series]:
    """Parse datetime series from first available time column."""
    if df is None or df.empty:
        return None

    for col in ['timestamp', 'submitted_at', 'created_at', 'date']:
        if col in df.columns:
            parsed = pd.to_datetime(df[col], errors='coerce')
            parsed = parsed.dropna()
            if not parsed.empty:
                return parsed
    return None


def get_last_datetime_iso(df: pd.DataFrame) -> Optional[str]:
    """Return last datetime as ISO string."""
    series = get_datetime_series(df)
    if series is None or series.empty:
        return None
    return series.iloc[-1].isoformat()


def calculate_streak(practice_df: pd.DataFrame) -> int:
    """Calculate current practice streak in days."""
    if practice_df.empty or 'timestamp' not in practice_df.columns:
        return 0

    dates = pd.to_datetime(practice_df['timestamp']).dt.date.unique()
    dates = sorted(dates, reverse=True)

    if not dates:
        return 0

    today = datetime.now().date()
    if dates[0] != today:
        return 0

    streak = 1
    for i in range(1, len(dates)):
        if (dates[i-1] - dates[i]).days == 1:
            streak += 1
        else:
            break

    return streak


def timer_frame_from_retention(retention_score: float) -> int:
    """Get timer frame in seconds from retention score."""
    score = float(np.clip(retention_score, 0.0, 1.0))
    if score < 0.30:
        return 30
    elif score < 0.45:
        return 60
    elif score < 0.55:
        return 120
    elif score < 0.65:
        return 300
    elif score < 0.75:
        return 600
    elif score < 0.88:
        return 3600
    return 7200

def timer_frame_label(seconds: int) -> str:
    """Get timer frame label from seconds."""
    labels = {
        30: "30_seconds",
        60: "1_minute",
        120: "2_minutes",
        300: "5_minutes",
        600: "10_minutes",
        3600: "1_hour",
        7200: "2_hours"
    }
    return labels.get(int(seconds), f"{int(seconds)}_seconds")

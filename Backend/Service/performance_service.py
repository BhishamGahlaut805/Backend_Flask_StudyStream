"""Performance Service - Computes retention performance metrics."""

import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

from Service.data_manager import StudentDataManager

logger = logging.getLogger(__name__)


class PerformanceService:
    """Computes retention-related performance metrics."""

    def __init__(self, config):
        self.config = config
        logger.info("PerformanceService initialized")

    def _get_data_manager(self, user_id: str) -> StudentDataManager:
        """Get data manager for user."""
        return StudentDataManager(user_id)

    def calculate_all_metrics(self, user_id: str, days: int = 30,
                             subject: Optional[str] = None) -> Dict:
        """Calculate all performance metrics."""
        try:
            data_manager = self._get_data_manager(user_id)
            interactions = data_manager.load_interactions()
            daily = data_manager.load_daily_aggregates()

            if interactions.empty:
                return self._default_metrics(user_id, subject)

            accuracy = float(interactions.get('correct', pd.Series([0.5])).mean())
            stress = float(interactions.get('stress_level', pd.Series([0.3])).mean())
            fatigue = float(interactions.get('fatigue_index', pd.Series([0.3])).mean())

            return {
                'timestamp': datetime.now().isoformat(),
                'user_id': user_id,
                'subject': subject,
                'learning_velocity': {'value': 0.0, 'trend': 'stable'},
                'retention_rate': {'overall': accuracy, 'trend': 'stable'},
                'stress_pattern': {
                    'average_stress': stress,
                    'max_stress': min(1.0, stress * 1.5),
                    'risk_level': 'high' if stress > 0.6 else 'medium' if stress > 0.4 else 'low'
                },
                'fatigue_index': {
                    'average_fatigue': fatigue,
                    'current_fatigue': fatigue,
                    'risk_level': 'high' if fatigue > 0.6 else 'medium' if fatigue > 0.4 else 'low'
                },
                'focus_score': {'average': 0.5, 'trend': 'stable'},
                'confidence_trend': {'average': 0.5, 'trend': 'stable'},
                'mastery_progress': {'overall': accuracy, 'topics_mastered': 0, 'topics_struggling': 0},
                'efficiency_score': {'score': 0.5},
                'consistency_index': {'score': 0.5},
                'momentum_score': {'score': 0.5, 'direction': 'neutral'},
                'learning_efficiency': {'score': 0.5, 'rank': 'beginner', 'percentile': 50}
            }

        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return self._default_metrics(user_id, subject)

    def _default_metrics(self, user_id: str, subject: Optional[str] = None) -> Dict:
        """Return default metrics."""
        return {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'subject': subject,
            'learning_velocity': {'value': 0.0, 'trend': 'stable'},
            'retention_rate': {'overall': 0.5, 'trend': 'stable'},
            'stress_pattern': {
                'average_stress': 0.3,
                'max_stress': 0.3,
                'risk_level': 'low'
            },
            'fatigue_index': {
                'average_fatigue': 0.3,
                'current_fatigue': 0.3,
                'risk_level': 'low'
            },
            'focus_score': {'average': 0.5, 'trend': 'stable'},
            'confidence_trend': {'average': 0.5, 'trend': 'stable'},
            'mastery_progress': {'overall': 0.5, 'topics_mastered': 0, 'topics_struggling': 0},
            'efficiency_score': {'score': 0.5},
            'consistency_index': {'score': 0.5},
            'momentum_score': {'score': 0.5, 'direction': 'neutral'},
            'learning_efficiency': {'score': 0.5, 'rank': 'beginner', 'percentile': 50}
        }

    def get_metrics_summary(self, user_id: str, subject: Optional[str] = None) -> Dict:
        """Get metrics summary for dashboard."""
        metrics = self.calculate_all_metrics(user_id, 30, subject)
        return {
            'overall_score': round(metrics['learning_efficiency']['score'] * 100, 1),
            'rank': metrics['learning_efficiency']['rank'],
            'retention': round(metrics['retention_rate']['overall'] * 100, 1),
            'stress_level': round(metrics['stress_pattern']['average_stress'] * 100, 1),
            'fatigue_level': round(metrics['fatigue_index']['average_fatigue'] * 100, 1),
            'focus_level': round(metrics['focus_score']['average'] * 100, 1),
            'momentum': metrics['momentum_score']['direction']
        }

    def get_subject_comparison(self, user_id: str) -> Dict:
        """Get performance comparison across subjects."""
        data_manager = self._get_data_manager(user_id)
        interactions = data_manager.load_interactions()

        if interactions.empty or 'subject' not in interactions.columns:
            return {'subjects': {}, 'best_subject': None, 'needs_support': None}

        subjects = {}
        for subject, sdf in interactions.groupby('subject'):
            acc = float(sdf.get('correct', pd.Series([0.5])).mean())
            subjects[str(subject)] = {
                'accuracy': round(acc, 3),
                'interactions': int(len(sdf))
            }

        ranked = sorted(subjects.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        best_subject = ranked[0][0] if ranked else None
        needs_support = ranked[-1][0] if ranked else None

        return {
            'subjects': subjects,
            'best_subject': best_subject,
            'needs_support': needs_support
        }
        
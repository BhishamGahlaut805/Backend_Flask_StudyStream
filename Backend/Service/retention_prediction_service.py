"""Retention Prediction Service - Serves retention predictions."""

import logging
import json
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from Service.data_manager import StudentDataManager

logger = logging.getLogger(__name__)


class RetentionPredictionService:
    """Handles retention prediction retrieval and generation."""

    def __init__(self, config):
        self.config = config
        logger.info("RetentionPredictionService initialized")

    def _get_data_manager(self, user_id: str) -> StudentDataManager:
        """Get data manager for user."""
        return StudentDataManager(user_id)

    def get_all_predictions(self, user_id: str, subject: Optional[str] = None) -> Dict:
        """Get all retention predictions."""
        try:
            data_manager = self._get_data_manager(user_id)

            # ==================== FIX: Load predictions from MongoDB ====================
            micro = data_manager.load_predictions('micro') or []
            meso = data_manager.load_predictions('meso') or []
            macro = data_manager.load_predictions('macro') or {}

            # If no predictions exist, generate default ones
            if not micro:
                micro = self._generate_default_micro_predictions(user_id, subject)
                data_manager.save_predictions('micro', micro)

            if not meso:
                meso = self._generate_default_meso_predictions(user_id, subject)
                data_manager.save_predictions('meso', meso)

            if not macro:
                macro = self._generate_default_macro_predictions(user_id, subject)
                data_manager.save_predictions('macro', macro)
            # ==================== END OF FIX ====================

            return {
                "user_id": user_id,
                "subject": subject,
                "timestamp": datetime.now().isoformat(),
                "micro": self._format_micro_predictions(micro),
                "meso": self._format_meso_predictions(meso),
                "macro": self._format_macro_predictions(macro),
                "forgetting_curves": self.generate_forgetting_curves(user_id, micro)
            }

        except Exception as e:
            logger.error(f"Error getting predictions: {e}")
            return {"user_id": user_id, "micro": [], "meso": [], "macro": {}}

    # retention_prediction_service.py - Fixed _generate_default_micro_predictions method

    def _generate_default_micro_predictions(self, user_id: str, subject: Optional[str] = None) -> List[Dict]:
        """Generate default micro predictions from interactions."""
        try:
            data_manager = self._get_data_manager(user_id)
            interactions = data_manager.load_interactions()

            if interactions.empty:
                return [
                    {
                        'topic_id': 'general',
                        'subject': subject or 'general',
                        'current_retention': 0.5,
                        'next_retention': 0.5,
                        'stress_impact': 0.3,
                        'fatigue_level': 0.3,
                        'repeat_in_seconds': 300,
                        'batch_type': 'medium_term'
                    }
                ]

            # ==================== FIX: Check column names properly ====================
            # Use 'concept' or 'concept_area' or 'topic' as topic_id
            topic_col = None
            for col in ['topic_id', 'concept', 'concept_area', 'topic', 'subject']:
                if col in interactions.columns:
                    topic_col = col
                    break

            if topic_col is None:
                # No topic column, use a single default topic
                return [{
                    'topic_id': 'general',
                    'subject': subject or 'general',
                    'current_retention': float(interactions.get('correct', [0.5]).mean() if 'correct' in interactions else 0.5),
                    'next_retention': float(interactions.get('correct', [0.5]).mean() * 0.9 if 'correct' in interactions else 0.45),
                    'stress_impact': 0.3,
                    'fatigue_level': 0.3,
                    'repeat_in_seconds': 300,
                    'batch_type': 'medium_term'
                }]

            # Group by topic and compute retention
            predictions = []
            for topic, group in interactions.groupby(topic_col):
                # Safely get accuracy
                if 'correct' in group.columns:
                    accuracy = group['correct'].mean()
                elif 'isCorrect' in group.columns:
                    accuracy = group['isCorrect'].mean()
                else:
                    accuracy = 0.5

                # Safely get stress and fatigue
                stress = group['stress_level'].mean() if 'stress_level' in group.columns else 0.3
                fatigue = group['fatigue_index'].mean() if 'fatigue_index' in group.columns else 0.3

                # Get subject
                subject_val = subject or 'general'
                if 'subject' in group.columns:
                    subject_val = group['subject'].iloc[0]
                elif 'concept_area' in group.columns:
                    subject_val = group['concept_area'].iloc[0]

                predictions.append({
                    'topic_id': str(topic),
                    'subject': str(subject_val),
                    'current_retention': float(accuracy),
                    'next_retention': float(accuracy * 0.9),
                    'stress_impact': float(stress),
                    'fatigue_level': float(fatigue),
                    'repeat_in_seconds': 300,
                    'batch_type': 'medium_term'
                })
            # ==================== END OF FIX ====================

            return predictions

        except Exception as e:
            logger.error(f"Error generating default micro predictions: {e}")
            return []

    def _generate_default_meso_predictions(self, user_id: str, subject: Optional[str] = None) -> List[Dict]:
        """Generate default meso predictions."""
        try:
            data_manager = self._get_data_manager(user_id)
            interactions = data_manager.load_interactions()

            if interactions.empty or 'subject' not in interactions.columns:
                return []

            predictions = []
            for subject_name, group in interactions.groupby('subject'):
                accuracy = group['correct'].mean() if 'correct' in group else 0.5
                predictions.append({
                    'subject': str(subject_name),
                    'topic_id': 'general',
                    'retention_7d': float(accuracy),
                    'retention_30d': float(accuracy * 0.85),
                    'retention_90d': float(accuracy * 0.7)
                })

            return predictions

        except Exception as e:
            logger.error(f"Error generating default meso predictions: {e}")
            return []

    def _generate_default_macro_predictions(self, user_id: str, subject: Optional[str] = None) -> Dict:
        """Generate default macro predictions."""
        try:
            data_manager = self._get_data_manager(user_id)
            interactions = data_manager.load_interactions()

            if interactions.empty:
                return {
                    'projected_retention': 0.5,
                    'burnout_risk': 0.3,
                    'optimal_daily_minutes': 60,
                    'weekly_structure': {
                        'revision_days': ['Monday', 'Thursday'],
                        'new_learning_days': ['Tuesday', 'Wednesday', 'Friday', 'Saturday'],
                        'light_review_day': 'Sunday'
                    }
                }

            accuracy = interactions['correct'].mean() if 'correct' in interactions else 0.5
            stress = interactions['stress_level'].mean() if 'stress_level' in interactions else 0.3

            return {
                'projected_retention': float(accuracy),
                'burnout_risk': float(stress),
                'optimal_daily_minutes': 60,
                'weekly_structure': {
                    'revision_days': ['Monday', 'Thursday'] if accuracy < 0.6 else ['Monday', 'Thursday'],
                    'new_learning_days': ['Tuesday', 'Wednesday', 'Friday', 'Saturday'],
                    'light_review_day': 'Sunday'
                }
            }

        except Exception as e:
            logger.error(f"Error generating default macro predictions: {e}")
            return {}

    def _format_micro_predictions(self, micro: List[Dict]) -> List[Dict]:
        """Format micro predictions for frontend."""
        formatted = []
        for pred in micro:
            repeat_seconds = int(pred.get("repeat_in_seconds", 300))
            formatted.append({
                "topic_id": pred.get("topic_id", "unknown"),
                "subject": pred.get("subject", "unknown"),
                "retention_probability": pred.get("current_retention", 0.5),
                "next_question_difficulty": self._calculate_next_difficulty(pred.get("current_retention", 0.5)),
                "probability_correct_next": pred.get("next_retention", 0.5),
                "stress_impact": pred.get("stress_impact", 0.3),
                "fatigue_level": pred.get("fatigue_level", 0.3),
                "repeat_in_seconds": repeat_seconds,
                "repeat_in_days": repeat_seconds / 86400,
                "batch_type": pred.get("batch_type", "medium_term")
            })
        return formatted

    def _format_meso_predictions(self, meso: List[Dict]) -> List[Dict]:
        """Format meso predictions for frontend."""
        formatted = []
        for pred in meso:
            formatted.append({
                "subject": pred.get("subject", "unknown"),
                "topic_id": pred.get("topic_id", "unknown"),
                "subject_retention_score": pred.get("retention_7d", 0.5),
                "retention_7d": pred.get("retention_7d", 0.5),
                "retention_30d": pred.get("retention_30d", 0.5),
                "retention_90d": pred.get("retention_90d", 0.5)
            })
        return formatted

    def _format_macro_predictions(self, macro: Dict) -> Dict:
        """Format macro predictions for frontend."""
        return {
            "optimal_daily_study_schedule": macro.get("weekly_structure", {}),
            "subject_priority_order": macro.get("subject_priority_order", []),
            "predicted_long_term_retention_score": macro.get("projected_retention", 0.5),
            "fatigue_risk_probability": macro.get("burnout_risk", 0.3),
            "burnout_status": macro.get("fatigue_burnout_check", {}).get("status", "low"),
            "recommended_break_minutes": macro.get("fatigue_burnout_check", {}).get("recommended_break_minutes", 10),
            "optimal_daily_minutes": macro.get("optimal_daily_minutes", 60)
        }

    def _calculate_next_difficulty(self, retention: float) -> int:
        """Calculate next question difficulty based on retention."""
        if retention < 0.3:
            return 1
        elif retention < 0.5:
            return 2
        elif retention < 0.7:
            return 3
        elif retention < 0.85:
            return 4
        else:
            return 5

    def generate_forgetting_curves(self, user_id: str, micro_predictions: List[Dict]) -> Dict:
        """Generate forgetting curves."""
        curves = {}
        time_points = self.config.FORGETTING_CURVE.get("time_points", [1, 3, 7, 14, 30])

        for pred in micro_predictions:
            topic_id = str(pred.get("topic_id", "unknown_topic"))
            current_retention = float(pred.get("current_retention", 0.5))
            tau = 30 * (1 + current_retention)

            points = []
            for day in time_points:
                retention = current_retention * np.exp(-day / tau)
                retention = float(min(1.0, max(0.0, retention)))
                points.append({
                    "day": int(day),
                    "retention": round(retention, 2),
                    "needs_review": retention < 0.5,
                    "optimal_review_day": self._find_optimal_review_day(retention)
                })
            curves[topic_id] = points

        return curves

    def _find_optimal_review_day(self, retention: float) -> int:
        """Find optimal day for next review."""
        if retention < 0.3:
            return 0
        elif retention < 0.5:
            return 1
        elif retention < 0.7:
            return 3
        elif retention < 0.85:
            return 7
        else:
            return 30

    def get_topic_predictions(self, user_id: str, topic_id: str) -> Dict:
        """Get prediction for a specific topic."""
        predictions = self.get_all_predictions(user_id)
        micro = predictions.get("micro", [])
        topic = next((p for p in micro if str(p.get("topic_id")) == str(topic_id)), None)

        if not topic:
            return {"error": "Topic not found"}

        return {
            "topic_id": str(topic_id),
            "retention_probability": round(float(topic.get("retention_probability", 0.5)), 2),
            "next_question_difficulty": topic.get("next_question_difficulty", 3),
            "probability_correct_next": round(float(topic.get("probability_correct_next", 0.5)), 2),
            "stress_impact": round(float(topic.get("stress_impact", 0.3)), 2),
            "fatigue_level": round(float(topic.get("fatigue_level", 0.3)), 2)
        }

    def get_stress_fatigue_predictions(self, user_id: str, subject: Optional[str] = None) -> Dict:
        """Get stress and fatigue predictions."""
        try:
            data_manager = self._get_data_manager(user_id)
            interactions = data_manager.load_interactions()

            if not interactions.empty:
                stress_series = interactions.get("stress_level", pd.Series([0.3]))
                fatigue_series = interactions.get("fatigue_index", pd.Series([0.3]))

                current_stress = float(stress_series.tail(20).mean()) if len(stress_series) > 0 else 0.3
                current_fatigue = float(fatigue_series.tail(20).mean()) if len(fatigue_series) > 0 else 0.3

                stress_trend = "increasing" if len(stress_series) > 20 and stress_series.tail(10).mean() > stress_series.head(10).mean() else "stable"
                fatigue_trend = "increasing" if len(fatigue_series) > 20 and fatigue_series.tail(10).mean() > fatigue_series.head(10).mean() else "stable"
            else:
                current_stress = 0.3
                current_fatigue = 0.3
                stress_trend = "stable"
                fatigue_trend = "stable"

            return {
                "current_stress": round(current_stress, 2),
                "current_fatigue": round(current_fatigue, 2),
                "stress_trend": stress_trend,
                "fatigue_trend": fatigue_trend,
                "recommended_intensity": "low" if current_stress > 0.7 or current_fatigue > 0.7 else "moderate"
            }

        except Exception as e:
            logger.error(f"Error getting stress/fatigue: {e}")
            return {
                "current_stress": 0.3,
                "current_fatigue": 0.3,
                "stress_trend": "stable",
                "fatigue_trend": "stable",
                "recommended_intensity": "moderate"
            }

    def persist_interactions(self, user_id: str, session_id: str, subject: str, responses: List[Dict]):
        """Persist interactions to database."""
        try:
            data_manager = self._get_data_manager(user_id)
            data_manager.save_interactions(responses)
            logger.info(f"Persisted {len(responses)} interactions for user {user_id}")
        except Exception as e:
            logger.error(f"Error persisting interactions: {e}")

    def get_question_sequence(self, user_id: str, subject: Optional[str],
                              batch_type: str = "immediate", count: int = 10) -> List[Dict]:
        """Get question sequence for scheduling."""
        predictions = self.get_all_predictions(user_id, subject)
        micro = predictions.get("micro", [])

        filtered = [m for m in micro if m.get("batch_type") == batch_type]
        if not filtered:
            filtered = sorted(micro, key=lambda x: x.get("retention_probability", 0.5))

        sequence = []
        for row in filtered[:count]:
            sequence.append({
                "topic_id": row.get("topic_id", "unknown"),
                "question_id": f"{row.get('topic_id', 'unknown')}_q1",
                "priority": round(1 - row.get("retention_probability", 0.5), 2),
                "batch_type": row.get("batch_type", batch_type),
                "retention_probability": row.get("retention_probability", 0.5),
                "repeat_in_seconds": row.get("repeat_in_seconds", 300)
            })

        return sequence

    def get_retention_summary(self, user_id: str, subject: Optional[str] = None) -> Dict:
        """Get retention summary."""
        predictions = self.get_all_predictions(user_id, subject)
        micro = predictions.get("micro", [])

        if not micro:
            return {
                "overall_retention": 0.5,
                "median_retention": 0.5,
                "total_topics": 0
            }

        retentions = [float(m.get("retention_probability", 0.5)) for m in micro]

        return {
            "overall_retention": round(float(np.mean(retentions)), 2),
            "median_retention": round(float(np.median(retentions)), 2),
            "std_retention": round(float(np.std(retentions)), 2),
            "total_topics": len(micro)
        }

    def get_model_status(self, user_id: str, model_name: str) -> Dict:
        """Get model status."""
        data_manager = self._get_data_manager(user_id)
        model = data_manager.load_latest_model(f"{model_name}_lstm", user_id)
        return {
            "trained": model is not None,
            "last_trained": datetime.now().isoformat() if model else None
        }

    def prepare_for_nodejs(self, user_id: str, subject: Optional[str] = None) -> Dict:
        """Prepare predictions for Node.js integration."""
        predictions = self.get_all_predictions(user_id, subject)
        summary = self.get_retention_summary(user_id, subject)

        return {
            "success": True,
            "user_id": user_id,
            "subject": subject,
            "timestamp": datetime.now().isoformat(),
            "predictions": {
                "micro": predictions.get("micro", [])[:100],
                "meso": predictions.get("meso", []),
                "macro": predictions.get("macro", {}),
                "summary": summary,
                "forgetting_curves": predictions.get("forgetting_curves", {})
            },
            "models_ready": {
                "micro": bool(predictions.get("micro")),
                "meso": bool(predictions.get("meso")),
                "macro": bool(predictions.get("macro"))
            }
        }

    def generate_stress_fatigue_recommendations(self, stress_fatigue: Dict) -> List[Dict]:
        """Generate recommendations based on stress/fatigue."""
        recommendations = []
        stress = float(stress_fatigue.get("current_stress", 0.3))
        fatigue = float(stress_fatigue.get("current_fatigue", 0.3))

        if stress > 0.7:
            recommendations.append({
                "type": "stress",
                "severity": "high",
                "message": "Stress levels are high. Consider taking a break.",
                "action": "take_break",
                "duration": 15
            })
        elif stress > 0.5:
            recommendations.append({
                "type": "stress",
                "severity": "moderate",
                "message": "Moderate stress detected. Use breathing exercises.",
                "action": "relaxation_exercise",
                "duration": 5
            })

        if fatigue > 0.7:
            recommendations.append({
                "type": "fatigue",
                "severity": "high",
                "message": "High fatigue detected. End the session or take a long break.",
                "action": "end_session",
                "duration": 0
            })
        elif fatigue > 0.5:
            recommendations.append({
                "type": "fatigue",
                "severity": "moderate",
                "message": "Fatigue is rising. Take a short break.",
                "action": "short_break",
                "duration": 10
            })

        return recommendations

    def update_after_batch(self, user_id: str, subject: Optional[str],
                          batch_type: Optional[str], performance: Optional[Dict]):
        """Update after batch completion."""
        logger.info(f"Batch completed for user {user_id}: {batch_type}")
        return {"success": True}


    # retention_prediction_service.py - Fixed generate_forgetting_curves method

def generate_forgetting_curves(self, user_id: str, micro_predictions: List[Dict]) -> Dict:
    """Generate forgetting curves."""
    curves = {}
    # ==================== FIX: Use default time points if config missing ====================
    try:
        time_points = self.config.FORGETTING_CURVE.get("time_points", [1, 3, 7, 14, 30])
    except AttributeError:
        time_points = [1, 3, 7, 14, 30]
        logger.warning("FORGETTING_CURVE not found in config, using default time points")
    # ==================== END OF FIX ====================

    for pred in micro_predictions:
        topic_id = str(pred.get("topic_id", "unknown_topic"))
        current_retention = float(pred.get("current_retention", 0.5))
        tau = 30 * (1 + current_retention)

        points = []
        for day in time_points:
            retention = current_retention * np.exp(-day / tau)
            retention = float(min(1.0, max(0.0, retention)))
            points.append({
                "day": int(day),
                "retention": round(retention, 2),
                "needs_review": retention < 0.5,
                "optimal_review_day": self._find_optimal_review_day(retention)
            })
        curves[topic_id] = points

    return curves

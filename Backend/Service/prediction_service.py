"""Prediction Service - Handles model predictions with MongoDB storage."""

import logging
import traceback
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd

from Service.data_manager import StudentDataManager
from Model.practice_difficulty import PracticeDifficultyModel
from Model.exam_difficulty import ExamDifficultyModel
from Model.learning_velocity import LearningVelocityModel
from Model.burnout_risk import BurnoutRiskModel
from Model.adaptive_scheduling import AdaptiveSchedulingModel
from Utils.helpers import difficulty_level_from_value, bounded_smooth_difficulty

logger = logging.getLogger(__name__)


class PredictionService:
    """Enhanced service for making predictions with trained models."""

    def __init__(self, config):
        self.config = config
        self.models_cache = {}
        logger.info("PredictionService initialized")

    def _get_data_manager(self, student_id: str) -> StudentDataManager:
        """Get data manager for student."""
        return StudentDataManager(student_id)

    def _load_model(self, student_id: str, model_class, model_name: str,
                    sequence_length: int, n_features: int):
        """Load model from MongoDB with caching."""
        cache_key = f"{student_id}_{model_name}"

        # Check cache
        if cache_key in self.models_cache:
            return self.models_cache[cache_key]

        try:
            data_manager = self._get_data_manager(student_id)
            model = data_manager.load_latest_model(model_name, student_id)

            if model:
                self.models_cache[cache_key] = model
                logger.info(f"Loaded {model_name} model for student {student_id}")
                return model

            logger.info(f"No trained model found for {model_name}")
            return None

        except Exception as e:
            logger.error(f"Error loading {model_name} model: {e}")
            return None

    def clear_student_cache(self, student_id: str):
        """Clear in-memory cache for a student."""
        cache_keys = [key for key in self.models_cache.keys() if key.startswith(f"{student_id}_")]
        for key in cache_keys:
            self.models_cache.pop(key, None)
        logger.info(f"Cleared prediction cache for {student_id}")

    # ==================== PRACTICE DIFFICULTY PREDICTION ====================

    def predict_practice_difficulty(self, student_id: str, features: List[float]) -> Dict[str, Any]:
        """Predict next difficulty for practice mode."""
        logger.info(f"Predicting practice difficulty for student {student_id}")

        try:
            data_manager = self._get_data_manager(student_id)
            practice_df = data_manager.load_practice_features()

            model = self._load_model(
                student_id, PracticeDifficultyModel, 'practice_difficulty',
                self.config.SEQUENCE_LENGTH_PRACTICE or 10, self.config.PRACTICE_FEATURES_COUNT or 12
            )

            current_diff = 0.5
            if isinstance(features, list) and len(features) > 7:
                try:
                    current_diff = float(np.clip(float(features[7]), 0.2, 0.95))
                except Exception:
                    current_diff = 0.5

            if model:
                try:
                    feature_cols = self.config.PRACTICE_FEATURES

                    if practice_df.empty:
                        recent_data = pd.DataFrame(columns=feature_cols)
                    else:
                        recent_data = practice_df.reindex(columns=feature_cols, fill_value=0.5).tail(
                            self.config.SEQUENCE_LENGTH_PRACTICE or 10 - 1
                        )

                    current_features = np.array(features, dtype=np.float32)
                    if current_features.shape[0] != self.config.PRACTICE_FEATURES_COUNT:
                        if current_features.shape[0] < self.config.PRACTICE_FEATURES_COUNT:
                            pad = np.full(
                                (self.config.PRACTICE_FEATURES_COUNT - current_features.shape[0],),
                                0.5, dtype=np.float32
                            )
                            current_features = np.concatenate([current_features, pad])
                        else:
                            current_features = current_features[:self.config.PRACTICE_FEATURES_COUNT]

                    if len(recent_data) > 0:
                        recent_features = recent_data.values.astype(np.float32)
                        sequence = np.vstack([recent_features, current_features.reshape(1, -1)])
                    else:
                        sequence = current_features.reshape(1, -1)

                    if len(sequence) < self.config.SEQUENCE_LENGTH_PRACTICE:
                        padding = np.zeros((
                            self.config.SEQUENCE_LENGTH_PRACTICE - len(sequence),
                            self.config.PRACTICE_FEATURES_COUNT
                        ))
                        sequence = np.vstack([padding, sequence])
                    else:
                        sequence = sequence[-self.config.SEQUENCE_LENGTH_PRACTICE:]

                    sequence = np.clip(sequence, 0, 1)
                    prediction = model.predict_next(sequence)

                    raw_pred = float(prediction['predicted_difficulty'])
                    smoothed = bounded_smooth_difficulty(current_diff, raw_pred)

                    return {
                        'method': 'random_forest',
                        'predicted_difficulty': round(raw_pred, 2),
                        'smoothed_difficulty': round(smoothed, 2),
                        'confidence': round(float(prediction.get('confidence', 0.8)), 2),
                        'model_trained': True
                    }

                except Exception as e:
                    logger.error(f"Model prediction error: {e}\n{traceback.format_exc()}")

            # Fallback
            min_samples = getattr(self.config, 'MIN_PRACTICE_SAMPLES', 10)
            if len(practice_df) >= min_samples:
                method = 'model_unavailable'
            else:
                method = 'insufficient_training_data'

            return {
                'method': method,
                'predicted_difficulty': round(current_diff, 2),
                'smoothed_difficulty': round(bounded_smooth_difficulty(current_diff, current_diff), 2),
                'confidence': 0.5,
                'model_trained': False
            }

        except Exception as e:
            logger.error(f"Practice prediction error: {e}\n{traceback.format_exc()}")
            return {
                'method': 'fallback',
                'predicted_difficulty': 0.5,
                'smoothed_difficulty': 0.5,
                'confidence': 0.5,
                'model_trained': False
            }

    # ==================== EXAM DIFFICULTY PREDICTION ====================

    def predict_exam_difficulty(self, student_id: str, features: List[float]) -> Dict[str, Any]:
        """Predict recommended exam-level difficulty."""
        logger.info(f"Predicting exam difficulty for student {student_id}")

        try:
            data_manager = self._get_data_manager(student_id)
            exam_df = data_manager.load_exam_features()

            current_features = np.array(features if isinstance(features, list) else [], dtype=np.float32)
            expected_size = int(getattr(self.config, 'EXAM_FEATURES', 8))

            if current_features.shape[0] < expected_size:
                pad = np.full((expected_size - current_features.shape[0],), 0.5, dtype=np.float32)
                current_features = np.concatenate([current_features, pad])
            elif current_features.shape[0] > expected_size:
                current_features = current_features[:expected_size]

            current_features = np.clip(current_features, 0.0, 1.0)

            model = self._load_model(
                student_id, ExamDifficultyModel, 'exam_difficulty',
                self.config.SEQUENCE_LENGTH_EXAM or 40, self.config.EXAM_FEATURES or 12
            )

            if model:
                try:
                    feature_cols = [
                        'overall_accuracy_avg', 'avg_difficulty_handled',
                        'readiness_score', 'consistency_index',
                        'exam_performance_trend', 'concept_coverage_ratio',
                        'time_efficiency_score', 'stamina_index'
                    ]

                    if not exam_df.empty:
                        safe_df = exam_df.reindex(columns=feature_cols, fill_value=0.5)
                        safe_df = safe_df.apply(pd.to_numeric, errors='coerce').fillna(0.5)
                        recent_rows = safe_df.tail(max(self.config.SEQUENCE_LENGTH_EXAM - 1, 0)).values.astype(np.float32)
                    else:
                        recent_rows = np.empty((0, expected_size), dtype=np.float32)

                    recent_sequence = []
                    if len(recent_rows) > 0:
                        recent_sequence.extend(recent_rows.tolist())
                    recent_sequence.append(current_features.tolist())

                    readiness_hint = float(current_features[2]) if expected_size >= 3 else None
                    prediction = model.predict_exam_difficulty(recent_sequence, student_readiness=readiness_hint)

                    recommended = float(np.clip(prediction.get('recommended_difficulty', 0.5), 0.2, 0.95))

                    return {
                        'recommended_difficulty': round(recommended, 2),
                        'difficulty_level': prediction.get('difficulty_level',
                                                         difficulty_level_from_value(recommended)),
                        'confidence': round(float(prediction.get('confidence', 0.75)), 2),
                        'method': 'random_forest',
                        'model_trained': True
                    }

                except Exception as e:
                    logger.error(f"Exam prediction error: {e}\n{traceback.format_exc()}")

            # Fallback
            readiness = float(current_features[2]) if expected_size >= 3 else 0.5
            consistency = float(current_features[3]) if expected_size >= 4 else 0.5
            recommended = float(np.clip(readiness * 0.7 + consistency * 0.3, 0.2, 0.95))

            if len(exam_df) >= int(getattr(self.config, 'MIN_EXAM_SAMPLES', 5)):
                method = 'model_unavailable'
            else:
                method = 'insufficient_training_data'

            return {
                'recommended_difficulty': round(recommended, 2),
                'difficulty_level': difficulty_level_from_value(recommended),
                'confidence': 0.55,
                'method': method,
                'model_trained': False
            }

        except Exception as e:
            logger.error(f"Exam prediction error: {e}\n{traceback.format_exc()}")
            return {
                'recommended_difficulty': 0.5,
                'difficulty_level': 'medium-hard',
                'confidence': 0.5,
                'method': 'fallback',
                'model_trained': False
            }

    # ==================== LEARNING VELOCITY PREDICTION ====================

    def predict_learning_velocity(self, student_id: str, concept: str,
                                  features: List[List[float]]) -> Dict[str, Any]:
        """Predict learning velocity for a concept."""
        try:
            model = self._load_model(
                student_id, LearningVelocityModel, f'learning_velocity_{concept}',
                self.config.SEQUENCE_LENGTH_DAILY, self.config.LEARNING_VELOCITY_FEATURES
            )

            if model:
                features_array = np.array(features, dtype=np.float32)
                if len(features_array.shape) == 2:
                    features_array = features_array.reshape(1, -1, self.config.LEARNING_VELOCITY_FEATURES)

                prediction = model.predict(features_array)

                return {
                    'concept': concept,
                    'mastery_slope_next_7_days': float(prediction[0]) if len(prediction) > 0 else 0,
                    'confidence': 0.8,
                    'method': 'random_forest'
                }

            return {
                'concept': concept,
                'mastery_slope_next_7_days': 0,
                'confidence': 0.5,
                'method': 'fallback'
            }

        except Exception as e:
            logger.error(f"Learning velocity prediction error: {e}")
            return {'concept': concept, 'mastery_slope_next_7_days': 0, 'confidence': 0.5, 'method': 'error'}

    # ==================== BURNOUT RISK PREDICTION ====================

    def predict_burnout_risk(self, student_id: str, features: List[float]) -> Dict[str, Any]:
        """Predict burnout risk."""
        try:
            model = self._load_model(
                student_id, BurnoutRiskModel, 'burnout_risk',
                self.config.SEQUENCE_LENGTH_SESSION, self.config.BURNOUT_RISK_FEATURES
            )

            if model and len(features) >= self.config.BURNOUT_RISK_FEATURES:
                features_array = np.array(features[:self.config.BURNOUT_RISK_FEATURES], dtype=np.float32)
                features_array = features_array.reshape(1, 1, -1)

                prediction = model.predict(features_array)
                risk = float(prediction[0]) if len(prediction) > 0 else 0.3

                return {
                    'burnout_risk': round(risk, 2),
                    'risk_level': 'high' if risk > 0.7 else 'moderate' if risk > 0.4 else 'low',
                    'confidence': 0.75,
                    'method': 'random_forest'
                }

            return {
                'burnout_risk': 0.3,
                'risk_level': 'low',
                'confidence': 0.5,
                'method': 'fallback'
            }

        except Exception as e:
            logger.error(f"Burnout risk prediction error: {e}")
            return {'burnout_risk': 0.3, 'risk_level': 'low', 'confidence': 0.5, 'method': 'error'}

    # ==================== ADAPTIVE SCHEDULING PREDICTION ====================

    def predict_priority_scores(self, student_id: str,
                                concept_features: Dict[str, List[float]]) -> Dict[str, Any]:
        """Predict priority scores for concepts."""
        try:
            model = self._load_model(
                student_id, AdaptiveSchedulingModel, 'adaptive_scheduling',
                1, 13
            )

            priorities = []
            if model:
                for concept, features in concept_features.items():
                    if len(features) < 13:
                        features = features + [0.5] * (13 - len(features))
                    features_array = np.array(features[:13], dtype=np.float32).reshape(1, 1, -1)

                    prediction = model.predict(features_array)
                    priority = float(prediction[0]) if len(prediction) > 0 else 0.5

                    priorities.append({
                        'concept': concept,
                        'priority_score': round(priority, 2)
                    })

                priorities = sorted(priorities, key=lambda x: x['priority_score'], reverse=True)

                return {
                    'priorities': priorities,
                    'study_plan': [p['concept'] for p in priorities[:5]],
                    'method': 'random_forest'
                }

            # Fallback
            for concept, features in concept_features.items():
                priority = (1 - features[0]) * 0.5 + features[1] * 0.3 + min(features[4] / 30, 1) * 0.2
                priorities.append({
                    'concept': concept,
                    'priority_score': round(priority, 2)
                })

            priorities = sorted(priorities, key=lambda x: x['priority_score'], reverse=True)

            return {
                'priorities': priorities,
                'study_plan': [p['concept'] for p in priorities[:5]],
                'method': 'rule_based'
            }

        except Exception as e:
            logger.error(f"Priority prediction error: {e}")
            return {'priorities': [], 'study_plan': [], 'method': 'error'}

    # Service/prediction_service.py - Add these methods if missing

    # ==================== ADD THESE METHODS IF MISSING ====================

    def _load_model(self, student_id: str, model_class, model_name: str,
                    sequence_length: int, n_features: int):
        """Load model from MongoDB with caching."""
        cache_key = f"{student_id}_{model_name}"

        # Check cache
        if cache_key in self.models_cache:
            return self.models_cache[cache_key]

        try:
            data_manager = self._get_data_manager(student_id)
            model = data_manager.load_latest_model(model_name, student_id)

            if model:
                self.models_cache[cache_key] = model
                logger.info(f"Loaded {model_name} model for student {student_id}")
                return model

            logger.info(f"No trained model found for {model_name}")
            return None

        except Exception as e:
            logger.error(f"Error loading {model_name} model: {e}")
            return None

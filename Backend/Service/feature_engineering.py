"""Feature Engineering Service - Computes features from raw attempt data."""

import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class FeatureEngineeringService:
    """Advanced feature engineering service."""

    @staticmethod
    def compute_practice_features(attempts_data: List[Dict]) -> pd.DataFrame:
        """Compute practice features from raw attempts."""
        if not attempts_data or len(attempts_data) < 1:
            return pd.DataFrame()

        try:
            df = pd.DataFrame(attempts_data)

            # ==================== FIX: Handle missing columns with defaults ====================
            # Ensure required columns exist with defaults
            if 'timestamp' not in df.columns:
                df['timestamp'] = datetime.now()
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce').fillna(datetime.now())

            # Handle 'correct' field - could be 'correct' or 'isCorrect'
            if 'correct' not in df.columns:
                if 'isCorrect' in df.columns:
                    df['correct'] = df['isCorrect']
                else:
                    df['correct'] = False

            # Handle 'time_spent' field
            if 'time_spent' in df.columns:
                # If time_spent is in milliseconds (> 600 means > 10 minutes), convert
                if df['time_spent'].max() > 600:
                    df['time_spent'] = df['time_spent'] / 1000.0
            elif 'response_time_ms' in df.columns:
                df['time_spent'] = df['response_time_ms'] / 1000.0
            else:
                df['time_spent'] = 10.0

        # Cap time_spent at reasonable values (1s to 600s = 10 minutes)
            df['time_spent'] = df['time_spent'].clip(1.0, 600.0)

            # Handle 'difficulty' field
            if 'difficulty' not in df.columns:
                if 'question_difficulty' in df.columns:
                    df['difficulty'] = df['question_difficulty']
                else:
                    df['difficulty'] = 0.5

            # Handle 'confidence' field
            if 'confidence' not in df.columns:
                if 'confidence_rating' in df.columns:
                    df['confidence'] = df['confidence_rating'] / 5.0
                else:
                    df['confidence'] = 0.5

            # Handle 'answer_changed' field
            if 'answer_changed' not in df.columns:
                if 'answer_changes' in df.columns:
                    df['answer_changed'] = df['answer_changes'] > 0
                elif 'hesitation_count' in df.columns:
                    df['answer_changed'] = df['hesitation_count'] > 0
                else:
                    df['answer_changed'] = False

            # Handle 'concept' field
            if 'concept' not in df.columns:
                if 'concept_area' in df.columns:
                    df['concept'] = df['concept_area']
                elif 'topic_id' in df.columns:
                    df['concept'] = df['topic_id']
                else:
                    df['concept'] = 'general'
            # ==================== END OF FIX ====================

            # Convert to numeric types
            df['accuracy'] = df['correct'].astype(float)
            df['time_spent'] = pd.to_numeric(df['time_spent'], errors='coerce').fillna(10)
            df['difficulty'] = pd.to_numeric(df['difficulty'], errors='coerce').fillna(0.5)
            df['confidence'] = pd.to_numeric(df['confidence'], errors='coerce').fillna(0.5)

            # Feature 1: accuracy (already done)

            # Feature 2: normalized_response_time
            # Use rolling mean of last 5 attempts or fallback
            if len(df) >= 2:
                rolling_mean = df['time_spent'].rolling(5, min_periods=1).mean().fillna(df['time_spent'])
                df['normalized_response_time'] = df['time_spent'] / rolling_mean
            else:
                df['normalized_response_time'] = df['time_spent'] / max(df['time_spent'].mean(), 1.0)
            df['normalized_response_time'] = df['normalized_response_time'].clip(0, 2)

            # Feature 3: rolling_time_variance
            df['rolling_time_variance'] = df['time_spent'].rolling(5, min_periods=1).var().fillna(0).clip(0, 100)

            # Feature 4: answer_change_count
            if 'answer_changed' in df.columns:
                df['answer_change_count'] = df['answer_changed'].rolling(5, min_periods=1).sum().fillna(0)
            else:
                df['answer_change_count'] = 0
            df['answer_change_count'] = df['answer_change_count'].clip(0, 5)

            # Feature 5: stress_score
            df['stress_score'] = (1 - df['accuracy']) * 0.6 + df['normalized_response_time'] * 0.4
            df['stress_score'] = df['stress_score'].clip(0, 1)

            # Feature 6: confidence_index
            df['confidence_index'] = df['confidence'].clip(0, 1)

            # Feature 7: concept_mastery_score
            df['concept_mastery_score'] = df['accuracy'].expanding().mean().fillna(0.5).clip(0, 1)

            # Feature 8: current_question_difficulty
            df['current_question_difficulty'] = df['difficulty'].clip(0, 1)

            # Feature 9: consecutive_correct_streak
            df['consecutive_correct_streak'] = df['accuracy'].rolling(10, min_periods=1).sum().fillna(0) / 10
            df['consecutive_correct_streak'] = df['consecutive_correct_streak'].clip(0, 1)

            # Feature 10: fatigue_indicator
            df['fatigue_indicator'] = np.clip(np.arange(len(df)) / 20, 0, 1)

            # Feature 11: focus_loss_frequency
            df['focus_loss_frequency'] = ((df['answer_change_count'] > 0) & (df['confidence'] < 0.6)).astype(float)
            df['focus_loss_frequency'] = df['focus_loss_frequency'].rolling(5, min_periods=1).mean().fillna(0).clip(0, 1)

            # Feature 12: preferred_difficulty_offset
            df['preferred_difficulty_offset'] = (df['current_question_difficulty'] - df['concept_mastery_score'] + 1) / 2
            df['preferred_difficulty_offset'] = df['preferred_difficulty_offset'].clip(0, 1)

            # Target: next_difficulty (shift difficulty for next question)
            df['next_difficulty'] = df['current_question_difficulty'].shift(-1).fillna(df['current_question_difficulty'])
            df['next_difficulty'] = df['next_difficulty'].clip(0, 1)

            # Select features
            feature_cols = [
                'accuracy', 'normalized_response_time', 'rolling_time_variance',
                'answer_change_count', 'stress_score', 'confidence_index',
                'concept_mastery_score', 'current_question_difficulty',
                'consecutive_correct_streak', 'fatigue_indicator',
                'focus_loss_frequency', 'preferred_difficulty_offset', 'next_difficulty'
            ]

            result = df[feature_cols].copy()

            # Clean and round
            for col in result.columns:
                if col != 'next_difficulty':
                    result[col] = pd.to_numeric(result[col], errors='coerce').fillna(0.5)
                    result[col] = result[col].clip(0, 1).round(2)

            result['next_difficulty'] = pd.to_numeric(result['next_difficulty'], errors='coerce').fillna(0.5)
            result['next_difficulty'] = result['next_difficulty'].clip(0, 1).round(2)

            logger.info(f"✅ Computed {len(result)} practice feature rows")
            return result

        except Exception as e:
            logger.error(f"Error computing practice features: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return pd.DataFrame()

    @staticmethod
    def compute_global_features(practice_df: pd.DataFrame) -> pd.DataFrame:
        """Compute global features from practice data."""
        if practice_df.empty or len(practice_df) < 10:
            return pd.DataFrame()

        try:
            df = practice_df.copy()
            df['session_id'] = df.get('session_id', 'default_session')

            global_rows = []

            for session_id in df['session_id'].unique():
                session_data = df[df['session_id'] == session_id]

                if len(session_data) < 3:
                    continue

                row = {
                    'session_id': session_id,
                    'session_accuracy_avg': session_data['accuracy'].mean(),
                    'avg_solved_difficulty': session_data['current_question_difficulty'].mean(),
                    'max_difficulty_sustained': session_data['current_question_difficulty'].max(),
                    'performance_trend_slope': 0,
                    'retention_score': session_data['accuracy'].mean(),
                    'burnout_risk_index': session_data['fatigue_indicator'].mean(),
                    'stress_trend_slope': 0,
                    'concept_coverage_ratio': len(session_data['concept'].unique()) / len(session_data) if 'concept' in session_data else 0.5,
                    'high_difficulty_accuracy': session_data[session_data['current_question_difficulty'] > 0.7]['accuracy'].mean() if len(session_data[session_data['current_question_difficulty'] > 0.7]) > 0 else 0.5,
                    'consistency_index': 1 - session_data['accuracy'].std(),
                    'avg_response_time_trend': 0,
                    'serious_test_performance_score': session_data['accuracy'].mean() * 0.5 + 0.5
                }

                global_rows.append(row)

            result = pd.DataFrame(global_rows)

            for col in result.columns:
                if col != 'session_id':
                    result[col] = pd.to_numeric(result[col], errors='coerce').fillna(0.5)
                    result[col] = result[col].clip(0, 1).round(2)

            result['readiness_difficulty_score'] = (
                result['session_accuracy_avg'] * 0.35 +
                result['high_difficulty_accuracy'] * 0.25 +
                result['consistency_index'] * 0.25 +
                result['serious_test_performance_score'] * 0.15
            ).clip(0, 1).round(2)

            return result

        except Exception as e:
            logger.error(f"Error computing global features: {e}")
            return pd.DataFrame()

    @staticmethod
    def compute_performance_metrics(practice_df: pd.DataFrame, exam_df: pd.DataFrame) -> Dict[str, Any]:
        """Compute performance metrics for dashboard."""
        metrics = {
            'practice': {
                'total_questions': len(practice_df),
                'overall_accuracy': practice_df['accuracy'].mean() if 'accuracy' in practice_df else 0.5,
                'recent_accuracy': practice_df['accuracy'].tail(20).mean() if 'accuracy' in practice_df else 0.5,
                'avg_difficulty': practice_df['current_question_difficulty'].mean() if 'current_question_difficulty' in practice_df else 0.5,
                'fatigue_level': practice_df['fatigue_indicator'].mean() if 'fatigue_indicator' in practice_df else 0.5
            },
            'exam': {
                'total_exams': len(exam_df),
                'avg_score': exam_df.get('score', 0.5)
            },
            'overall': {
                'readiness_score': 0.5,
                'burnout_risk': 0.3
            },
            'concepts': []
        }

        if 'concept' in practice_df.columns:
            concept_groups = practice_df.groupby('concept')
            for concept, group in concept_groups:
                metrics['concepts'].append({
                    'concept': concept,
                    'attempts': len(group),
                    'accuracy': group['accuracy'].mean(),
                    'avg_difficulty': group['current_question_difficulty'].mean() if 'current_question_difficulty' in group else 0.5
                })

        return metrics

    @staticmethod
    def compute_exam_features(exam_records: List[Dict]) -> pd.DataFrame:
        """Compute exam features from exam records."""
        if not exam_records:
            return pd.DataFrame()

        df = pd.DataFrame(exam_records)

        features = {
            'overall_accuracy_avg': df.get('accuracy', 0.5),
            'avg_difficulty_handled': 0.5,
            'readiness_score': df.get('score', 0.5) / 100 if 'score' in df else 0.5,
            'consistency_index': 0.5
        }

        return pd.DataFrame([features])
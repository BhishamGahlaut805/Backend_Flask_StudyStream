"""Schedule Service - Generates daily and adaptive study schedules."""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

from Service.data_manager import StudentDataManager
from Utils.helpers import timer_frame_from_retention, timer_frame_label

logger = logging.getLogger(__name__)


class ScheduleService:
    """Creates actionable schedules based on retention predictions."""

    def __init__(self, config):
        self.config = config
        logger.info("ScheduleService initialized")

    def _get_data_manager(self, user_id: str) -> StudentDataManager:
        """Get data manager for user."""
        return StudentDataManager(user_id)

    def generate_daily_schedule(self, user_id: str, subject: Optional[str] = None,
                                predictions: Optional[Any] = None) -> Dict:
        """Generate a daily learning schedule."""
        try:
            data_manager = self._get_data_manager(user_id)

            if predictions is None:
                predictions = data_manager.load_predictions('micro') or []
            elif isinstance(predictions, dict):
                predictions = predictions.get('micro', [])

            categorized = {
                'immediate': [],
                'short_term': [],
                'medium_term': [],
                'long_term': [],
                'mastered': []
            }

            for pred in predictions:
                retention = float(pred.get('current_retention', pred.get('retention_probability', 0.5)))
                placed = False

                for category, cfg in self.config.REPETITION_SCHEDULES.items():
                    low, high = cfg.get('retention_range', (0.0, 1.0))
                    if low <= retention < high:
                        repeat_seconds = int(pred.get('repeat_in_seconds',
                                                     timer_frame_from_retention(retention)))
                        categorized[category].append({
                            'topic_id': str(pred.get('topic_id', 'unknown')),
                            'subject': pred.get('subject', subject),
                            'retention': retention,
                            'questions_needed': int(cfg.get('questions_per_topic', 3)),
                            'batch_size': int(cfg.get('batch_size', 5)),
                            'repeat_in_seconds': repeat_seconds,
                            'timer_frame_label': timer_frame_label(repeat_seconds),
                            'next_repeat_at': (datetime.now() + timedelta(seconds=repeat_seconds)).isoformat()
                        })
                        placed = True
                        break

                if not placed:
                    categorized['medium_term'].append({
                        'topic_id': str(pred.get('topic_id', 'unknown')),
                        'subject': pred.get('subject', subject),
                        'retention': retention,
                        'questions_needed': 3,
                        'batch_size': 5,
                        'repeat_in_seconds': 300,
                        'timer_frame_label': '5_minutes',
                        'next_repeat_at': (datetime.now() + timedelta(seconds=300)).isoformat()
                    })

            immediate_questions = []
            for row in sorted(categorized['immediate'], key=lambda x: x['retention'])[:3]:
                for i in range(min(3, row['questions_needed'])):
                    immediate_questions.append({
                        'topic_id': row['topic_id'],
                        'subject': row.get('subject'),
                        'question_number': i + 1,
                        'priority': round(1 - row['retention'], 4),
                        'repeat_in_seconds': int(row.get('repeat_in_seconds', 300)),
                        'timer_frame_label': row.get('timer_frame_label', '5_minutes'),
                        'next_repeat_at': row.get('next_repeat_at')
                    })

            session_batches = []
            cur_batch = []
            for row in sorted(categorized['short_term'], key=lambda x: x['retention']):
                for i in range(min(4, row['questions_needed'])):
                    cur_batch.append({
                        'topic_id': row['topic_id'],
                        'subject': row.get('subject'),
                        'question_number': i + 1,
                        'priority': round(1 - row['retention'], 4),
                        'repeat_in_seconds': int(row.get('repeat_in_seconds', 300)),
                        'timer_frame_label': row.get('timer_frame_label', '5_minutes'),
                        'next_repeat_at': row.get('next_repeat_at')
                    })
                    if len(cur_batch) >= 5:
                        session_batches.append(cur_batch)
                        cur_batch = []
            if cur_batch:
                session_batches.append(cur_batch)

            chapter_reviews = [
                {
                    'topic_id': row['topic_id'],
                    'subject': row.get('subject'),
                    'timing': 'next_day',
                    'questions': row['questions_needed'],
                    'repeat_in_seconds': int(row.get('repeat_in_seconds', 300)),
                    'timer_frame_label': row.get('timer_frame_label', '5_minutes'),
                    'next_repeat_at': row.get('next_repeat_at')
                }
                for row in categorized['medium_term'][:10]
            ]

            schedule = {
                'user_id': user_id,
                'subject': subject,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'generated_at': datetime.now().isoformat(),
                'immediate_batch': {
                    'questions': immediate_questions,
                    'total_questions': len(immediate_questions),
                    'focus_topics': [q['topic_id'] for q in immediate_questions[:3]]
                },
                'session_batch': {
                    'batches': session_batches,
                    'batch_count': len(session_batches),
                    'total_questions': sum(len(b) for b in session_batches)
                },
                'chapter_reviews': chapter_reviews,
                'long_term_plan': {
                    'review_topics': [x['topic_id'] for x in categorized['long_term'][:15]],
                    'mastered_topics': [x['topic_id'] for x in categorized['mastered'][:15]]
                },
                'summary': {
                    'total_topics': sum(len(v) for v in categorized.values()),
                    'estimated_total_questions': len(immediate_questions) +
                        sum(len(b) for b in session_batches) +
                        sum(c['questions'] for c in chapter_reviews)
                }
            }

            data_manager.save_schedule(schedule)
            return schedule

        except Exception as e:
            logger.error(f"Error generating schedule: {e}")
            return {
                'user_id': user_id,
                'subject': subject,
                'date': datetime.now().strftime('%Y-%m-%d'),
                'generated_at': datetime.now().isoformat(),
                'immediate_batch': {'questions': [], 'total_questions': 0, 'focus_topics': []},
                'session_batch': {'batches': [], 'batch_count': 0, 'total_questions': 0},
                'chapter_reviews': [],
                'long_term_plan': {'review_topics': [], 'mastered_topics': []},
                'summary': {'total_topics': 0, 'estimated_total_questions': 0}
            }

    def get_next_questions(self, user_id: str, subject: Optional[str] = None,
                          current_stress: float = 0.3, current_fatigue: float = 0.3) -> Dict:
        """Get next set of questions for immediate learning."""
        try:
            schedule = self.generate_daily_schedule(user_id, subject)

            max_q = 2 if current_stress > 0.7 or current_fatigue > 0.7 else 3
            immediate = schedule.get('immediate_batch', {}).get('questions', [])
            session_batches = schedule.get('session_batch', {}).get('batches', [])

            if immediate:
                picked = immediate[:max_q]
            elif session_batches:
                picked = session_batches[0][:max_q]
            else:
                picked = []

            return {
                'questions': picked,
                'recommended_break': bool(current_stress > 0.7 or current_fatigue > 0.7),
                'remaining_in_batch': max(0, len(immediate) - len(picked))
            }

        except Exception as e:
            logger.error(f"Error getting next questions: {e}")
            return {'questions': [], 'recommended_break': False, 'remaining_in_batch': 0}

    def get_subject_repetition_schedule(self, user_id: str, subject: str) -> Dict:
        """Get subject-level repetition schedule."""
        schedule = self.generate_daily_schedule(user_id, subject)
        return {
            'user_id': user_id,
            'subject': subject,
            'topics_for_repetition': [q['topic_id'] for q in schedule.get('immediate_batch', {}).get('questions', [])[:10]],
            'schedule': schedule
        }

    def get_topic_repetition_schedule(self, user_id: str, topic_id: str) -> Dict:
        """Get topic-level repetition schedule."""
        schedule = self.generate_daily_schedule(user_id)

        occurrences = []
        for q in schedule.get('immediate_batch', {}).get('questions', []):
            if str(q.get('topic_id')) == str(topic_id):
                occurrences.append({'phase': 'immediate', 'question': q})

        for batch in schedule.get('session_batch', {}).get('batches', []):
            for q in batch:
                if str(q.get('topic_id')) == str(topic_id):
                    occurrences.append({'phase': 'session', 'question': q})

        return {
            'user_id': user_id,
            'topic_id': topic_id,
            'occurrences': occurrences,
            'repeat_count': len(occurrences)
        }

    def get_optimal_study_times(self, user_id: str, subject: Optional[str] = None,
                               stress_fatigue: Optional[Dict] = None) -> Dict:
        """Get optimal study times based on stress and fatigue."""
        stress_fatigue = stress_fatigue or {}
        stress = float(stress_fatigue.get('current_stress', 0.3))
        fatigue = float(stress_fatigue.get('current_fatigue', 0.3))

        if stress > 0.7 or fatigue > 0.7:
            windows = ['09:00-10:00', '18:00-19:00']
            intensity = 'light'
        elif stress > 0.5 or fatigue > 0.5:
            windows = ['08:30-10:00', '17:30-19:00']
            intensity = 'moderate'
        else:
            windows = ['07:30-10:00', '16:30-19:30']
            intensity = 'high'

        return {
            'user_id': user_id,
            'subject': subject,
            'optimal_windows': windows,
            'recommended_session_minutes': 35 if intensity == 'high' else 25,
            'intensity': intensity,
            'generated_at': datetime.now().isoformat()
        }
        
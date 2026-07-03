# dashboard.py - Update datetime handling

from flask import Blueprint, request, jsonify, current_app
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from collections import defaultdict
import json

from Utils.validators import validate_student_id
from Utils.helpers import calculate_streak, get_datetime_series, get_last_datetime_iso
from Utils.decorators import handle_errors, log_request

dashboard_bp = Blueprint('dashboard', __name__)
logger = logging.getLogger(__name__)


# ==================== HELPER: Safe datetime parsing ====================
def safe_parse_datetime(value):
    """Safely parse datetime string to timezone-aware datetime."""
    if not value:
        return None
    try:
        # Parse ISO format
        dt = datetime.fromisoformat(str(value).replace('Z', '+00:00'))
        # Make it timezone-aware if it's naive
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except Exception:
        return None


def safe_now():
    """Get timezone-aware current time."""
    return datetime.now(timezone.utc)


def safe_total_seconds(td):
    """Safely get total seconds from timedelta."""
    if td is None:
        return 0
    return td.total_seconds()


@dashboard_bp.route('/performance/<student_id>', methods=['GET'])
@handle_errors
@log_request
def get_dashboard_performance(student_id):
    """
    Complete in-depth performance analysis for dashboard.
    Fetches data from all MongoDB collections and provides comprehensive analytics.
    """
    validate_student_id(student_id)

    data_manager = current_app.prediction_service._get_data_manager(student_id)

    # ==================== LOAD ALL DATA FROM MONGODB ====================
    practice_df = data_manager.load_practice_features()
    exam_df = data_manager.load_exam_features()
    concept_features = data_manager.load_concept_features()
    interactions_df = data_manager.load_interactions()
    daily_aggregates_df = data_manager.load_daily_aggregates()
    sessions_df = pd.DataFrame(data_manager.load_sessions())
    predictions = data_manager.load_predictions('micro') or []
    meso_predictions = data_manager.load_predictions('meso') or []
    macro_predictions = data_manager.load_predictions('macro') or {}

    # Load sequences for trend analysis
    micro_sequences = data_manager.load_micro_sequences()
    meso_sequences = data_manager.load_meso_sequences()
    macro_sequences = data_manager.load_macro_sequences()

    logger.info(f"[Dashboard] Loaded data for {student_id}: "
                f"practice={len(practice_df)}, exams={len(exam_df)}, "
                f"concepts={len(concept_features)}, interactions={len(interactions_df)}, "
                f"daily={len(daily_aggregates_df)}, sessions={len(sessions_df)}")

    # ==================== COMPUTE BASE METRICS ====================
    from Service.feature_engineering import FeatureEngineeringService
    feature_service = FeatureEngineeringService()
    base_metrics = feature_service.compute_performance_metrics(practice_df, exam_df)

    # ==================== IN-DEPTH ANALYSIS ====================
    topic_analysis = _analyze_topics(practice_df, concept_features, interactions_df)
    concept_mastery = _calculate_concept_mastery(practice_df, concept_features)
    stability_index = _calculate_stability_index(practice_df, concept_features)
    confidence_calibration = _calculate_confidence_calibration(interactions_df, practice_df)
    error_patterns = _analyze_error_patterns(interactions_df, practice_df)
    weakness_priority = _calculate_weakness_priority(concept_features, practice_df)
    forgetting_curve = _calculate_forgetting_curve(concept_features, daily_aggregates_df)
    fatigue_index = _calculate_fatigue_index(interactions_df, practice_df, sessions_df)
    behavior_profile = _analyze_behavior_profile(interactions_df, practice_df)
    difficulty_tolerance = _calculate_difficulty_tolerance(practice_df)
    study_efficiency = _calculate_study_efficiency(practice_df, daily_aggregates_df)
    focus_loss = _analyze_focus_loss(interactions_df, practice_df)
    time_allocation = _generate_time_allocation(weakness_priority, concept_features)
    stress_patterns = _analyze_stress_patterns(interactions_df, practice_df)
    burnout_risk = _calculate_burnout_risk(interactions_df, practice_df, fatigue_index)
    learning_velocity = _calculate_learning_velocity(concept_features, daily_aggregates_df)
    trend_data = _calculate_trends(practice_df, daily_aggregates_df)
    session_analysis = _analyze_sessions(sessions_df, practice_df)
    predictions_summary = _summarize_predictions(predictions, meso_predictions, macro_predictions)
    recommendations = _generate_recommendations(
        weakness_priority, burnout_risk, fatigue_index,
        concept_mastery, study_efficiency
    )

    # ==================== BUILD COMPLETE RESPONSE ====================
    dashboard_data = {
        'student_id': student_id,
        'last_updated': datetime.now(timezone.utc).isoformat(),

        # Summary Metrics
        'summary': {
            'total_practice_questions': int(base_metrics['practice'].get('total_questions', 0)),
            'overall_accuracy': float(base_metrics['practice'].get('overall_accuracy', 0)),
            'avg_difficulty': float(base_metrics['practice'].get('avg_difficulty', 0.5)),
            'total_exams': int(base_metrics.get('exam', {}).get('total_exams', 0)),
            'exam_avg_score': float(base_metrics.get('exam', {}).get('avg_score', 0)),
            'readiness_score': float(base_metrics['overall'].get('readiness_score', 0.5)),
            'burnout_risk': burnout_risk['current_risk'],
            'total_concepts': len(concept_features),
            'mastered_concepts': sum(1 for c in concept_mastery.values() if c >= 0.8),
            'struggling_concepts': sum(1 for c in concept_mastery.values() if c < 0.5),
            'study_streak': calculate_streak(practice_df),
            'total_study_time_minutes': _calculate_total_study_time(practice_df),
        },

        # In-Depth Analysis Sections
        'topic_analysis': topic_analysis,
        'concept_mastery': concept_mastery,
        'stability_index': stability_index,
        'confidence_calibration': confidence_calibration,
        'error_patterns': error_patterns,
        'weakness_priority': weakness_priority,
        'forgetting_curve': forgetting_curve,
        'fatigue_index': fatigue_index,
        'behavior_profile': behavior_profile,
        'difficulty_tolerance': difficulty_tolerance,
        'study_efficiency': study_efficiency,
        'focus_loss': focus_loss,
        'time_allocation': time_allocation,
        'stress_patterns': stress_patterns,
        'burnout_risk': burnout_risk,
        'learning_velocity': learning_velocity,
        'trend_data': trend_data,
        'session_analysis': session_analysis,
        'predictions_summary': predictions_summary,
        'recommendations': recommendations,

        # Chart Data
        'charts': _generate_chart_data(practice_df, daily_aggregates_df, trend_data),

        # Recent Activity
        'recent_activity': {
            'last_practice': get_last_datetime_iso(practice_df),
            'last_exam': get_last_datetime_iso(exam_df),
            'questions_today': _get_questions_today(practice_df),
            'streak_days': calculate_streak(practice_df),
            'sessions_this_week': _get_sessions_this_week(sessions_df),
            'avg_daily_accuracy': float(daily_aggregates_df['accuracy'].mean()) if not daily_aggregates_df.empty else 0
        }
    }

    logger.info(f"[Dashboard] Complete analysis generated for {student_id}")
    return jsonify({'success': True, 'dashboard_data': dashboard_data})


# ==================== HELPER FUNCTIONS ====================

def _get_sessions_this_week(sessions_df):
    """Get number of sessions in the last 7 days with safe datetime handling."""
    if sessions_df.empty or 'started_at' not in sessions_df.columns:
        return 0

    now = safe_now()
    week_ago = now - timedelta(days=7)

    count = 0
    for _, row in sessions_df.iterrows():
        dt = safe_parse_datetime(row.get('started_at'))
        if dt and dt >= week_ago:
            count += 1
    return count
def _calculate_total_study_time(practice_df):
    """Calculate total study time in minutes safely."""
    if practice_df.empty:
        return 0

    if 'time_spent' in practice_df.columns:
        return int(practice_df['time_spent'].sum() / 60)
    elif 'normalized_response_time' in practice_df.columns:
        # Approximate: normalized_time * 10 seconds per question
        return int(practice_df['normalized_response_time'].sum() * 10 / 60)
    else:
        # Default: 10 seconds per question
        return int(len(practice_df) * 10 / 60)

def _calculate_fatigue_index(interactions_df, practice_df, sessions_df):
    """Calculate fatigue index from interactions and sessions with safe datetime handling."""
    fatigue = {
        'current': 0.3,
        'trend': 'stable',
        'by_session': [],
        'risk_level': 'low',
        'recommendation': 'Continue studying with regular breaks'
    }

    df = interactions_df if not interactions_df.empty else practice_df

    if df.empty:
        return fatigue

    # Calculate fatigue from session data
    if not sessions_df.empty and 'started_at' in sessions_df.columns:
        # Sort sessions by time
        sessions_df = sessions_df.sort_values('started_at')
        now = safe_now()

        # Calculate fatigue per session
        for _, session in sessions_df.iterrows():
            dt = safe_parse_datetime(session.get('started_at'))
            if dt:
                session_hours = safe_total_seconds(now - dt) / 3600
                if session_hours < 24:  # Only recent sessions
                    fatigue_value = min(1.0, 0.3 + (session_hours / 48))
                    fatigue['by_session'].append(fatigue_value)

    # Calculate from practice data
    if not practice_df.empty and 'fatigue_indicator' in practice_df.columns:
        recent_fatigue = practice_df['fatigue_indicator'].tail(20).mean()
        fatigue['current'] = float(recent_fatigue)

        # Determine trend
        if len(practice_df) >= 10:
            first_half = practice_df['fatigue_indicator'].head(len(practice_df)//2).mean()
            second_half = practice_df['fatigue_indicator'].tail(len(practice_df)//2).mean()
            fatigue['trend'] = 'increasing' if second_half > first_half + 0.05 else 'decreasing' if second_half < first_half - 0.05 else 'stable'

    # Determine risk level
    if fatigue['current'] > 0.7:
        fatigue['risk_level'] = 'high'
        fatigue['recommendation'] = 'High fatigue detected - take a long break and rest'
    elif fatigue['current'] > 0.5:
        fatigue['risk_level'] = 'moderate'
        fatigue['recommendation'] = 'Moderate fatigue - take short breaks between sessions'
    else:
        fatigue['risk_level'] = 'low'
        fatigue['recommendation'] = 'Feeling fresh - continue effective studying'

    return fatigue


def _analyze_sessions(sessions_df, practice_df):
    """Analyze session data with safe datetime handling."""
    analysis = {
        'total_sessions': 0,
        'average_duration_minutes': 0,
        'average_questions_per_session': 0,
        'average_accuracy_per_session': 0,
        'sessions_by_day': [],
        'most_active_days': []
    }

    if sessions_df.empty:
        return analysis

    analysis['total_sessions'] = len(sessions_df)

    if not practice_df.empty and 'session_id' in practice_df.columns:
        sessions = practice_df.groupby('session_id')
        analysis['average_questions_per_session'] = float(sessions.size().mean())
        analysis['average_accuracy_per_session'] = float(sessions['accuracy'].mean().mean())

    # Sessions by day of week
    day_counts = defaultdict(int)
    if 'started_at' in sessions_df.columns:
        for _, session in sessions_df.iterrows():
            dt = safe_parse_datetime(session.get('started_at'))
            if dt:
                day_counts[dt.strftime('%A')] += 1

    analysis['sessions_by_day'] = [{'day': day, 'count': count} for day, count in day_counts.items()]
    analysis['most_active_days'] = sorted(day_counts.items(), key=lambda x: x[1], reverse=True)[:3]

    return analysis


def _get_questions_today(practice_df):
    """Get number of questions practiced today with safe datetime handling."""
    if practice_df.empty or 'timestamp' not in practice_df.columns:
        return 0

    today = safe_now().date()
    count = 0

    for _, row in practice_df.iterrows():
        dt = safe_parse_datetime(row.get('timestamp'))
        if dt and dt.date() == today:
            count += 1

    return count


def _get_last_datetime_iso_safe(df):
    """Safely get last datetime as ISO string."""
    if df is None or df.empty:
        return None

    for col in ['timestamp', 'submitted_at', 'created_at', 'date']:
        if col in df.columns:
            try:
                dt = safe_parse_datetime(df[col].iloc[-1])
                if dt:
                    return dt.isoformat()
            except Exception:
                pass
    return None


# ==================== EXISTING HELPER FUNCTIONS (with safe datetime fixes) ====================

def _analyze_topics(practice_df, concept_features, interactions_df):
    """Analyze topic performance with detailed metrics."""
    if practice_df.empty:
        return {'strong_topics': [], 'weak_topics': [], 'topic_stats': []}

    topic_stats = []
    if 'concept' in practice_df.columns:
        for concept, group in practice_df.groupby('concept'):
            accuracy = group['accuracy'].mean()
            attempts = len(group)
            avg_difficulty = group['current_question_difficulty'].mean() if 'current_question_difficulty' in group else 0.5

            confidence = 0.5
            stress = 0.3
            if not interactions_df.empty and 'topic_id' in interactions_df.columns:
                topic_interactions = interactions_df[interactions_df['topic_id'] == concept]
                if not topic_interactions.empty:
                    confidence = topic_interactions['confidence'].mean() if 'confidence' in topic_interactions else 0.5
                    stress = topic_interactions['stress_level'].mean() if 'stress_level' in topic_interactions else 0.3

            topic_stats.append({
                'topic': str(concept),
                'accuracy': float(accuracy),
                'attempts': int(attempts),
                'avg_difficulty': float(avg_difficulty),
                'confidence': float(confidence),
                'stress': float(stress),
                'mastery_level': _get_mastery_level(accuracy)
            })

        topic_stats.sort(key=lambda x: x['accuracy'], reverse=True)

        return {
            'strong_topics': [t for t in topic_stats if t['accuracy'] >= 0.7][:10],
            'weak_topics': [t for t in topic_stats if t['accuracy'] < 0.5][:10],
            'topic_stats': topic_stats
        }

    return {'strong_topics': [], 'weak_topics': [], 'topic_stats': []}


def _calculate_concept_mastery(practice_df, concept_features):
    """Calculate concept mastery scores."""
    mastery = {}

    if concept_features:
        for concept, feat in concept_features.items():
            accuracy = feat.get('accuracy', 0.5)
            attempts = feat.get('attempts', 0)
            if attempts > 0:
                mastery[concept] = min(1.0, accuracy * (1 + 0.05 * min(attempts / 10, 1)))
            else:
                mastery[concept] = 0.5

    if practice_df.empty:
        return mastery

    if not mastery and 'concept' in practice_df.columns:
        for concept, group in practice_df.groupby('concept'):
            accuracy = group['accuracy'].mean()
            attempts = len(group)
            mastery[concept] = min(1.0, accuracy * (1 + 0.05 * min(attempts / 10, 1)))

    return mastery


def _calculate_stability_index(practice_df, concept_features):
    """Calculate stability index for each concept."""
    stability = {}

    if concept_features:
        for concept, feat in concept_features.items():
            history = feat.get('concept_mastery_history', [])
            if len(history) >= 3:
                recent = history[-10:]
                mean = np.mean(recent)
                variance = np.var(recent)
                stability[concept] = max(0, min(1, 1 - (variance / 0.25)))
            else:
                stability[concept] = 0.5

    if practice_df.empty:
        return stability

    if not stability and 'concept' in practice_df.columns:
        for concept, group in practice_df.groupby('concept'):
            if len(group) >= 3:
                accuracies = group['accuracy'].values[-10:]
                variance = np.var(accuracies)
                stability[concept] = max(0, min(1, 1 - (variance / 0.25)))
            else:
                stability[concept] = 0.5

    return stability


def _calculate_confidence_calibration(interactions_df, practice_df):
    """Calculate confidence calibration metrics."""
    calibration = {
        'overall': 0.15,
        'by_difficulty': {'easy': 0.08, 'medium': 0.12, 'hard': 0.18, 'very_hard': 0.22},
        'calibration_error': 0.14,
        'overconfidence_bias': 0.08
    }

    if interactions_df.empty and practice_df.empty:
        return calibration

    df = interactions_df if not interactions_df.empty else practice_df

    if 'confidence' in df.columns and 'correct' in df.columns:
        confidence_col = 'confidence'
        correct_col = 'correct'
    elif 'confidence_index' in df.columns and 'accuracy' in df.columns:
        confidence_col = 'confidence_index'
        correct_col = 'accuracy'
    else:
        return calibration

    accuracy = df[correct_col].mean() if not df.empty else 0.5
    avg_confidence = df[confidence_col].mean() if not df.empty else 0.5

    calibration['overall'] = abs(accuracy - avg_confidence)
    calibration['calibration_error'] = calibration['overall']
    calibration['overconfidence_bias'] = max(0, avg_confidence - accuracy)

    if 'difficulty' in df.columns:
        for diff_bucket, (low, high) in [('easy', (0, 0.3)), ('medium', (0.3, 0.6)),
                                          ('hard', (0.6, 0.8)), ('very_hard', (0.8, 1.0))]:
            mask = df['difficulty'].between(low, high)
            if mask.any():
                subset = df[mask]
                acc = subset[correct_col].mean()
                conf = subset[confidence_col].mean()
                calibration['by_difficulty'][diff_bucket] = abs(acc - conf)

    return calibration


def _analyze_error_patterns(interactions_df, practice_df):
    """Analyze error patterns from interactions."""
    patterns = {
        'conceptual': 0.3,
        'careless': 0.3,
        'guess': 0.2,
        'overconfidence': 0.2,
        'by_topic': {}
    }

    df = interactions_df if not interactions_df.empty else practice_df

    if df.empty:
        return patterns

    if 'correct' in df.columns and 'confidence' in df.columns:
        correct_bool = df['correct'] if df['correct'].dtype == bool else df['correct'] > 0.5
        total_errors = (~correct_bool).sum()
        if total_errors > 0:
            conceptual_errors = ((~correct_bool) & (df['confidence'] < 0.4)).sum()
            careless_errors = ((~correct_bool) & (df['confidence'] > 0.7)).sum()
            guess_errors = ((~correct_bool) & (df['confidence'].between(0.4, 0.6))).sum()

            patterns['conceptual'] = min(1.0, conceptual_errors / max(1, total_errors))
            patterns['careless'] = min(1.0, careless_errors / max(1, total_errors))
            patterns['guess'] = min(1.0, guess_errors / max(1, total_errors))
            patterns['overconfidence'] = min(1.0, (careless_errors + (df['confidence'] > 0.8).sum() / max(1, len(df))) / 2)

    if 'topic_id' in df.columns or 'concept' in df.columns:
        topic_col = 'topic_id' if 'topic_id' in df.columns else 'concept'
        for topic, group in df.groupby(topic_col):
            if len(group) >= 3:
                correct_bool = group['correct'] if group['correct'].dtype == bool else group['correct'] > 0.5
                confs = group['confidence']
                errors = (~correct_bool).sum()
                if errors > 0:
                    patterns['by_topic'][str(topic)] = {
                        'conceptual': min(1.0, ((~correct_bool) & (confs < 0.4)).sum() / errors),
                        'careless': min(1.0, ((~correct_bool) & (confs > 0.7)).sum() / errors),
                        'guess': min(1.0, ((~correct_bool) & (confs.between(0.4, 0.6))).sum() / errors)
                    }

    return patterns


def _calculate_weakness_priority(concept_features, practice_df):
    """Calculate priority ranking of weaknesses."""
    weaknesses = []

    if concept_features:
        for concept, feat in concept_features.items():
            accuracy = feat.get('accuracy', 0.5)
            attempts = feat.get('attempts', 0)
            days_since = feat.get('days_since_last_practice', 0)

            weakness_score = (1 - accuracy) * 0.5 + (1 / (1 + attempts)) * 0.3 + (days_since / 30) * 0.2
            weakness_score = min(1.0, weakness_score)

            if weakness_score > 0.3:
                weaknesses.append({
                    'topic': str(concept),
                    'score': float(weakness_score),
                    'accuracy': float(accuracy),
                    'attempts': int(attempts),
                    'days_since': int(days_since),
                    'urgency': 'high' if weakness_score > 0.7 else 'medium' if weakness_score > 0.5 else 'low',
                    'recommendation': _get_weakness_recommendation(accuracy, attempts, days_since)
                })

    if not weaknesses and not practice_df.empty and 'concept' in practice_df.columns:
        for concept, group in practice_df.groupby('concept'):
            accuracy = group['accuracy'].mean()
            attempts = len(group)
            weakness_score = (1 - accuracy) * 0.5 + (1 / (1 + attempts)) * 0.3
            weakness_score = min(1.0, weakness_score)

            if weakness_score > 0.3:
                weaknesses.append({
                    'topic': str(concept),
                    'score': float(weakness_score),
                    'accuracy': float(accuracy),
                    'attempts': int(attempts),
                    'days_since': 0,
                    'urgency': 'high' if weakness_score > 0.7 else 'medium' if weakness_score > 0.5 else 'low',
                    'recommendation': _get_weakness_recommendation(accuracy, attempts, 0)
                })

    weaknesses.sort(key=lambda x: x['score'], reverse=True)
    return weaknesses[:20]


def _get_weakness_recommendation(accuracy, attempts, days_since):
    """Generate recommendation for weakness."""
    if accuracy < 0.4:
        return "Critical: Review fundamental concepts and practice basics"
    elif accuracy < 0.6:
        if attempts < 5:
            return "Needs more practice: Attempt more questions in this topic"
        else:
            return "Focus: Review mistakes and practice medium-level questions"
    elif days_since > 14:
        return "Long time since practice: Review and refresh your understanding"
    else:
        return "Maintain: Regular practice and review to strengthen understanding"


def _calculate_forgetting_curve(concept_features, daily_aggregates_df):
    """Calculate forgetting curve for concepts."""
    forgetting_curve = {
        'decay_constant': 0.085,
        'retention_scores': {},
        'review_priority': []
    }

    if concept_features:
        time_points = [1, 3, 7, 14, 30, 60, 90]

        for concept, feat in concept_features.items():
            current_retention = feat.get('accuracy', 0.5)
            days_since = feat.get('days_since_last_practice', 0)
            initial_retention = feat.get('initial_accuracy', current_retention)
            decay_rate = 0.05 + (0.05 * (1 - current_retention))

            retention_scores = {}
            for day in time_points:
                retention = current_retention * np.exp(-day / (30 * (1 + current_retention)))
                retention = min(1.0, max(0.0, retention))
                if day in [1, 3, 7, 14, 30]:
                    retention += 0.15 * current_retention
                retention_scores[day] = min(1.0, retention)

            forgetting_curve['retention_scores'][str(concept)] = {
                'current': float(current_retention),
                'initial': float(initial_retention),
                'days_since': int(days_since),
                'decay_rate': float(decay_rate),
                'retention_scores': retention_scores
            }

            if current_retention < 0.3:
                forgetting_curve['review_priority'].append({
                    'topic': str(concept),
                    'priority': 'high',
                    'reason': 'Critical retention drop - immediate review needed'
                })
            elif current_retention < 0.5:
                forgetting_curve['review_priority'].append({
                    'topic': str(concept),
                    'priority': 'medium',
                    'reason': 'Retention declining - review soon'
                })
            elif days_since > 14:
                forgetting_curve['review_priority'].append({
                    'topic': str(concept),
                    'priority': 'low',
                    'reason': 'Long time since practice - consider review'
                })

    priority_order = {'high': 0, 'medium': 1, 'low': 2}
    forgetting_curve['review_priority'].sort(key=lambda x: priority_order.get(x['priority'], 3))

    return forgetting_curve


def _analyze_behavior_profile(interactions_df, practice_df):
    """Analyze student behavior profile."""
    profile = {
        'cluster': 'balanced',
        'average_time_per_question': 60,
        'skip_rate': 0.05,
        'hard_question_rate': 0.2,
        'answer_change_frequency': 0.15,
        'difficulty_preference': 0.5,
        'persistence_score': 0.7,
        'help_seeking_behavior': 'moderate',
        'review_frequency': 'regular'
    }

    df = interactions_df if not interactions_df.empty else practice_df

    if df.empty:
        return profile

    if 'time_spent' in df.columns:
        profile['average_time_per_question'] = float(df['time_spent'].mean())
    elif 'response_time_ms' in df.columns:
        profile['average_time_per_question'] = float(df['response_time_ms'].mean() / 1000)
    elif 'normalized_response_time' in df.columns:
        # Convert normalized time to approximate seconds
        profile['average_time_per_question'] = float(df['normalized_response_time'].mean() * 10)
    else:
        profile['average_time_per_question'] = 60

    if 'hesitation_count' in df.columns:
        profile['answer_change_frequency'] = float(df['hesitation_count'].mean() / 5)
    elif 'answer_changed' in df.columns:
        profile['answer_change_frequency'] = float(df['answer_changed'].mean())

    if 'difficulty' in df.columns:
        profile['difficulty_preference'] = float(df['difficulty'].mean())

    if not df.empty and 'session_id' in df.columns:
        avg_attempts_per_session = df.groupby('session_id').size().mean()
        profile['persistence_score'] = min(1.0, avg_attempts_per_session / 20)

    avg_time = profile['average_time_per_question']
    change_freq = profile['answer_change_frequency']

    if avg_time < 30 and change_freq < 0.1:
        profile['cluster'] = 'impulsive'
    elif avg_time > 90 and change_freq > 0.3:
        profile['cluster'] = 'overthinker'
    elif avg_time > 60 and change_freq < 0.15:
        profile['cluster'] = 'methodical'
    elif avg_time < 40 and change_freq > 0.2:
        profile['cluster'] = 'exploratory'
    else:
        profile['cluster'] = 'balanced'

    return profile


def _calculate_difficulty_tolerance(practice_df):
    """Calculate difficulty tolerance metrics."""
    tolerance = {
        'max_sustainable': 0.5,
        'easy_accuracy': 0.8,
        'medium_accuracy': 0.6,
        'hard_accuracy': 0.4,
        'very_hard_accuracy': 0.2
    }

    if practice_df.empty:
        return tolerance

    if 'current_question_difficulty' in practice_df.columns and 'accuracy' in practice_df.columns:
        diff_buckets = {
            'easy': (0, 0.3),
            'medium': (0.3, 0.6),
            'hard': (0.6, 0.8),
            'very_hard': (0.8, 1.0)
        }

        for bucket, (low, high) in diff_buckets.items():
            mask = practice_df['current_question_difficulty'].between(low, high)
            if mask.any():
                acc = practice_df[mask]['accuracy'].mean()
                tolerance[f'{bucket}_accuracy'] = float(acc)

        max_diff = 0.5
        for diff in sorted(practice_df['current_question_difficulty'].unique()):
            mask = practice_df['current_question_difficulty'] >= diff
            if mask.any() and practice_df[mask]['accuracy'].mean() > 0.6:
                max_diff = diff
        tolerance['max_sustainable'] = float(max_diff)

    return tolerance


def _calculate_study_efficiency(practice_df, daily_aggregates_df):
    """Calculate study efficiency metrics."""
    efficiency = {
        'score': 0.5,
        'improvement_per_hour': 0,
        'trend': 'stable',
        'peak_efficiency_hours': '8:00 AM - 11:00 AM',
        'efficiency_by_topic': {}
    }

    if practice_df.empty:
        return efficiency

    if 'time_spent' in practice_df.columns:
        total_time = practice_df['time_spent'].sum() / 60  # Convert to minutes
    elif 'normalized_response_time' in practice_df.columns:
        # Use normalized_response_time as proxy (scaled to seconds)
        total_time = practice_df['normalized_response_time'].sum() / 10  # Approximate
    else:
        total_time = len(practice_df) * 10 / 60  # Default: 10 seconds per question

    correct_count = practice_df['accuracy'].sum()

    if total_time > 0:
        efficiency['score'] = min(1.0, correct_count / (total_time / 5))

    if not daily_aggregates_df.empty and len(daily_aggregates_df) >= 3:
        accuracies = daily_aggregates_df['accuracy'].values
        days = np.arange(len(accuracies))
        slope = np.polyfit(days, accuracies, 1)[0]
        efficiency['trend'] = 'improving' if slope > 0.02 else 'declining' if slope < -0.02 else 'stable'
        efficiency['improvement_per_hour'] = float(slope * 60)

    if 'concept' in practice_df.columns and 'accuracy' in practice_df.columns:
        for concept, group in practice_df.groupby('concept'):
            topic_time = group['time_spent'].sum() / 60
            if topic_time > 0:
                topic_efficiency = group['accuracy'].mean() / (topic_time / 10)
                efficiency['efficiency_by_topic'][str(concept)] = min(1.0, float(topic_efficiency))

    return efficiency


def _analyze_focus_loss(interactions_df, practice_df):
    """Analyze focus loss patterns."""
    focus_loss = {
        'frequency': 0.1,
        'last_detected': None,
        'triggers': [],
        'focus_duration': {'average': 25, 'max': 45, 'min': 10},
        'distraction_patterns': {
            'most_common': ['Time pressure', 'Fatigue', 'Task switching'],
            'recovery_time': 8.5
        }
    }

    df = interactions_df if not interactions_df.empty else practice_df

    if df.empty:
        return focus_loss

    focus_loss_events = 0

    if 'time_spent' in df.columns:
        mean_time = df['time_spent'].mean()
        std_time = df['time_spent'].std()
        spikes = (df['time_spent'] > mean_time + 2 * std_time).sum()
        focus_loss_events += spikes * 0.5

    if 'answer_changed' in df.columns:
        changes = df['answer_changed'].sum()
        focus_loss_events += changes * 0.3
    elif 'hesitation_count' in df.columns:
        hesitation = df['hesitation_count'].sum()
        focus_loss_events += hesitation * 0.1

    if 'accuracy' in df.columns and 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
        if len(df) >= 10:
            first_half_acc = df['accuracy'].head(len(df)//2).mean()
            second_half_acc = df['accuracy'].tail(len(df)//2).mean()
            if second_half_acc < first_half_acc - 0.15:
                focus_loss_events += 1

    total_questions = len(df)
    focus_loss['frequency'] = min(1.0, focus_loss_events / max(1, total_questions))

    if focus_loss_events > 0 and not df.empty:
        last_dt = safe_parse_datetime(df['timestamp'].max()) if 'timestamp' in df.columns else None
        focus_loss['last_detected'] = last_dt.isoformat() if last_dt else None

    triggers = []
    if focus_loss['frequency'] > 0.3:
        triggers.append("Significant accuracy drop in long sessions")
    if 'time_spent' in df.columns and (df['time_spent'] > df['time_spent'].quantile(0.95)).any():
        triggers.append("Unexpected time spikes on questions")
    if 'answer_changed' in df.columns and df['answer_changed'].mean() > 0.3:
        triggers.append("Frequent answer changes indicating uncertainty")

    focus_loss['triggers'] = triggers[:5]

    return focus_loss


def _analyze_stress_patterns(interactions_df, practice_df):
    """Analyze stress patterns from interactions."""
    stress_patterns = {
        'average_stress': 0.3,
        'max_stress': 0.6,
        'volatility': 0.1,
        'high_stress_moments': [],
        'risk_level': 'low',
        'trend': 'stable',
        'by_topic': {}
    }

    df = interactions_df if not interactions_df.empty else practice_df

    if df.empty:
        return stress_patterns

    stress_col = 'stress_level' if 'stress_level' in df.columns else 'stress_score'

    if stress_col in df.columns:
        stress_patterns['average_stress'] = float(df[stress_col].mean())
        stress_patterns['max_stress'] = float(df[stress_col].max())
        stress_patterns['volatility'] = float(df[stress_col].std())

        high_stress = df[df[stress_col] > 0.7]
        if not high_stress.empty:
            stress_patterns['high_stress_moments'] = [
                {'position': i, 'stress_level': float(row[stress_col])}
                for i, (_, row) in enumerate(high_stress.iterrows())
            ][:10]

        avg_stress = stress_patterns['average_stress']
        if avg_stress > 0.6:
            stress_patterns['risk_level'] = 'high'
        elif avg_stress > 0.4:
            stress_patterns['risk_level'] = 'moderate'
        else:
            stress_patterns['risk_level'] = 'low'

        if len(df) >= 10:
            first_half = df[stress_col].head(len(df)//2).mean()
            second_half = df[stress_col].tail(len(df)//2).mean()
            stress_patterns['trend'] = 'increasing' if second_half > first_half + 0.05 else 'decreasing' if second_half < first_half - 0.05 else 'stable'

        if 'topic_id' in df.columns or 'concept' in df.columns:
            topic_col = 'topic_id' if 'topic_id' in df.columns else 'concept'
            for topic, group in df.groupby(topic_col):
                if len(group) >= 3:
                    stress_patterns['by_topic'][str(topic)] = float(group[stress_col].mean())

    return stress_patterns


def _calculate_burnout_risk(interactions_df, practice_df, fatigue_index):
    """Calculate burnout risk."""
    burnout = {
        'current_risk': 0.3,
        'risk_level': 'low',
        'warning_signs': [],
        'recommendations': [],
        'trend': 'stable'
    }

    df = interactions_df if not interactions_df.empty else practice_df

    if df.empty:
        return burnout

    risk_factors = []

    fatigue_value = fatigue_index.get('current', 0.3)
    risk_factors.append(fatigue_value * 0.3)

    stress_col = 'stress_level' if 'stress_level' in df.columns else 'stress_score'
    if stress_col in df.columns:
        avg_stress = df[stress_col].mean()
        risk_factors.append(avg_stress * 0.25)

    if 'accuracy' in df.columns and len(df) >= 10:
        first_half = df['accuracy'].head(len(df)//2).mean()
        second_half = df['accuracy'].tail(len(df)//2).mean()
        if second_half < first_half:
            drop = (first_half - second_half) / first_half if first_half > 0 else 0
            risk_factors.append(min(0.3, drop * 0.5))

    if 'session_id' in df.columns:
        avg_per_session = df.groupby('session_id').size().mean()
        risk_factors.append(min(0.2, avg_per_session / 50 * 0.2))

    burnout['current_risk'] = min(1.0, sum(risk_factors))

    if burnout['current_risk'] > 0.6:
        burnout['risk_level'] = 'high'
        burnout['recommendations'] = [
            "Take a break for 30 minutes",
            "Reduce study intensity today",
            "Focus on review rather than new topics",
            "Get adequate sleep and rest"
        ]
        burnout['warning_signs'] = [
            "Performance declining",
            "High fatigue detected",
            "Stress levels elevated"
        ]
    elif burnout['current_risk'] > 0.35:
        burnout['risk_level'] = 'moderate'
        burnout['recommendations'] = [
            "Take short breaks between sessions",
            "Balance study with light activities",
            "Monitor energy levels"
        ]
        burnout['warning_signs'] = [
            "Moderate fatigue",
            "Performance stability decreasing"
        ]
    else:
        burnout['risk_level'] = 'low'
        burnout['recommendations'] = [
            "Continue your current study pattern",
            "Maintain regular breaks",
            "Stay hydrated and get proper sleep"
        ]
        burnout['warning_signs'] = []

    return burnout


def _calculate_learning_velocity(concept_features, daily_aggregates_df):
    """Calculate learning velocity metrics."""
    velocity = {
        'overall': 0,
        'by_topic': {},
        'trend': 'stable',
        'improvement_rate': 0
    }

    if concept_features:
        total_velocity = 0
        count = 0
        for concept, feat in concept_features.items():
            history = feat.get('concept_mastery_history', [])
            if len(history) >= 3:
                recent = history[-10:]
                if len(recent) >= 3:
                    x = np.arange(len(recent))
                    slope = np.polyfit(x, recent, 1)[0]
                    velocity['by_topic'][str(concept)] = float(max(-1, min(1, slope * 10)))
                    total_velocity += velocity['by_topic'][str(concept)]
                    count += 1

        if count > 0:
            velocity['overall'] = total_velocity / count

    if not daily_aggregates_df.empty and len(daily_aggregates_df) >= 3:
        accuracies = daily_aggregates_df['accuracy'].values
        days = np.arange(len(accuracies))
        slope = np.polyfit(days, accuracies, 1)[0]
        velocity['improvement_rate'] = float(slope)
        velocity['trend'] = 'improving' if slope > 0.02 else 'declining' if slope < -0.02 else 'stable'

    return velocity


def _calculate_trends(practice_df, daily_aggregates_df):
    """Calculate trend data for charts."""
    trends = {
        'accuracy': [],
        'difficulty': [],
        'stress': [],
        'fatigue': [],
        'confidence': [],
        'daily': []
    }

    if not practice_df.empty and 'timestamp' in practice_df.columns:
        practice_df = practice_df.sort_values('timestamp')

        window = min(10, len(practice_df))
        trends['accuracy'] = practice_df['accuracy'].rolling(window, min_periods=1).mean().tolist()

        if 'current_question_difficulty' in practice_df.columns:
            trends['difficulty'] = practice_df['current_question_difficulty'].rolling(window, min_periods=1).mean().tolist()

        if 'stress_score' in practice_df.columns:
            trends['stress'] = practice_df['stress_score'].rolling(window, min_periods=1).mean().tolist()

        if 'fatigue_indicator' in practice_df.columns:
            trends['fatigue'] = practice_df['fatigue_indicator'].rolling(window, min_periods=1).mean().tolist()

        if 'confidence_index' in practice_df.columns:
            trends['confidence'] = practice_df['confidence_index'].rolling(window, min_periods=1).mean().tolist()

    if not daily_aggregates_df.empty:
        for _, row in daily_aggregates_df.iterrows():
            trends['daily'].append({
                'date': row.get('date', ''),
                'accuracy': float(row.get('accuracy', 0)),
                'questions': int(row.get('total_attempts', row.get('questions_attempted', 0))),
                'topics_covered': int(row.get('topics_covered', 0))
            })

    return trends


def _generate_time_allocation(weakness_priority, concept_features):
    """Generate time allocation recommendations."""
    time_allocation = []

    if weakness_priority:
        for item in weakness_priority[:5]:
            time_allocation.append({
                'topic': item['topic'],
                'recommended_minutes': max(15, min(45, int(30 * (1 + item['score'])))),
                'priority': item['urgency'],
                'reason': item.get('recommendation', 'Focus on this topic'),
                'order': len(time_allocation) + 1
            })

    return time_allocation


def _summarize_predictions(micro, meso, macro):
    """Summarize predictions from all models."""
    summary = {
        'micro_count': len(micro),
        'meso_count': len(meso),
        'macro_keys': len(macro) if isinstance(macro, dict) else 0,
        'average_retention': 0.5,
        'topics_with_low_retention': [],
        'topics_with_high_retention': []
    }

    if micro:
        retentions = [p.get('current_retention', p.get('retention_probability', 0.5)) for p in micro]
        summary['average_retention'] = float(np.mean(retentions))

        low_retention = [p.get('topic_id', 'unknown') for p in micro if p.get('current_retention', 0.5) < 0.3]
        summary['topics_with_low_retention'] = low_retention[:10]

        high_retention = [p.get('topic_id', 'unknown') for p in micro if p.get('current_retention', 0.5) > 0.8]
        summary['topics_with_high_retention'] = high_retention[:10]

    return summary


def _generate_recommendations(weakness_priority, burnout_risk, fatigue_index, concept_mastery, study_efficiency):
    """Generate personalized recommendations."""
    recommendations = []

    if weakness_priority:
        top_weakness = weakness_priority[0] if weakness_priority else None
        if top_weakness:
            recommendations.append({
                'priority': 'high',
                'category': 'weakness_focus',
                'message': f"Focus on improving '{top_weakness['topic']}'",
                'detail': top_weakness.get('recommendation', 'Practice more in this topic'),
                'action_items': [
                    f"Review fundamental concepts of {top_weakness['topic']}",
                    f"Practice 5-10 medium difficulty questions on {top_weakness['topic']}",
                    f"Track improvement with short quizzes"
                ]
            })

    if burnout_risk['risk_level'] == 'high':
        recommendations.append({
            'priority': 'high',
            'category': 'wellness',
            'message': "High burnout risk detected - prioritize rest",
            'detail': "Your stress and fatigue levels indicate need for recovery",
            'action_items': [
                "Take a break for 30 minutes",
                "Reduce study intensity for today",
                "Focus on light review instead of new topics",
                "Get adequate sleep and rest"
            ]
        })
    elif burnout_risk['risk_level'] == 'moderate':
        recommendations.append({
            'priority': 'medium',
            'category': 'wellness',
            'message': "Moderate burnout risk - balance your study",
            'detail': "Monitor your energy levels and take regular breaks",
            'action_items': [
                "Take 5-10 minute breaks every 30 minutes",
                "Balance study with light activities",
                "Stay hydrated and maintain proper sleep"
            ]
        })

    if fatigue_index['risk_level'] == 'high':
        recommendations.append({
            'priority': 'high',
            'category': 'fatigue_management',
            'message': "High fatigue detected - rest and recover",
            'detail': fatigue_index.get('recommendation', 'Take a break'),
            'action_items': [
                "End current session",
                "Take a 30-60 minute break",
                "Review only light topics if continuing"
            ]
        })

    if concept_mastery:
        weak_concepts = [c for c, v in concept_mastery.items() if v < 0.4]
        if weak_concepts:
            recommendations.append({
                'priority': 'medium',
                'category': 'concept_review',
                'message': f"Review {len(weak_concepts)} concepts with low mastery",
                'detail': f"Concepts: {', '.join(weak_concepts[:3])}" + (f" and {len(weak_concepts)-3} more" if len(weak_concepts) > 3 else ""),
                'action_items': [
                    "Review core concepts and definitions",
                    "Practice with easier questions first",
                    "Use spaced repetition for these topics"
                ]
            })

    if study_efficiency['score'] < 0.4:
        recommendations.append({
            'priority': 'medium',
            'category': 'efficiency',
            'message': "Improve study efficiency",
            'detail': "Your current study efficiency is below optimal levels",
            'action_items': [
                "Use focused 25-minute study sessions (Pomodoro)",
                "Minimize distractions during study",
                "Review mistakes immediately after practice"
            ]
        })

    return recommendations


def _generate_chart_data(practice_df, daily_aggregates_df, trend_data):
    """Generate chart data for frontend."""
    charts = {
        'accuracy_over_time': [],
        'difficulty_over_time': [],
        'concept_mastery': [],
        'burnout_trend': [],
        'weekly_progress': [],
        'concept_radar': [],
        'stress_over_time': [],
        'fatigue_over_time': [],
        'confidence_over_time': [],
        'topic_performance': []
    }

    if not daily_aggregates_df.empty:
        for _, row in daily_aggregates_df.iterrows():
            charts['accuracy_over_time'].append({
                'date': str(row.get('date', '')),
                'value': float(row.get('accuracy', 0))
            })

    if not charts['accuracy_over_time'] and not practice_df.empty:
        practice_df = practice_df.sort_values('timestamp')
        for _, row in practice_df.iterrows():
            charts['accuracy_over_time'].append({
                'date': str(row.get('timestamp', '')),
                'value': float(row.get('accuracy', 0))
            })

    if not practice_df.empty and 'current_question_difficulty' in practice_df.columns:
        practice_df = practice_df.sort_values('timestamp')
        for _, row in practice_df.iterrows():
            charts['difficulty_over_time'].append({
                'date': str(row.get('timestamp', '')),
                'value': float(row.get('current_question_difficulty', 0))
            })

    concept_features = practice_df.groupby('concept')['accuracy'].mean().to_dict() if 'concept' in practice_df.columns else {}
    for concept, mastery in concept_features.items():
        charts['concept_radar'].append({
            'concept': str(concept),
            'mastery': float(mastery)
        })

    if 'concept' in practice_df.columns:
        for concept, group in practice_df.groupby('concept'):
            charts['topic_performance'].append({
                'topic': str(concept),
                'accuracy': float(group['accuracy'].mean()),
                'attempts': int(len(group)),
                'avg_difficulty': float(group['current_question_difficulty'].mean()) if 'current_question_difficulty' in group else 0.5
            })

    return charts


def _get_mastery_level(accuracy):
    """Get mastery level from accuracy score."""
    if accuracy >= 0.85:
        return 'excellent'
    elif accuracy >= 0.7:
        return 'good'
    elif accuracy >= 0.5:
        return 'moderate'
    elif accuracy >= 0.3:
        return 'poor'
    else:
        return 'critical'

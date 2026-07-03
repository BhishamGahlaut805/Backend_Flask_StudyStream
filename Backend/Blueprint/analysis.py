from flask import Blueprint, request, jsonify, current_app
import logging
from datetime import datetime
import numpy as np
import pandas as pd

from Utils.validators import validate_student_id
from Utils.decorators import handle_errors, log_request

analysis_bp = Blueprint('analysis', __name__)
logger = logging.getLogger(__name__)


@analysis_bp.route('/practice/live', methods=['POST'])
@handle_errors
@log_request
def analyze_practice_live():
    """Live analysis during practice mode."""
    data = request.get_json()

    if not data or 'student_id' not in data:
        return jsonify({'success': False, 'error': 'Missing student_id'}), 400

    student_id = data['student_id']
    concept = data.get('concept', 'general')
    current_features = data.get('practice_features', [])
    concept_history = data.get('concept_history', [])
    session_features = data.get('session_features', [])

    validate_student_id(student_id)

    data_manager = current_app.prediction_service._get_data_manager(student_id)

    analysis = {
        'timestamp': datetime.now().isoformat(),
        'student_id': student_id,
        'concept': concept,
        'models': {}
    }

    # 1. Practice Difficulty Model
    if current_features:
        practice_pred = current_app.prediction_service.predict_practice_difficulty(
            student_id, current_features
        )
        analysis['models']['practice_difficulty'] = {
            'next_difficulty': practice_pred['predicted_difficulty'],
            'confidence': practice_pred['confidence'],
            'method': practice_pred['method']
        }

    # 2. Learning Velocity Model
    if concept_history and len(concept_history) >= 5:
        velocity_features = []
        for mastery in concept_history[-30:]:
            velocity_features.append([
                mastery, 1.0, 0.5, 0.6, 0.7, 0.8, 30.0, 0.1, 0.6
            ])

        velocity_pred = current_app.prediction_service.predict_learning_velocity(
            student_id, concept, velocity_features
        )
        analysis['models']['learning_velocity'] = velocity_pred

    # 3. Burnout Risk Model
    if session_features:
        burnout_pred = current_app.prediction_service.predict_burnout_risk(
            student_id, session_features
        )
        analysis['models']['burnout_risk'] = burnout_pred
    else:
        practice_df = data_manager.load_practice_features()
        if not practice_df.empty and len(practice_df) >= 10:
            recent = practice_df.tail(20)
            session_data = [
                float(recent['accuracy'].mean()),
                float(recent['accuracy'].diff().mean()),
                float(recent['stress_score'].diff().mean()) if 'stress_score' in recent.columns else 0,
                float(recent['normalized_response_time'].diff().mean()) if 'normalized_response_time' in recent.columns else 0,
                float(recent['fatigue_indicator'].iloc[-1] - recent['fatigue_indicator'].iloc[0]) if 'fatigue_indicator' in recent.columns else 0,
                float(recent['time_spent'].sum() / 60) if 'time_spent' in recent.columns else 10,
                1,
                float(recent[recent['current_question_difficulty'] > 0.7]['accuracy'].mean()) if 'current_question_difficulty' in recent.columns else 0.5,
                float(1 - recent['accuracy'].std()),
                float(recent['confidence_index'].diff().mean()) if 'confidence_index' in recent.columns else 0,
                float((recent['time_spent'] < 5).mean()) if 'time_spent' in recent.columns else 0,
                float(recent['accuracy'].iloc[-len(recent)//2:].mean() - recent['accuracy'].iloc[:len(recent)//2].mean())
            ]
            burnout_pred = current_app.prediction_service.predict_burnout_risk(
                student_id, session_data
            )
            analysis['models']['burnout_risk'] = burnout_pred

    # 4. Adaptive Scheduling Model
    if concept_history:
        concept_features = data_manager.load_concept_features()
        if concept in concept_features:
            feat = concept_features[concept]
            priority_input = {
                concept: [
                    feat.get('accuracy', 0.5),
                    0.5,
                    feat.get('avg_difficulty', 0.5) * 2,
                    0.5,
                    0.5,
                    0.5
                ]
            }
            priority_pred = current_app.prediction_service.predict_priority_scores(
                student_id, priority_input
            )
            if priority_pred['priorities']:
                analysis['models']['adaptive_scheduling'] = {
                    'priority_score': priority_pred['priorities'][0]['priority_score'],
                    'study_plan': priority_pred.get('study_plan', [])
                }

    # 5. Real Exam Difficulty Model
    exam_df = data_manager.load_exam_features()
    if not exam_df.empty:
        exam_features = [
            exam_df['overall_accuracy_avg'].iloc[-1] if 'overall_accuracy_avg' in exam_df.columns else 0.5,
            exam_df['avg_difficulty_handled'].iloc[-1] if 'avg_difficulty_handled' in exam_df.columns else 0.5,
            exam_df['readiness_score'].iloc[-1] if 'readiness_score' in exam_df.columns else 0.5,
            exam_df['consistency_index'].iloc[-1] if 'consistency_index' in exam_df.columns else 0.5,
            0, 0.5, 0.5, 0.5
        ]
        exam_pred = current_app.prediction_service.predict_exam_difficulty(
            student_id, exam_features
        )
        analysis['models']['exam_difficulty'] = {
            'recommended_difficulty': exam_pred['recommended_difficulty'],
            'level': exam_pred.get('difficulty_level', 'medium')
        }

    return jsonify({'success': True, 'analysis': analysis})


@analysis_bp.route('/practice/results', methods=['POST'])
@handle_errors
@log_request
def analyze_practice_results():
    """Practice mode results analysis."""
    data = request.get_json()

    if not data or 'student_id' not in data:
        return jsonify({'success': False, 'error': 'Missing student_id'}), 400

    student_id = data['student_id']
    session_data = data.get('practice_session', {})
    responses = session_data.get('responses', [])
    validate_student_id(student_id)

    if not responses:
        return jsonify({'success': False, 'error': 'No responses data'}), 400

    df = pd.DataFrame(responses)

    total_questions = len(df)
    correct_count = df['correct'].sum() if 'correct' in df else 0
    accuracy = correct_count / total_questions if total_questions > 0 else 0

    diff_performance = {}
    if 'difficulty' in df.columns:
        for diff_level in [0.3, 0.5, 0.7]:
            mask = df['difficulty'] >= diff_level
            if mask.any():
                diff_performance[f'above_{int(diff_level*10)}'] = float(df.loc[mask, 'correct'].mean())

    time_analysis = {}
    if 'time_spent' in df.columns:
        time_analysis = {
            'avg_time': float(df['time_spent'].mean()),
            'total_time': float(df['time_spent'].sum()),
            'fastest': float(df['time_spent'].min()),
            'slowest': float(df['time_spent'].max())
        }

    concept_performance = {}
    if 'concept' in df.columns:
        concept_performance = df.groupby('concept')['correct'].mean().to_dict()
        concept_performance = {k: float(v) for k, v in concept_performance.items()}

    analysis = {
        'session_summary': {
            'total_questions': total_questions,
            'correct_count': int(correct_count),
            'accuracy': float(accuracy),
            'duration_minutes': session_data.get('duration', 0)
        },
        'difficulty_performance': diff_performance,
        'time_analysis': time_analysis,
        'concept_performance': concept_performance,
        'strengths': [c for c, acc in concept_performance.items() if acc > 0.7][:3],
        'weaknesses': [c for c, acc in concept_performance.items() if acc < 0.4][:3]
    }

    return jsonify({'success': True, 'analysis': analysis})


@analysis_bp.route('/real-exam/results', methods=['POST'])
@handle_errors
@log_request
def analyze_real_exam_results():
    """Real exam mode results analysis."""
    data = request.get_json()

    if not data or 'student_id' not in data:
        return jsonify({'success': False, 'error': 'Missing student_id'}), 400

    student_id = data['student_id']
    exam_data = data.get('exam_session', {})
    questions = exam_data.get('questions', [])
    validate_student_id(student_id)

    if not questions:
        return jsonify({'success': False, 'error': 'No exam data'}), 400

    df = pd.DataFrame(questions)

    total_questions = len(df)
    score = df['correct'].sum() if 'correct' in df else 0
    percentage = (score / total_questions * 100) if total_questions > 0 else 0

    difficulty_breakdown = {}
    if 'difficulty' in df.columns:
        for diff in sorted(df['difficulty'].unique()):
            mask = df['difficulty'] == diff
            if mask.any():
                difficulty_breakdown[f'diff_{int(diff*10)}'] = {
                    'count': int(mask.sum()),
                    'accuracy': float(df.loc[mask, 'correct'].mean())
                }

    time_stats = {}
    if 'time_spent' in df.columns:
        time_stats = {
            'avg_per_question': float(df['time_spent'].mean()),
            'total_time': float(df['time_spent'].sum()),
            'efficiency_score': float(1 - (df['time_spent'].mean() / 120))
        }

    concept_mastery = {}
    if 'concept' in df.columns:
        concept_mastery = df.groupby('concept')['correct'].mean().to_dict()
        concept_mastery = {k: float(v) for k, v in concept_mastery.items()}

    data_manager = current_app.prediction_service._get_data_manager(student_id)

    learning_velocity_updates = {}
    for concept, mastery in concept_mastery.items():
        concept_features = data_manager.load_concept_features()
        if concept in concept_features:
            history = concept_features[concept].get('concept_mastery_history', [])
            history.append(mastery)
            if len(history) >= 5:
                vel_pred = current_app.prediction_service.predict_learning_velocity(
                    student_id, concept, history[-30:]
                )
                learning_velocity_updates[concept] = vel_pred

    analysis = {
        'exam_summary': {
            'score': int(score),
            'total_questions': total_questions,
            'percentage': float(percentage),
            'percentile': exam_data.get('percentile', 50)
        },
        'difficulty_breakdown': difficulty_breakdown,
        'time_analysis': time_stats,
        'concept_mastery': concept_mastery,
        'learning_velocity_updates': learning_velocity_updates,
        'readiness_score': float(percentage / 100 * 0.7 + 0.3)
    }

    return jsonify({'success': True, 'analysis': analysis})
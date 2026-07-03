"""
Schedule blueprint - Handles learning schedule API endpoints with MongoDB storage.
"""
from flask import Blueprint, request, jsonify, current_app
import logging
from datetime import datetime

from Utils.validators import validate_student_id, validate_subject
from Utils.decorators import handle_errors, log_request

schedule_bp = Blueprint('schedule', __name__)
logger = logging.getLogger(__name__)


def _prediction_service():
    return getattr(current_app, 'retention_prediction_service', getattr(current_app, 'prediction_service', None))


def _schedule_service():
    return getattr(current_app, 'retention_schedule_service', getattr(current_app, 'schedule_service', None))


@schedule_bp.route('/daily/<user_id>', methods=['GET'])
@handle_errors
@log_request
def get_daily_schedule(user_id):
    """Get today's learning schedule from MongoDB."""
    validate_student_id(user_id)
    subject = request.args.get('subject')

    predictions = _prediction_service().get_all_predictions(user_id, subject)
    schedule = _schedule_service().generate_daily_schedule(
        user_id, subject, predictions
    )

    return jsonify({
        'success': True,
        'user_id': user_id,
        'subject': subject,
        'schedule': schedule,
        'storage': 'mongodb',
        'timestamp': datetime.now().isoformat()
    }), 200


@schedule_bp.route('/next-questions/<user_id>', methods=['GET'])
@handle_errors
@log_request
def get_next_questions(user_id):
    """Get next set of questions for immediate learning from MongoDB."""
    validate_student_id(user_id)
    subject = request.args.get('subject')
    current_stress = request.args.get('current_stress', 0.3, type=float)
    current_fatigue = request.args.get('current_fatigue', 0.3, type=float)

    questions = _schedule_service().get_next_questions(
        user_id, subject, current_stress, current_fatigue
    )

    return jsonify({
        'success': True,
        'user_id': user_id,
        'subject': subject,
        'questions': questions,
        'timestamp': datetime.now().isoformat()
    }), 200


@schedule_bp.route('/subject-repetition/<user_id>/<subject>', methods=['GET'])
@handle_errors
@log_request
def get_subject_repetition(user_id, subject):
    """Get subject-level repetition schedule from MongoDB."""
    validate_student_id(user_id)
    validate_subject(subject)

    schedule = _schedule_service().get_subject_repetition_schedule(
        user_id, subject
    )

    return jsonify({
        'success': True,
        'user_id': user_id,
        'subject': subject,
        'schedule': schedule
    }), 200


@schedule_bp.route('/topic-repetition/<user_id>/<topic_id>', methods=['GET'])
@handle_errors
@log_request
def get_topic_repetition(user_id, topic_id):
    """Get topic-level repetition schedule from MongoDB."""
    validate_student_id(user_id)

    schedule = _schedule_service().get_topic_repetition_schedule(
        user_id, topic_id
    )

    return jsonify({
        'success': True,
        'user_id': user_id,
        'topic_id': topic_id,
        'schedule': schedule
    }), 200


@schedule_bp.route('/optimal-study-times/<user_id>', methods=['GET'])
@handle_errors
@log_request
def get_optimal_study_times(user_id):
    """Get optimal study times based on stress and fatigue patterns from MongoDB."""
    validate_student_id(user_id)
    subject = request.args.get('subject')

    optimal_times = _schedule_service().get_optimal_study_times(user_id, subject)

    return jsonify({
        'success': True,
        'user_id': user_id,
        'optimal_times': optimal_times
    }), 200


@schedule_bp.route('/update', methods=['POST'])
@handle_errors
@log_request
def update_schedule():
    """Update schedule based on performance using MongoDB data."""
    data = request.get_json()
    user_id = data.get('user_id')
    subject = data.get('subject')
    validate_student_id(user_id)

    predictions = _prediction_service().get_all_predictions(user_id, subject)

    if not predictions.get('micro'):
        return jsonify({'error': 'No predictions available'}), 404

    schedule = _schedule_service().generate_daily_schedule(
        user_id, subject, predictions
    )

    question_sequence = _prediction_service().get_question_sequence(
        user_id, subject, 'immediate', 10
    )

    return jsonify({
        'success': True,
        'user_id': user_id,
        'subject': subject,
        'schedule': schedule,
        'question_sequence': question_sequence,
        'updated_at': datetime.now().isoformat()
    }), 200
    
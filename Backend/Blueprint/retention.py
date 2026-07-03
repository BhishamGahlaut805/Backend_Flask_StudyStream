"""
Retention blueprint - Handles retention-related API endpoints with MongoDB storage.
"""
from flask import Blueprint, request, jsonify, current_app
import logging
import uuid
from datetime import datetime
import numpy as np
import pandas as pd

from Utils.validators import validate_student_id
from Utils.helpers import normalize_subject, get_display_subject, default_topics_for_subject
from Utils.decorators import handle_errors, log_request

retention_bp = Blueprint('retention', __name__)
logger = logging.getLogger(__name__)

# In-memory session state
_RETENTION_SESSIONS = {}


def _get_services():
    """Get retention services from current app."""
    return {
        'prediction': current_app.retention_prediction_service,
        'schedule': current_app.schedule_service,
        'training': current_app.retention_training_service,
        'performance': current_app.performance_service
    }


def _persist_interactions_to_mongodb(user_id: str, session_id: str, subject: str, responses: list):
    """
    Persist interactions to MongoDB using the data manager.
    This replaces the old CSV-based persistence.
    """
    if not responses:
        return

    try:
        data_manager = current_app.retention_prediction_service._get_data_manager(user_id)

        # Save each interaction
        for response in responses:
            interaction = {
                'user_id': user_id,
                'session_id': session_id,
                'subject': normalize_subject(subject),
                'topic_id': response.get('topic_id') or response.get('concept_area') or response.get('topic') or 'unknown_topic',
                'question_id': response.get('question_id'),
                'correct': bool(response.get('correct', False)),
                'response_time_ms': float(response.get('time_spent', response.get('response_time_ms', 0)) or 0),
                'confidence': float(response.get('confidence', 0.5) or 0.5),
                'difficulty': float(response.get('difficulty', 0.5) or 0.5),
                'hesitation_count': int(response.get('hesitation_count', response.get('answer_changes', 0)) or 0),
                'fatigue_index': float(response.get('fatigue_index', 0.3) or 0.3),
                'focus_score': float(response.get('focus_score', 0.7) or 0.7),
                'stress_level': float(response.get('stress_level', 0.3) or 0.3),
                'attempt_number': int(response.get('attempt_number', 1) or 1),
                'streak': int(response.get('streak', 0) or 0),
                'timestamp': response.get('timestamp', datetime.now().isoformat())
            }
            data_manager.save_interaction(interaction)

        # After saving interactions, process them as features
        # This creates the practice_features and sequences
        data_manager.append_practice_attempts_as_features(responses)

        logger.info(f"Persisted {len(responses)} interactions and features for user {user_id}")

    except Exception as e:
        logger.error(f"Error persisting interactions to MongoDB: {e}")
        raise


@retention_bp.route('/health', methods=['GET'])
def retention_health():
    """Health endpoint scoped to retention blueprint."""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'service': 'retention',
        'timestamp': datetime.now().isoformat()
    }), 200


@retention_bp.route('/session/start', methods=['POST'])
@handle_errors
@log_request
def start_retention_session():
    """Start a retention session with MongoDB storage."""
    data = request.get_json() or {}

    user_id = data.get('student_id') or data.get('user_id')
    original_subject = data.get('subject')
    subject = normalize_subject(original_subject)
    display_subject = get_display_subject(subject)
    topics = data.get('topics') or default_topics_for_subject(display_subject)
    session_type = data.get('session_type', 'practice')
    session_id = data.get('session_id') or str(uuid.uuid4())

    validate_student_id(user_id)

    services = _get_services()

    # Get data manager for this user
    data_manager = current_app.retention_prediction_service._get_data_manager(user_id)

    # Check if training is needed based on stored data
    training_needed = services['training'].check_retrain_needed(user_id)
    if training_needed.get('needed'):
        services['training'].train_all_models(user_id, training_needed)

    # Get predictions from MongoDB
    predictions = services['prediction'].get_all_predictions(user_id, display_subject)
    question_batch = services['schedule'].get_next_questions(user_id, display_subject, 0.3, 0.3)

    # Store session in memory
    _RETENTION_SESSIONS[session_id] = {
        'session_id': session_id,
        'user_id': str(user_id),
        'subject': original_subject,
        'display_subject': display_subject,
        'topics': topics,
        'session_type': session_type,
        'started_at': datetime.now().isoformat(),
        'events_count': 0
    }

    # Save session to MongoDB
    data_manager.save_session({
        'session_id': session_id,
        'user_id': str(user_id),
        'subject': original_subject,
        'display_subject': display_subject,
        'topics': topics,
        'session_type': session_type,
        'started_at': datetime.now().isoformat(),
        'events_count': 0
    })

    return jsonify({
        'success': True,
        'session_id': session_id,
        'user_id': str(user_id),
        'subject': original_subject,
        'topics': topics,
        'session_type': session_type,
        'predictions': {
            'micro': predictions.get('micro', []),
            'meso': predictions.get('meso', []),
            'macro': predictions.get('macro', {}),
            'forgetting_curves': predictions.get('forgetting_curves', {}),
            'stressFatigue': services['prediction'].get_stress_fatigue_predictions(user_id, display_subject)
        },
        'questions': question_batch.get('questions', []),
        'metadata': {
            'training_needed': training_needed,
            'recommended_break': question_batch.get('recommended_break', False),
            'remaining_in_batch': question_batch.get('remaining_in_batch', 0),
            'storage': 'mongodb',
            'timestamp': datetime.now().isoformat()
        }
    }), 200


@retention_bp.route('/session/<session_id>/next', methods=['POST'])
@handle_errors
@log_request
def get_next_session_questions(session_id):
    """Return next questions and updated predictions after recent answers."""
    session = _RETENTION_SESSIONS.get(session_id)
    data = request.get_json() or {}
    responses = data.get('responses', [])

    inferred_subject = None
    if responses and isinstance(responses, list):
        first = responses[0] or {}
        inferred_subject = first.get('subject') or first.get('subject_id')

    user_id = data.get('student_id') or data.get('user_id') or (session or {}).get('user_id')
    original_subject = data.get('subject') or inferred_subject or (session or {}).get('subject')
    subject = normalize_subject(original_subject)
    display_subject = get_display_subject(subject)
    current_stress = float(data.get('current_stress', 0.3) or 0.3)
    current_fatigue = float(data.get('current_fatigue', 0.3) or 0.3)

    validate_student_id(user_id)

    services = _get_services()

    # Persist interactions to MongoDB
    if responses:
        _persist_interactions_to_mongodb(str(user_id), session_id, display_subject, responses)

    # Check training based on MongoDB data
    training_needed = services['training'].check_retrain_needed(user_id)
    if training_needed.get('needed'):
        services['training'].train_all_models(user_id, training_needed)

    if session:
        session['events_count'] = int(session.get('events_count', 0)) + len(responses)

    question_batch = services['schedule'].get_next_questions(str(user_id), display_subject, current_stress, current_fatigue)
    predictions = services['prediction'].get_all_predictions(str(user_id), display_subject)

    return jsonify({
        'success': True,
        'session_id': session_id,
        'questions': question_batch.get('questions', []),
        'predictions': {
            'micro': predictions.get('micro', []),
            'meso': predictions.get('meso', []),
            'macro': predictions.get('macro', {}),
            'stress_fatigue': services['prediction'].get_stress_fatigue_predictions(str(user_id), display_subject)
        },
        'metadata': {
            'recommended_break': question_batch.get('recommended_break', False),
            'remaining_in_batch': question_batch.get('remaining_in_batch', 0),
            'timestamp': datetime.now().isoformat()
        }
    }), 200


@retention_bp.route('/session/<session_id>/complete', methods=['POST'])
@handle_errors
@log_request
def complete_retention_session(session_id):
    """Finalize a retention session with MongoDB storage."""
    data = request.get_json() or {}
    session = _RETENTION_SESSIONS.get(session_id, {})

    user_id = data.get('student_id') or data.get('user_id') or session.get('user_id')
    original_subject = data.get('subject') or session.get('subject')
    subject = normalize_subject(original_subject)
    display_subject = get_display_subject(subject)
    answers = data.get('answers', [])

    validate_student_id(user_id)

    services = _get_services()

    if answers:
        _persist_interactions_to_mongodb(str(user_id), session_id, display_subject, answers)

    # Check training based on MongoDB data
    training_needed = services['training'].check_retrain_needed(user_id)
    if training_needed.get('needed'):
        services['training'].train_all_models(user_id, training_needed)

    predictions = services['prediction'].get_all_predictions(str(user_id), display_subject)
    schedule = services['schedule'].generate_daily_schedule(str(user_id), display_subject, predictions)

    # Remove from memory
    _RETENTION_SESSIONS.pop(session_id, None)

    return jsonify({
        'success': True,
        'session_id': session_id,
        'analysis': {
            'training_needed': training_needed,
            'retention_summary': services['prediction'].get_retention_summary(str(user_id), display_subject)
        },
        'updated_predictions': {
            'micro': predictions.get('micro', []),
            'meso': predictions.get('meso', []),
            'macro': predictions.get('macro', {}),
            'forgetting_curves': predictions.get('forgetting_curves', {}),
            'stress_fatigue': services['prediction'].get_stress_fatigue_predictions(str(user_id), display_subject)
        },
        'schedule': schedule,
        'timestamp': datetime.now().isoformat()
    }), 200


# ==================== PREDICTIONS ROUTES ====================

@retention_bp.route('/predictions/<user_id>', methods=['GET'])
@handle_errors
def get_predictions(user_id):
    """Get all retention predictions for a user from MongoDB."""
    validate_student_id(user_id)
    subject = request.args.get('subject')
    predictions = current_app.retention_prediction_service.get_all_predictions(user_id, subject)

    return jsonify({
        'success': True,
        'user_id': user_id,
        'subject': subject,
        'predictions': predictions,
        'storage': 'mongodb'
    }), 200


@retention_bp.route('/predictions/update/<user_id>', methods=['POST'])
@handle_errors
@log_request
def update_predictions_after_answers(user_id):
    """
    Update predictions after recent answers.
    This is the route that was missing and causing 404 errors.
    """
    validate_student_id(user_id)

    data = request.get_json() or {}
    original_subject = data.get('subject')
    subject = normalize_subject(original_subject)
    display_subject = get_display_subject(subject)
    answers = data.get('answers', [])

    services = _get_services()

    if answers:
        session_id = data.get('session_id', f"update_{int(datetime.now().timestamp())}")
        _persist_interactions_to_mongodb(str(user_id), session_id, display_subject, answers)

    # Check if training is needed
    training_needed = services['training'].check_retrain_needed(user_id)
    if training_needed.get('needed'):
        services['training'].train_all_models(user_id, training_needed)

    # Get updated predictions
    predictions = services['prediction'].get_all_predictions(str(user_id), display_subject)

    # ==================== FIX: Safely load sequence data with try-catch ====================
    data_manager = current_app.retention_prediction_service._get_data_manager(user_id)
    practice_df = data_manager.load_practice_features()

    # Safely load sequences with fallback
    try:
        micro_sequences = data_manager.load_micro_sequences()
    except AttributeError:
        logger.warning(f"load_micro_sequences not available, using empty DataFrame")
        micro_sequences = pd.DataFrame()

    try:
        meso_sequences = data_manager.load_meso_sequences()
    except AttributeError:
        logger.warning(f"load_meso_sequences not available, using empty DataFrame")
        meso_sequences = pd.DataFrame()

    try:
        macro_sequences = data_manager.load_macro_sequences()
    except AttributeError:
        logger.warning(f"load_macro_sequences not available, using empty DataFrame")
        macro_sequences = pd.DataFrame()
    # ==================== END OF FIX ====================

    sequence_status = {
        'practice_features_count': len(practice_df),
        'micro_sequences_count': len(micro_sequences) if not micro_sequences.empty else 0,
        'meso_sequences_count': len(meso_sequences) if not meso_sequences.empty else 0,
        'macro_sequences_count': len(macro_sequences) if not macro_sequences.empty else 0
    }

    # Get model status
    models_ready = {
        'micro': bool(predictions.get('micro')),
        'meso': bool(predictions.get('meso')),
        'macro': bool(predictions.get('macro'))
    }

    # Build model outputs
    model_outputs = _build_model_outputs(user_id, display_subject, answers, predictions, training_needed)

    return jsonify({
        'success': True,
        'user_id': str(user_id),
        'predictions': predictions,
        'schedule_update_needed': bool(training_needed.get('needed', False)),
        'training_needed': training_needed,
        'sequence_status': sequence_status,
        'models_ready': models_ready,
        'model_outputs': model_outputs,
        'storage': 'mongodb',
        'timestamp': datetime.now().isoformat()
    }), 200


@retention_bp.route('/predictions/<user_id>/topic/<topic_id>', methods=['GET'])
@handle_errors
def get_topic_prediction(user_id, topic_id):
    """Get prediction for a specific topic from MongoDB."""
    validate_student_id(user_id)
    prediction = current_app.retention_prediction_service.get_topic_predictions(user_id, topic_id)

    return jsonify({
        'success': True,
        'user_id': user_id,
        'topic_id': topic_id,
        'prediction': prediction
    }), 200


@retention_bp.route('/predictions/<user_id>/subject/<subject>', methods=['GET'])
@handle_errors
def get_subject_predictions(user_id, subject):
    """Get predictions for a specific subject from MongoDB."""
    validate_student_id(user_id)
    predictions = current_app.retention_prediction_service.get_subject_predictions(user_id, subject)

    return jsonify({
        'success': True,
        'user_id': user_id,
        'subject': subject,
        'predictions': predictions
    }), 200


# ==================== SUMMARY ROUTES ====================

@retention_bp.route('/summary/<user_id>', methods=['GET'])
@handle_errors
def get_retention_summary(user_id):
    """Get retention summary for dashboard from MongoDB."""
    validate_student_id(user_id)
    subject = request.args.get('subject')
    summary = current_app.retention_prediction_service.get_retention_summary(user_id, subject)

    return jsonify({
        'success': True,
        'user_id': user_id,
        'subject': subject,
        'summary': summary
    }), 200


# ==================== FORGETTING CURVES ROUTES ====================

@retention_bp.route('/forgetting-curves/<user_id>', methods=['GET'])
@handle_errors
def get_forgetting_curves(user_id):
    """Get forgetting curves for topics from MongoDB."""
    validate_student_id(user_id)
    subject = request.args.get('subject')
    topic_id = request.args.get('topic_id')

    if topic_id:
        curve = current_app.retention_prediction_service.get_topic_forgetting_curve(user_id, topic_id)
        return jsonify({
            'success': True,
            'user_id': user_id,
            'topic_id': topic_id,
            'curve': curve
        }), 200
    else:
        curves = current_app.retention_prediction_service.get_all_forgetting_curves(user_id, subject)
        return jsonify({
            'success': True,
            'user_id': user_id,
            'subject': subject,
            'curves': curves
        }), 200


# ==================== BATCH RECOMMENDATIONS ROUTES ====================

@retention_bp.route('/batch-recommendations/<user_id>', methods=['GET'])
@handle_errors
def get_batch_recommendations(user_id):
    """Get batch recommendations for scheduling from MongoDB."""
    validate_student_id(user_id)
    batch_type = request.args.get('batch_type')
    subject = request.args.get('subject')

    recommendations = current_app.retention_prediction_service.get_batch_recommendations(
        user_id, batch_type, subject
    )

    return jsonify({
        'success': True,
        'user_id': user_id,
        'subject': subject,
        'recommendations': recommendations
    }), 200


# ==================== STRESS/FATIGUE ROUTES ====================

@retention_bp.route('/stress-fatigue/<user_id>', methods=['GET'])
@handle_errors
def get_stress_fatigue(user_id):
    """Get stress and fatigue predictions from MongoDB."""
    validate_student_id(user_id)
    predictions = current_app.retention_prediction_service.get_stress_fatigue_predictions(user_id)

    return jsonify({
        'success': True,
        'user_id': user_id,
        'stress_fatigue': predictions
    }), 200


# ==================== QUESTION SEQUENCE ROUTES ====================

@retention_bp.route('/question-sequence/<user_id>', methods=['GET'])
@handle_errors
def get_question_sequence(user_id):
    """Get question sequence for scheduling from MongoDB."""
    validate_student_id(user_id)
    subject = request.args.get('subject')
    batch_type = request.args.get('batch_type', 'immediate')
    count = request.args.get('count', 10, type=int)

    sequence = current_app.retention_prediction_service.get_question_sequence(
        user_id, subject, batch_type, count
    )

    return jsonify({
        'success': True,
        'user_id': user_id,
        'subject': subject,
        'batch_type': batch_type,
        'sequence': sequence,
        'count': len(sequence)
    }), 200


# ==================== BATCH COMPLETE ROUTE ====================

@retention_bp.route('/batch-complete/<user_id>', methods=['POST'])
@handle_errors
def batch_complete(user_id):
    """Handle batch completion notification with MongoDB storage."""
    validate_student_id(user_id)
    data = request.get_json()
    batch_type = data.get('batch_type')
    subject = data.get('subject')
    performance = data.get('performance', {})

    current_app.retention_prediction_service.update_after_batch(
        user_id, subject, batch_type, performance
    )

    new_schedule = current_app.schedule_service.generate_daily_schedule(
        user_id, subject
    )

    return jsonify({
        'success': True,
        'user_id': user_id,
        'batch_type': batch_type,
        'subject': subject,
        'new_schedule': new_schedule,
        'timestamp': datetime.now().isoformat()
    }), 200


# ==================== UPDATE AFTER INTERACTION ROUTE ====================

@retention_bp.route('/update-after-interaction', methods=['POST'])
@handle_errors
@log_request
def update_after_interaction():
    """Update retention after a learning interaction."""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        session_ctx = _RETENTION_SESSIONS.get(session_id, {})

        user_id = data.get('user_id') or data.get('student_id') or session_ctx.get('user_id')
        topic_id = data.get('topic_id') or data.get('concept_area') or 'unknown_topic'
        question_id = data.get('question_id')
        was_correct = data.get('correct', False)
        response_time = data.get('response_time_ms', 2000)
        stress_level = data.get('stress_level', 0.3)
        fatigue_level = data.get('fatigue_index', 0.3)

        if not user_id:
            return jsonify({'success': False, 'error': 'user_id/student_id is required'}), 400

        # Persist the interaction
        _persist_interactions_to_mongodb(
            str(user_id),
            session_id or f"interaction_{int(datetime.now().timestamp())}",
            session_ctx.get('subject', data.get('subject', 'english')),
            [{
                'question_id': question_id,
                'topic_id': topic_id,
                'correct': was_correct,
                'response_time_ms': response_time,
                'stress_level': stress_level,
                'fatigue_index': fatigue_level,
                'timestamp': datetime.now().isoformat(),
            }]
        )

        # Update schedule
        current_app.schedule_service.update_schedule_after_interaction(
            user_id, topic_id, was_correct
        )

        # Check if retraining needed
        services = _get_services()
        training_needed = services['training'].check_retrain_needed(user_id)

        # Get updated predictions for the topic
        updated_prediction = current_app.retention_prediction_service.get_topic_predictions(
            user_id, topic_id
        )

        return jsonify({
            'success': True,
            'user_id': user_id,
            'topic_id': topic_id,
            'question_id': question_id,
            'training_needed': training_needed,
            'updated_prediction': updated_prediction,
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        logger.error(f"Error updating after interaction: {str(e)}")
        return jsonify({'error': str(e)}), 500


# ==================== HELPER FUNCTIONS ====================

def _build_model_outputs(user_id, subject, answers, predictions, training_needed):
    """Build model outputs for the response."""
    micro_predictions = predictions.get('micro', []) or []
    meso_predictions = predictions.get('meso', []) or []
    macro_prediction = predictions.get('macro', {}) or {}

    latest_topic = None
    if answers:
        latest_topic = (
            answers[-1].get('topic_id')
            or answers[-1].get('concept_area')
            or answers[-1].get('topic')
        )

    latest_micro = None
    if latest_topic:
        latest_micro = next(
            (m for m in micro_predictions if str(m.get('topic_id')) == str(latest_topic)),
            None,
        )

    if not latest_micro and micro_predictions:
        latest_micro = micro_predictions[0]

    latest_current_ret = float(
        (latest_micro or {}).get('current_retention', (latest_micro or {}).get('retention_probability', 0.0) or 0.0)
    )
    latest_next_ret = float(
        (latest_micro or {}).get('next_retention', (latest_micro or {}).get('probability_correct_next', 0.0) or 0.0)
    )
    latest_stress = float((latest_micro or {}).get('stress_impact', 0.3) or 0.3)
    latest_fatigue = float((latest_micro or {}).get('fatigue_level', 0.3) or 0.3)

    return {
        'micro_lstm': {
            'output': {
                'topic_id': latest_micro.get('topic_id') if latest_micro else latest_topic,
                'retention_score': round(latest_current_ret, 2),
                'current_retention': round(latest_current_ret, 2),
                'next_retention': round(latest_next_ret, 2),
                'probability_correct_next_attempt': round(latest_next_ret, 2),
                'stress_impact': round(latest_stress, 2),
                'fatigue_prediction': round(latest_fatigue, 2),
            }
        },
        'meso_lstm': {
            'output': {
                'subject_retention_score': round(np.mean([float(m.get('retention_7d', 0.0)) for m in meso_predictions]) if meso_predictions else 0.0, 2),
                'subject_retention_7d': round(np.mean([float(m.get('retention_7d', 0.0)) for m in meso_predictions]) if meso_predictions else 0.0, 2),
            }
        },
        'macro_lstm': {
            'output': {
                'predicted_long_term_retention_score': round(macro_prediction.get('projected_retention', 0.0), 2),
                'fatigue_risk_probability': round(macro_prediction.get('burnout_risk', 0.0), 2),
            }
        }
    }

    # retention.py - Add these methods to ensure collections are populated

def _persist_interactions_to_mongodb(user_id: str, session_id: str, subject: str, responses: list):
    """
    Persist interactions to MongoDB and populate all relevant collections.
    """
    if not responses:
        return

    try:
        data_manager = current_app.retention_prediction_service._get_data_manager(user_id)

        # 1. Save interactions
        for response in responses:
            interaction = {
                'user_id': user_id,
                'session_id': session_id,
                'subject': normalize_subject(subject),
                'topic_id': response.get('topic_id') or response.get('concept_area') or response.get('topic') or 'unknown_topic',
                'question_id': response.get('question_id'),
                'correct': bool(response.get('correct', False)),
                'response_time_ms': float(response.get('time_spent', response.get('response_time_ms', 0)) or 0),
                'confidence': float(response.get('confidence', 0.5) or 0.5),
                'difficulty': float(response.get('difficulty', 0.5) or 0.5),
                'hesitation_count': int(response.get('hesitation_count', response.get('answer_changes', 0)) or 0),
                'fatigue_index': float(response.get('fatigue_index', 0.3) or 0.3),
                'focus_score': float(response.get('focus_score', 0.7) or 0.7),
                'stress_level': float(response.get('stress_level', 0.3) or 0.3),
                'attempt_number': int(response.get('attempt_number', 1) or 1),
                'streak': int(response.get('streak', 0) or 0),
                'timestamp': response.get('timestamp', datetime.now().isoformat())
            }
            data_manager.save_interaction(interaction)

        # 2. Process as features (populates practice_features)
        data_manager.append_practice_attempts_as_features(responses)

        # 3. Generate sequences (populates micro/meso/macro_sequences)
        data_manager._generate_micro_sequences_from_features()
        data_manager._generate_meso_sequences_from_features()
        data_manager._generate_macro_sequences_from_features()

        # 4. Update concept features
        concept_data = {}
        for response in responses:
            concept = response.get('topic_id') or response.get('concept_area') or response.get('topic') or 'general'
            if concept not in concept_data:
                concept_data[concept] = {
                    'accuracy': [],
                    'difficulty': [],
                    'confidence': [],
                    'time_spent': []
                }
            concept_data[concept]['accuracy'].append(1.0 if response.get('correct', False) else 0.0)
            concept_data[concept]['difficulty'].append(response.get('difficulty', 0.5))
            concept_data[concept]['confidence'].append(response.get('confidence', 0.5))
            concept_data[concept]['time_spent'].append(response.get('time_spent', 0))

        if concept_data:
            concept_features = {}
            for concept, values in concept_data.items():
                concept_features[concept] = {
                    'accuracy': float(np.mean(values['accuracy'])) if values['accuracy'] else 0.5,
                    'avg_difficulty': float(np.mean(values['difficulty'])) if values['difficulty'] else 0.5,
                    'avg_confidence': float(np.mean(values['confidence'])) if values['confidence'] else 0.5,
                    'avg_time_spent': float(np.mean(values['time_spent'])) if values['time_spent'] else 0,
                    'attempts': len(values['accuracy']),
                    'last_practiced': datetime.now().isoformat()
                }
            data_manager.save_concept_features(concept_features)

        # 5. Save daily aggregate
        daily_aggregate = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'total_attempts': len(responses),
            'correct_attempts': sum(1 for r in responses if r.get('correct', False)),
            'accuracy': sum(1 for r in responses if r.get('correct', False)) / max(1, len(responses)),
            'avg_time_spent': sum(r.get('time_spent', 0) for r in responses) / max(1, len(responses)),
            'unique_concepts': len(set(r.get('topic_id', r.get('concept_area', 'general')) for r in responses))
        }
        data_manager.save_daily_aggregate(daily_aggregate)

        # 6. Save session
        session_data = {
            'session_id': session_id,
            'started_at': responses[0].get('timestamp', datetime.now().isoformat()) if responses else datetime.now().isoformat(),
            'ended_at': datetime.now().isoformat(),
            'total_attempts': len(responses),
            'accuracy': sum(1 for r in responses if r.get('correct', False)) / max(1, len(responses)),
            'concepts': list(set(r.get('topic_id', r.get('concept_area', 'general')) for r in responses))
        }
        data_manager.save_session(session_data)

        # 7. Save predictions
        # Generate predictions from the responses
        predictions = []
        for response in responses:
            retention = float(response.get('confidence', 0.5))
            pred = {
                'topic_id': response.get('topic_id', response.get('concept_area', 'general')),
                'subject': normalize_subject(subject),
                'current_retention': retention,
                'next_retention': retention * 0.9,
                'stress_impact': float(response.get('stress_level', 0.3)),
                'fatigue_level': float(response.get('fatigue_index', 0.3)),
                'repeat_in_seconds': 300,
                'batch_type': 'medium_term'
            }
            predictions.append(pred)
        data_manager.save_predictions('micro', predictions)

        logger.info(f"✅ Persisted all data for user {user_id} - Interactions: {len(responses)}, Features: {len(responses)}, Sequences generated")

    except Exception as e:
        logger.error(f"Error persisting interactions to MongoDB: {e}")
        raise
    
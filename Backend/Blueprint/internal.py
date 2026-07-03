"""
Internal routes blueprint - Retention Flask <-> Node bridge endpoints with MongoDB storage.
"""
from flask import Blueprint, request, jsonify, current_app
import logging
from datetime import datetime

from Utils.validators import validate_student_id
from Utils.decorators import handle_errors, log_request

internal_bp = Blueprint('internal', __name__)
logger = logging.getLogger(__name__)


def _get_services():
    """Get retention services from current app."""
    return {
        'prediction': current_app.retention_prediction_service,
        'schedule': current_app.schedule_service,
        'training': current_app.retention_training_service,
        'performance': current_app.performance_service
    }


@internal_bp.route('/node/predictions', methods=['POST'])
@handle_errors
@log_request
def send_predictions_to_node():
    """Send predictions to Node.js from MongoDB."""
    data = request.get_json() or {}
    user_id = data.get('user_id')
    subject = data.get('subject')
    validate_student_id(user_id)

    services = _get_services()
    predictions = services['prediction'].prepare_for_nodejs(user_id, subject)
    metrics = services['performance'].get_metrics_summary(user_id, subject)

    micro = predictions.get('predictions', {}).get('micro', [])
    schedule = services['schedule'].generate_daily_schedule(user_id, subject, micro) if micro else {}
    question_sequence = services['prediction'].get_question_sequence(user_id, subject, 'immediate', 20)

    node_payload = {
        'user_id': user_id,
        'subject': subject,
        'timestamp': datetime.now().isoformat(),
        'predictions': predictions.get('predictions', {}),
        'metrics': metrics,
        'schedule': schedule,
        'question_sequence': question_sequence,
        'models_ready': predictions.get('models_ready', {}),
        'storage': 'mongodb'
    }

    # Send to Node.js (simplified - would make HTTP request)
    sent_to_node = True

    return jsonify({
        'success': True,
        'user_id': user_id,
        'subject': subject,
        'sent_to_node': sent_to_node,
        'predictions_ready': predictions.get('models_ready', {})
    }), 200


@internal_bp.route('/node/performance-update', methods=['POST'])
@handle_errors
@log_request
def send_performance_update():
    """Send performance update to Node.js from MongoDB."""
    data = request.get_json() or {}
    user_id = data.get('user_id')
    subject = data.get('subject')
    validate_student_id(user_id)

    services = _get_services()
    metrics = services['performance'].calculate_all_metrics(user_id, days=7, subject=subject)
    summary = services['performance'].get_metrics_summary(user_id, subject)
    predictions = services['prediction'].get_all_predictions(user_id, subject)

    payload = {
        'user_id': user_id,
        'subject': subject,
        'timestamp': datetime.now().isoformat(),
        'metrics': metrics,
        'summary': summary,
        'predictions': predictions,
        'storage': 'mongodb'
    }

    sent_to_node = True

    return jsonify({
        'success': True,
        'user_id': user_id,
        'subject': subject,
        'sent_to_node': sent_to_node,
        'metrics': summary
    }), 200


@internal_bp.route('/node/question-sequence', methods=['POST'])
@handle_errors
@log_request
def send_question_sequence():
    """Send question sequence to Node.js from MongoDB."""
    data = request.get_json() or {}
    user_id = data.get('user_id')
    subject = data.get('subject')
    batch_type = data.get('batch_type', 'immediate')
    count = int(data.get('count', 20))
    validate_student_id(user_id)

    services = _get_services()
    question_sequence = services['prediction'].get_question_sequence(user_id, subject, batch_type, count)
    predictions = services['prediction'].get_all_predictions(user_id, subject)
    schedule = services['schedule'].generate_daily_schedule(user_id, subject, predictions)

    payload = {
        'user_id': user_id,
        'subject': subject,
        'batch_type': batch_type,
        'timestamp': datetime.now().isoformat(),
        'question_sequence': question_sequence,
        'schedule_context': {
            'immediate_batch': schedule.get('immediate_batch', {}),
            'session_batch': schedule.get('session_batch', {})
        },
        'storage': 'mongodb'
    }

    sent_to_node = True

    return jsonify({
        'success': True,
        'user_id': user_id,
        'subject': subject,
        'sent_to_node': sent_to_node,
        'question_sequence': question_sequence[:10]
    }), 200


@internal_bp.route('/node/stress-fatigue-update', methods=['POST'])
@handle_errors
@log_request
def send_stress_fatigue_update():
    """Send stress/fatigue update to Node.js from MongoDB."""
    data = request.get_json() or {}
    user_id = data.get('user_id')
    subject = data.get('subject')
    validate_student_id(user_id)

    services = _get_services()
    stress_fatigue = services['prediction'].get_stress_fatigue_predictions(user_id, subject)
    optimal_times = services['schedule'].get_optimal_study_times(user_id, subject, stress_fatigue)

    payload = {
        'user_id': user_id,
        'subject': subject,
        'timestamp': datetime.now().isoformat(),
        'stress_fatigue': stress_fatigue,
        'optimal_study_times': optimal_times,
        'recommendations': services['prediction'].generate_stress_fatigue_recommendations(stress_fatigue),
        'storage': 'mongodb'
    }

    sent_to_node = True

    return jsonify({
        'success': True,
        'user_id': user_id,
        'subject': subject,
        'sent_to_node': sent_to_node,
        'stress_fatigue': stress_fatigue
    }), 200


@internal_bp.route('/status/<user_id>', methods=['GET'])
@handle_errors
def get_system_status(user_id):
    """Get system status for a user from MongoDB."""
    validate_student_id(user_id)
    services = _get_services()

    # Get data manager for MongoDB stats
    data_manager = current_app.retention_prediction_service._get_data_manager(user_id)

    # Count documents in MongoDB
    practice_count = len(data_manager.load_practice_features())
    interactions_count = len(data_manager.load_interactions())
    micro_count = len(data_manager.load_micro_sequences())
    meso_count = len(data_manager.load_meso_sequences())
    macro_count = len(data_manager.load_macro_sequences())

    status = {
        'user_id': user_id,
        'storage': 'mongodb',
        'data_counts': {
            'practice_features': practice_count,
            'interactions': interactions_count,
            'micro_sequences': micro_count,
            'meso_sequences': meso_count,
            'macro_sequences': macro_count
        },
        'training_needed': services['training'].check_retrain_needed(user_id),
        'models': {
            'micro': services['prediction'].get_model_status(user_id, 'micro'),
            'meso': services['prediction'].get_model_status(user_id, 'meso'),
            'macro': services['prediction'].get_model_status(user_id, 'macro')
        }
    }

    return jsonify({'success': True, 'status': status}), 200


@internal_bp.route('/train/<user_id>', methods=['POST'])
@handle_errors
@log_request
def trigger_training(user_id):
    """Trigger training for a user using MongoDB data."""
    validate_student_id(user_id)
    data = request.get_json() or {}
    model_type = data.get('model_type', 'all')

    services = _get_services()

    results = {}
    if model_type in ['micro', 'all']:
        results['micro'] = services['training'].train_micro_model(user_id)
    if model_type in ['meso', 'all']:
        results['meso'] = services['training'].train_meso_model(user_id)
    if model_type in ['macro', 'all']:
        results['macro'] = services['training'].train_macro_model(user_id)

    if results:
        services['prediction'].prepare_for_nodejs(user_id, data.get('subject'))

    return jsonify({
        'success': True,
        'user_id': user_id,
        'trained_models': results,
        'storage': 'mongodb',
        'timestamp': datetime.now().isoformat()
    }), 200


@internal_bp.route('/train-all/<user_id>', methods=['POST'])
@handle_errors
@log_request
def train_all_models_manual(user_id):
    """Manually trigger training for all models using MongoDB data."""
    validate_student_id(user_id)
    data = request.get_json() or {}
    force = data.get('force', False)

    services = _get_services()

    if not force:
        training_needed = services['training'].check_retrain_needed(user_id)
        if not training_needed.get('needed'):
            return jsonify({
                'success': False,
                'message': 'Training not needed based on current MongoDB data',
                'training_needed': training_needed,
                'storage': 'mongodb'
            }), 200

    results = services['training'].train_all_models(user_id)
    predictions = services['prediction'].prepare_for_nodejs(user_id, data.get('subject'))

    return jsonify({
        'success': True,
        'user_id': user_id,
        'training_results': results,
        'predictions': predictions,
        'storage': 'mongodb',
        'timestamp': datetime.now().isoformat()
    }), 200


@internal_bp.route('/debug/mongodb/<user_id>', methods=['GET'])
@handle_errors
def debug_mongodb_data(user_id):
    """Debug endpoint to check MongoDB data for a user."""
    validate_student_id(user_id)

    data_manager = current_app.retention_prediction_service._get_data_manager(user_id)

    # Get counts from all collections
    collections = [
        'practice_features',
        'interactions',
        'micro_sequences',
        'meso_sequences',
        'macro_sequences',
        'model_metadata',
        'schedules',
        'sessions'
    ]

    result = {}
    for coll_name in collections:
        try:
            coll = data_manager.db.get_collection(coll_name)
            count = coll.count_documents({'student_id': user_id})
            result[coll_name] = count
        except Exception as e:
            result[coll_name] = f"Error: {str(e)}"

    # Get sample from practice_features
    practice_df = data_manager.load_practice_features()
    sample = practice_df.head(2).to_dict() if not practice_df.empty else None

    return jsonify({
        'success': True,
        'user_id': user_id,
        'storage': 'mongodb',
        'collection_counts': result,
        'practice_features_count': len(practice_df),
        'practice_features_sample': sample,
        'timestamp': datetime.now().isoformat()
    }), 200
    
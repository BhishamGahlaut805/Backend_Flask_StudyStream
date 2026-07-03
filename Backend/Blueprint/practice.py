from flask import Blueprint, request, jsonify, current_app
import logging
from datetime import datetime
import traceback
import numpy as np
import uuid

from Utils.validators import validate_student_id, validate_features
from Utils.helpers import difficulty_level_from_value, extract_last_training_info
from Utils.decorators import log_request, handle_errors

practice_bp = Blueprint('practice', __name__)
logger = logging.getLogger(__name__)


@practice_bp.route('/profile/<student_id>', methods=['GET'])
@handle_errors
@log_request
def get_practice_profile(student_id):
    """
    Return comprehensive practice profile with training status.
    This endpoint returns the actual stored data count and training status.
    """
    validate_student_id(student_id)

    # Get data manager for this student
    data_manager = current_app.prediction_service._get_data_manager(student_id)

    # ==================== FIX: Load actual stored practice features ====================
    practice_df = data_manager.load_practice_features()
    actual_feature_rows = len(practice_df)

    logger.info(f"[Profile] Student {student_id} has {actual_feature_rows} practice feature rows")
    

    # Get current difficulty
    current_difficulty = 0.5
    if not practice_df.empty and 'current_question_difficulty' in practice_df.columns:
        try:
            current_difficulty = float(practice_df['current_question_difficulty'].iloc[-1])
        except Exception:
            current_difficulty = 0.5

    # ==================== FIX: Load training metadata properly ====================
    metadata_history = data_manager.load_model_metadata('practice_difficulty')

    # Extract last training info
    last_trained_rows = None
    last_trained_at = None

    if metadata_history and len(metadata_history) > 0:
        # Get the most recent training record
        latest = metadata_history[0]  # Already sorted by timestamp descending
        last_trained_rows = latest.get('feature_rows_at_training')
        last_trained_at = latest.get('timestamp')

        logger.info(f"[Profile] Student {student_id} last trained at {last_trained_at} with {last_trained_rows} rows")
    else:
        logger.info(f"[Profile] Student {student_id} has no training history yet")
    # ==================== END OF FIX ====================

    # Get configuration values
    min_samples = int(current_app.config.get('MIN_PRACTICE_SAMPLES', 100))
    retrain_interval = int(current_app.config.get('PRACTICE_RETRAIN_INTERVAL', 100))

    # ==================== FIX: Calculate rows to next training ====================
    if last_trained_rows is None:
        # No training yet - need to reach min_samples
        rows_to_next = max(0, min_samples - actual_feature_rows)
    else:
        # Training exists - need retrain_interval new rows
        rows_to_next = max(0, retrain_interval - (actual_feature_rows - last_trained_rows))
    # ==================== END OF FIX ====================

    # Determine if training is ready
    training_ready = actual_feature_rows >= min_samples and rows_to_next == 0

    response_data = {
        'success': True,
        'student_id': student_id,
        'current_difficulty': round(max(0.0, min(1.0, current_difficulty)), 2),
        'current_difficulty_level': difficulty_level_from_value(current_difficulty),
        'feature_rows': int(actual_feature_rows),
        'model_trained': len(metadata_history) > 0,
        'last_trained_feature_rows': last_trained_rows,
        'last_trained_at': last_trained_at,
        'min_samples_required': min_samples,
        'retrain_interval': retrain_interval,
        'rows_to_next_training': rows_to_next,
        'entries_left_for_retraining': rows_to_next,  # For frontend compatibility
        'training_ready': training_ready,
        'training_history_count': len(metadata_history),
        'timestamp': datetime.now().isoformat()
    }

    logger.info(f"[Profile] Response for {student_id}: feature_rows={actual_feature_rows}, "
                f"last_trained_rows={last_trained_rows}, rows_to_next={rows_to_next}")

    return jsonify(response_data)


@practice_bp.route('/reset-data', methods=['POST'])
@handle_errors
@log_request
def reset_practice_data():
    """Clear stored practice data for a fresh start."""
    data = request.get_json() or {}
    student_id = data.get('student_id')
    validate_student_id(student_id)

    data_manager = current_app.prediction_service._get_data_manager(student_id)
    current_app.training_service.cancel_practice_training(student_id)
    reset_result = data_manager.reset_practice_data()
    current_app.prediction_service.clear_student_cache(student_id)

    return jsonify({
        'success': True,
        'message': 'Practice history cleared successfully',
        'student_id': student_id,
        'cleared_files': reset_result.get('cleared_files', []),
        'timestamp': datetime.now().isoformat()
    })


@practice_bp.route('/next-difficulty', methods=['POST'])
@handle_errors
@log_request
def predict_next_difficulty():
    """Predict next difficulty for practice mode."""
    data = request.get_json()

    if not data or 'student_id' not in data or 'features' not in data:
        return jsonify({'success': False, 'error': 'Missing student_id or features'}), 400

    student_id = data['student_id']
    features = data['features']
    validate_student_id(student_id)
    validate_features(features, 12)

    # Convert and clip features
    features = [float(np.clip(f, 0.0, 1.0)) for f in features]

    # Get prediction
    prediction = current_app.prediction_service.predict_practice_difficulty(
        student_id, features
    )

    data_manager = current_app.prediction_service._get_data_manager(student_id)
    practice_df = data_manager.load_practice_features()
    metadata_history = data_manager.load_model_metadata('practice_difficulty')

    min_samples = int(current_app.config.get('MIN_PRACTICE_SAMPLES', 100))
    retrain_interval = int(current_app.config.get('PRACTICE_RETRAIN_INTERVAL', 100))
    feature_rows = int(len(practice_df))

    # Extract last training info
    last_trained_rows = None
    last_trained_at = None

    if metadata_history and len(metadata_history) > 0:
        latest = metadata_history[0]
        last_trained_rows = latest.get('feature_rows_at_training')
        last_trained_at = latest.get('timestamp')

    rows_to_next = max(0, retrain_interval - (feature_rows - (last_trained_rows or 0)))

    predicted_difficulty = float(prediction['predicted_difficulty'])

    return jsonify({
        'success': True,
        'next_difficulty': predicted_difficulty,
        'difficulty_level': difficulty_level_from_value(predicted_difficulty),
        'smoothed_difficulty': prediction['smoothed_difficulty'],
        'confidence': prediction['confidence'],
        'method': prediction['method'],
        'model_trained': bool(len(metadata_history) > 0),
        'feature_rows': feature_rows,
        'last_trained_feature_rows': last_trained_rows,
        'last_trained_at': last_trained_at,
        'min_samples_required': min_samples,
        'retrain_interval': retrain_interval,
        'rows_to_next_training': rows_to_next,
        'training_ready': feature_rows >= min_samples and rows_to_next == 0,
        'timestamp': datetime.now().isoformat()
    })


@practice_bp.route('/session-end', methods=['POST'])
@handle_errors
@log_request
def end_practice_session():
    """
    Session end handler that stores processed practice features in MongoDB.
    Each attempt is stored as a separate document with full feature vector.
    """
    data = request.get_json()

    if not data or 'student_id' not in data or 'attempts' not in data:
        return jsonify({'success': False, 'error': 'Missing data'}), 400

    student_id = data['student_id']
    attempts = data['attempts']
    session_id = data.get('session_id')
    finalize_session = bool(data.get('finalize_session', False))

    if not isinstance(attempts, list):
        return jsonify({'success': False, 'error': 'attempts must be a list'}), 400

    # Generate session_id if not provided
    if not session_id:
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    # Add session_id to each attempt
    for attempt in attempts:
        attempt['session_id'] = session_id

    data_manager = current_app.prediction_service._get_data_manager(student_id)

    # Get count before saving
    practice_rows_before = len(data_manager.load_practice_features())
    logger.info(f"[Session-End] Before saving: {practice_rows_before} rows")

    # Process each attempt
    feature_count = 0
    concept_data = {}

    for attempt in attempts:
        # Ensure required fields exist
        if 'timestamp' not in attempt:
            attempt['timestamp'] = datetime.now().isoformat()

        # 1. Store raw interaction
        data_manager.save_interaction(attempt)

        # 2. Process and store as features
        result = data_manager.append_practice_attempts_as_features([attempt])
        feature_count += result.get('added_rows', 0)

        # 3. Update concept features
        concept = attempt.get('concept', attempt.get('concept_area', 'general'))
        if concept not in concept_data:
            concept_data[concept] = {
                'accuracy': [],
                'difficulty': [],
                'confidence': [],
                'time_spent': []
            }
        concept_data[concept]['accuracy'].append(1.0 if attempt.get('correct', False) else 0.0)
        concept_data[concept]['difficulty'].append(attempt.get('difficulty', 0.5))
        concept_data[concept]['confidence'].append(attempt.get('confidence', 0.5))
        concept_data[concept]['time_spent'].append(attempt.get('time_spent', 0))

    # 4. Save concept features
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

    # 5. Get count after saving
    total_feature_rows = len(data_manager.load_practice_features())
    logger.info(f"[Session-End] After saving: {total_feature_rows} rows (added {feature_count})")

    # 6. Save daily aggregate
    daily_aggregate = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'total_attempts': len(attempts),
        'correct_attempts': sum(1 for a in attempts if a.get('correct', False)),
        'accuracy': sum(1 for a in attempts if a.get('correct', False)) / max(1, len(attempts)),
        'avg_time_spent': sum(a.get('time_spent', 0) for a in attempts) / max(1, len(attempts)),
        'unique_concepts': len(set(a.get('concept', a.get('concept_area', 'general')) for a in attempts))
    }
    data_manager.save_daily_aggregate(daily_aggregate)

    # 7. Save session
    session_data = {
        'session_id': session_id,
        'started_at': attempts[0].get('timestamp', datetime.now().isoformat()) if attempts else datetime.now().isoformat(),
        'ended_at': datetime.now().isoformat(),
        'total_attempts': len(attempts),
        'accuracy': sum(1 for a in attempts if a.get('correct', False)) / max(1, len(attempts)),
        'concepts': list(set(a.get('concept', a.get('concept_area', 'general')) for a in attempts))
    }
    data_manager.save_session(session_data)

    training_triggered = False
    global_triggered = False

    # Training Logic
    min_samples = current_app.config.get('MIN_PRACTICE_SAMPLES', 100)
    retrain_interval = current_app.config.get('PRACTICE_RETRAIN_INTERVAL', 100)

    metadata_history = data_manager.load_model_metadata('practice_difficulty')
    last_trained_rows = None
    last_trained_at = None

    if metadata_history and len(metadata_history) > 0:
        latest = metadata_history[0]
        last_trained_rows = latest.get('feature_rows_at_training')
        last_trained_at = latest.get('timestamp')

    should_trigger_training = False

    if finalize_session and total_feature_rows >= min_samples:
        if last_trained_rows is None:
            should_trigger_training = True
            logger.info(f"[Session-End] First training trigger: {total_feature_rows} rows >= {min_samples}")
        else:
            new_rows_since_training = total_feature_rows - last_trained_rows
            if new_rows_since_training >= retrain_interval:
                should_trigger_training = True
                logger.info(f"[Session-End] Retraining trigger: {new_rows_since_training} new rows >= {retrain_interval}")

    if should_trigger_training:
        logger.info(f"[Session-End] Triggering training for student {student_id}")
        training_triggered = current_app.training_service.train_practice_model_async(student_id)
        if training_triggered:
            last_trained_at = datetime.now().isoformat()

    # Trigger global feature pipeline
    min_global_samples = current_app.config.get('MIN_PRACTICE_SAMPLES_FOR_GLOBAL', 40)
    if practice_rows_before < min_global_samples <= total_feature_rows:
        logger.info(f"[Session-End] Triggering global features for student {student_id}")
        global_triggered = current_app.training_service.generate_global_features(student_id)

    if last_trained_rows is None:
        rows_to_next = max(0, min_samples - total_feature_rows)
    else:
        rows_to_next = max(0, retrain_interval - (total_feature_rows - last_trained_rows))

    # Save model metadata for training tracking
    if training_triggered:
        model_metadata = {
            'model_name': 'practice_difficulty',
            'feature_rows_at_training': total_feature_rows,
            'last_trained_at': datetime.now().isoformat()
        }
        data_manager.save_model_metadata('practice_difficulty', model_metadata)

    return jsonify({
        'success': True,
        'message': f'Session ended, {len(attempts)} attempts processed',
        'total_attempts': len(attempts),
        'feature_rows': feature_count,
        'total_feature_rows': total_feature_rows,
        'last_trained_feature_rows': last_trained_rows,
        'last_trained_at': last_trained_at,
        'retrain_interval': retrain_interval,
        'min_samples_required': min_samples,
        'rows_to_next_training': rows_to_next,
        'training_ready': total_feature_rows >= min_samples and rows_to_next == 0,
        'finalize_session': finalize_session,
        'training_triggered': training_triggered,
        'global_features_triggered': global_triggered,
        'session_id': session_id,
        'timestamp': datetime.now().isoformat()
    })

@practice_bp.route('/session-save', methods=['POST'])
@handle_errors
@log_request
def save_practice_session():
    """
    Save practice session data incrementally without ending the session.
    This allows saving data between questions while keeping the session active.
    No training is triggered here - only data storage.
    """
    data = request.get_json()

    if not data or 'student_id' not in data or 'attempt' not in data:
        return jsonify({'success': False, 'error': 'Missing data'}), 400

    student_id = data['student_id']
    attempt = data['attempt']
    session_id = data.get('session_id')

    if not isinstance(attempt, dict):
        return jsonify({'success': False, 'error': 'attempt must be a dict'}), 400

    # Add session_id if provided
    if session_id:
        attempt['session_id'] = session_id

    # Ensure timestamp exists
    if 'timestamp' not in attempt:
        attempt['timestamp'] = datetime.now().isoformat()

    # Save the attempt
    data_manager = current_app.prediction_service._get_data_manager(student_id)

    # Store interaction
    data_manager.save_interaction(attempt)

    # Process and store as features
    result = data_manager.append_practice_attempts_as_features([attempt])

    total_feature_rows = len(data_manager.load_practice_features())

    return jsonify({
        'success': True,
        'message': 'Practice attempt saved successfully',
        'student_id': student_id,
        'feature_rows_added': result.get('added_rows', 0),
        'total_feature_rows': total_feature_rows,
        'timestamp': datetime.now().isoformat()
    })


@practice_bp.route('/training-status/<student_id>', methods=['GET'])
@handle_errors
@log_request
def get_training_status(student_id):
    """
    Get detailed training status for a student.
    Useful for debugging and monitoring training progress.
    """
    validate_student_id(student_id)

    data_manager = current_app.prediction_service._get_data_manager(student_id)
    practice_df = data_manager.load_practice_features()
    metadata_history = data_manager.load_model_metadata('practice_difficulty')

    total_rows = len(practice_df)
    min_samples = int(current_app.config.get('MIN_PRACTICE_SAMPLES', 100))
    retrain_interval = int(current_app.config.get('PRACTICE_RETRAIN_INTERVAL', 100))

    # Extract last training info
    last_trained_rows = None
    last_trained_at = None

    if metadata_history and len(metadata_history) > 0:
        latest = metadata_history[0]
        last_trained_rows = latest.get('feature_rows_at_training')
        last_trained_at = latest.get('timestamp')

    if last_trained_rows is None:
        rows_to_next_training = max(0, min_samples - total_rows)
        training_status = 'collecting_data'
        message = f'Collecting initial data. Need {rows_to_next_training} more rows to reach {min_samples}'
        training_progress = min(100, int((total_rows / min_samples) * 100))
    else:
        rows_to_next_training = max(0, retrain_interval - (total_rows - last_trained_rows))
        if rows_to_next_training == 0:
            training_status = 'ready_for_retraining'
            message = 'Ready for retraining on next session end'
            training_progress = 100
        else:
            training_status = 'trained'
            message = f'Model trained. {rows_to_next_training} more rows needed for retraining'
            training_progress = min(100, int(((total_rows - last_trained_rows) / retrain_interval) * 100))

    return jsonify({
        'success': True,
        'student_id': student_id,
        'total_rows': total_rows,
        'min_samples_required': min_samples,
        'retrain_interval': retrain_interval,
        'last_trained_rows': last_trained_rows,
        'last_trained_at': last_trained_at,
        'rows_to_next_training': rows_to_next_training,
        'training_status': training_status,
        'training_progress': training_progress,
        'message': message,
        'model_exists': len(metadata_history) > 0,
        'training_history_count': len(metadata_history),
        'training_history': metadata_history[:5] if metadata_history else [],
        'timestamp': datetime.now().isoformat()
    })


    # practice.py - Add concept features endpoint

@practice_bp.route('/concept/<student_id>/<concept>', methods=['GET'])
@handle_errors
@log_request
def get_concept_features(student_id, concept):
    """Get concept features for a specific concept."""
    validate_student_id(student_id)

    data_manager = current_app.prediction_service._get_data_manager(student_id)
    concept_features = data_manager.load_concept_features()

    if concept in concept_features:
        feat = concept_features[concept]
        return jsonify({
            'success': True,
            'concept': concept,
            'accuracy': feat.get('accuracy', 0.5),
            'attempts': feat.get('attempts', 0),
            'mastery_history': feat.get('concept_mastery_history', []),
            'last_practiced': feat.get('last_practiced'),
            'avg_difficulty': feat.get('avg_difficulty', 0.5),
            'avg_confidence': feat.get('avg_confidence', 0.5),
            'avg_time_spent': feat.get('avg_time_spent', 0)
        })

    return jsonify({
        'success': True,
        'concept': concept,
        'accuracy': 0.5,
        'attempts': 0,
        'mastery_history': [],
        'last_practiced': None
    })


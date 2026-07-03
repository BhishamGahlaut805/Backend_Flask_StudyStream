from flask import Blueprint, request, jsonify, current_app
import logging
from datetime import datetime

from Utils.validators import validate_student_id
from Utils.decorators import handle_errors, log_request

real_exam_bp = Blueprint('real_exam', __name__)
logger = logging.getLogger(__name__)


@real_exam_bp.route('/difficulty', methods=['POST'])
@handle_errors
@log_request
def predict_exam_difficulty():
    """Predict recommended difficulty for entire exam."""
    data = request.get_json()

    if not data or 'student_id' not in data:
        return jsonify({'success': False, 'error': 'Missing student_id'}), 400

    student_id = data['student_id']
    features = data.get('features', [])
    exam_type = data.get('exam_type', 'standard')
    validate_student_id(student_id)

    # Get prediction
    prediction = current_app.prediction_service.predict_exam_difficulty(
        student_id, features
    )

    # Get additional insights
    data_manager = current_app.prediction_service._get_data_manager(student_id)
    practice_df = data_manager.load_practice_features()

    insights = {
        'strong_concepts': [],
        'weak_concepts': [],
        'estimated_score': round(float(prediction.get('recommended_difficulty', 0.5)) * 0.7 + 0.2, 2)
    }

    if not practice_df.empty and {'concept', 'accuracy'}.issubset(set(practice_df.columns)):
        concept_performance = practice_df.groupby('concept')['accuracy'].mean().to_dict()
        top_concepts = sorted(concept_performance.items(), key=lambda x: x[1], reverse=True)[:3]
        weak_concepts = sorted(concept_performance.items(), key=lambda x: x[1])[:3]

        insights['strong_concepts'] = [
            {'name': c, 'accuracy': round(float(a), 2)} for c, a in top_concepts
        ]
        insights['weak_concepts'] = [
            {'name': c, 'accuracy': round(float(a), 2)} for c, a in weak_concepts
        ]

    return jsonify({
        'success': True,
        'recommended_difficulty': prediction['recommended_difficulty'],
        'difficulty_level': prediction['difficulty_level'],
        'confidence': prediction['confidence'],
        'method': prediction['method'],
        'insights': insights,
        'timestamp': datetime.now().isoformat()
    })


@real_exam_bp.route('/submit', methods=['POST'])
@handle_errors
@log_request
def submit_exam_results():
    """Submit completed exam results."""
    data = request.get_json()

    if not data or 'student_id' not in data or 'exam_data' not in data:
        return jsonify({'success': False, 'error': 'Missing data'}), 400

    student_id = data['student_id']
    exam_data = data['exam_data']
    validate_student_id(student_id)

    if not isinstance(exam_data, dict):
        return jsonify({'success': False, 'error': 'exam_data must be an object'}), 400

    # Add timestamp
    exam_data['timestamp'] = datetime.now().isoformat()

    # Save exam record
    data_manager = current_app.prediction_service._get_data_manager(student_id)
    data_manager.save_exam_records([exam_data])

    # Compute exam features
    from Service.feature_engineering import FeatureEngineeringService

    all_exams = data_manager.load_exam_records()
    exam_df = FeatureEngineeringService.compute_exam_features(all_exams)

    training_triggered = False
    min_exam_samples = current_app.config.get('MIN_EXAM_SAMPLES', 5)
    retrain_interval = current_app.config.get('MODEL_RETRAIN_INTERVAL_ROWS', 100)

    if not exam_df.empty:
        data_manager.save_exam_features(exam_df)

        if len(exam_df) >= min_exam_samples:
            metadata_history = data_manager.load_model_metadata('exam_difficulty')
            last_trained_rows = None
            if metadata_history:
                for item in reversed(metadata_history):
                    if isinstance(item, dict) and item.get('feature_rows_at_training') is not None:
                        try:
                            last_trained_rows = int(item.get('feature_rows_at_training'))
                            break
                        except Exception:
                            continue

            if last_trained_rows is None or (len(exam_df) - last_trained_rows) >= retrain_interval:
                current_app.training_service.train_exam_model_async(student_id)
                training_triggered = True

    analysis = {
        'score': exam_data.get('score', 0),
        'accuracy': exam_data.get('accuracy', 0),
        'time_taken': exam_data.get('time_taken', 0),
        'concept_performance': exam_data.get('concept_performance', {})
    }

    return jsonify({
        'success': True,
        'message': 'Exam results saved',
        'analysis': analysis,
        'exam_feature_rows': int(len(exam_df)) if not exam_df.empty else 0,
        'training_triggered': training_triggered,
        'min_samples_required': int(min_exam_samples),
        'retrain_interval': int(retrain_interval)
    })
    
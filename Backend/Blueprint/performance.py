"""
Performance blueprint - Handles performance metrics API endpoints
Updated for MongoDB with StudentDataManager integration.
"""
from flask import Blueprint, request, jsonify, current_app
import logging
from datetime import datetime

from Utils.decorators import handle_errors, log_request
from Utils.validators import validate_student_id

performance_bp = Blueprint('performance', __name__)
logger = logging.getLogger(__name__)


def _performance_service():
    """Get performance service from current app."""
    return getattr(current_app, 'retention_performance_service', getattr(current_app, 'performance_service', None))


def _prediction_service():
    """Get prediction service from current app."""
    return getattr(current_app, 'retention_prediction_service', getattr(current_app, 'prediction_service', None))


@performance_bp.route('/metrics/<user_id>', methods=['GET'])
@handle_errors
@log_request
def get_performance_metrics(user_id):
    """
    Get all performance metrics for a user.

    Query Parameters:
        days: Number of days to look back (default: 30)
        subject: Filter by subject (optional)

    Returns:
        JSON with performance metrics
    """
    validate_student_id(user_id)

    days = request.args.get('days', 30, type=int)
    subject = request.args.get('subject')

    service = _performance_service()
    if service is None:
        return jsonify({'error': 'Performance service not available'}), 503

    metrics = service.calculate_all_metrics(user_id, days, subject)

    return jsonify({
        'success': True,
        'user_id': user_id,
        'subject': subject,
        'metrics': metrics,
        'period_days': days,
        'timestamp': datetime.now().isoformat()
    }), 200


@performance_bp.route('/summary/<user_id>', methods=['GET'])
@handle_errors
@log_request
def get_performance_summary(user_id):
    """
    Get performance summary for dashboard.

    Query Parameters:
        subject: Filter by subject (optional)

    Returns:
        JSON with performance summary
    """
    validate_student_id(user_id)

    subject = request.args.get('subject')

    service = _performance_service()
    if service is None:
        return jsonify({'error': 'Performance service not available'}), 503

    summary = service.get_metrics_summary(user_id, subject)

    return jsonify({
        'success': True,
        'user_id': user_id,
        'subject': subject,
        'summary': summary,
        'timestamp': datetime.now().isoformat()
    }), 200


@performance_bp.route('/stress-patterns/<user_id>', methods=['GET'])
@handle_errors
@log_request
def get_stress_patterns(user_id):
    """
    Get detailed stress pattern analysis.

    Query Parameters:
        subject: Filter by subject (optional)

    Returns:
        JSON with stress pattern analysis
    """
    validate_student_id(user_id)

    subject = request.args.get('subject')

    service = _performance_service()
    if service is None:
        return jsonify({'error': 'Performance service not available'}), 503

    metrics = service.calculate_all_metrics(user_id, 30, subject)
    stress = metrics.get('stress_pattern', {})

    return jsonify({
        'success': True,
        'user_id': user_id,
        'subject': subject,
        'stress_pattern': stress,
        'timestamp': datetime.now().isoformat()
    }), 200


@performance_bp.route('/fatigue-patterns/<user_id>', methods=['GET'])
@handle_errors
@log_request
def get_fatigue_patterns(user_id):
    """
    Get detailed fatigue pattern analysis.

    Query Parameters:
        subject: Filter by subject (optional)

    Returns:
        JSON with fatigue pattern analysis
    """
    validate_student_id(user_id)

    subject = request.args.get('subject')

    service = _performance_service()
    if service is None:
        return jsonify({'error': 'Performance service not available'}), 503

    metrics = service.calculate_all_metrics(user_id, 30, subject)
    fatigue = metrics.get('fatigue_index', {})

    return jsonify({
        'success': True,
        'user_id': user_id,
        'subject': subject,
        'fatigue_pattern': fatigue,
        'timestamp': datetime.now().isoformat()
    }), 200


@performance_bp.route('/learning-efficiency/<user_id>', methods=['GET'])
@handle_errors
@log_request
def get_learning_efficiency(user_id):
    """
    Get learning efficiency metrics.

    Query Parameters:
        subject: Filter by subject (optional)

    Returns:
        JSON with learning efficiency metrics
    """
    validate_student_id(user_id)

    subject = request.args.get('subject')

    service = _performance_service()
    if service is None:
        return jsonify({'error': 'Performance service not available'}), 503

    metrics = service.calculate_all_metrics(user_id, 30, subject)
    efficiency = metrics.get('learning_efficiency', {})

    return jsonify({
        'success': True,
        'user_id': user_id,
        'subject': subject,
        'learning_efficiency': efficiency,
        'timestamp': datetime.now().isoformat()
    }), 200


@performance_bp.route('/historical/<user_id>', methods=['GET'])
@handle_errors
@log_request
def get_historical_metrics(user_id):
    """
    Get historical performance metrics over time.

    Query Parameters:
        subject: Filter by subject (optional)
        days: Number of days to look back (default: 30)

    Returns:
        JSON with historical metrics
    """
    validate_student_id(user_id)

    subject = request.args.get('subject')
    days = request.args.get('days', 30, type=int)

    service = _performance_service()
    if service is None:
        return jsonify({'error': 'Performance service not available'}), 503

    historical = service.get_historical_metrics(user_id, subject, days)

    return jsonify({
        'success': True,
        'user_id': user_id,
        'subject': subject,
        'historical': historical,
        'period_days': days,
        'timestamp': datetime.now().isoformat()
    }), 200


@performance_bp.route('/subject-comparison/<user_id>', methods=['GET'])
@handle_errors
@log_request
def get_subject_comparison(user_id):
    """
    Get performance comparison across subjects.

    Returns:
        JSON with subject comparison
    """
    validate_student_id(user_id)

    service = _performance_service()
    if service is None:
        return jsonify({'error': 'Performance service not available'}), 503

    comparison = service.get_subject_comparison(user_id)

    return jsonify({
        'success': True,
        'user_id': user_id,
        'comparison': comparison,
        'timestamp': datetime.now().isoformat()
    }), 200


@performance_bp.route('/all/<user_id>', methods=['GET'])
@handle_errors
@log_request
def get_all_performance_data(user_id):
    """
    Get all performance data in one request.

    Query Parameters:
        subject: Filter by subject (optional)
        days: Number of days to look back (default: 30)

    Returns:
        JSON with all performance data
    """
    validate_student_id(user_id)

    subject = request.args.get('subject')
    days = request.args.get('days', 30, type=int)

    service = _performance_service()
    if service is None:
        return jsonify({'error': 'Performance service not available'}), 503

    # Get all metrics
    metrics = service.calculate_all_metrics(user_id, days, subject)
    summary = service.get_metrics_summary(user_id, subject)
    historical = service.get_historical_metrics(user_id, subject, days)
    comparison = service.get_subject_comparison(user_id)

    return jsonify({
        'success': True,
        'user_id': user_id,
        'subject': subject,
        'period_days': days,
        'metrics': metrics,
        'summary': summary,
        'historical': historical,
        'subject_comparison': comparison,
        'timestamp': datetime.now().isoformat()
    }), 200


@performance_bp.route('/refresh/<user_id>', methods=['POST'])
@handle_errors
@log_request
def refresh_performance_data(user_id):
    """
    Force refresh performance data for a user.

    Returns:
        JSON with refresh status
    """
    validate_student_id(user_id)

    service = _performance_service()
    if service is None:
        return jsonify({'error': 'Performance service not available'}), 503

    # Force recalculate metrics
    metrics = service.calculate_all_metrics(user_id, 30, None, force_refresh=True)

    return jsonify({
        'success': True,
        'user_id': user_id,
        'message': 'Performance data refreshed successfully',
        'metrics': metrics,
        'timestamp': datetime.now().isoformat()
    }), 200


@performance_bp.route('/export/<user_id>', methods=['GET'])
@handle_errors
@log_request
def export_performance_data(user_id):
    """
    Export performance data as CSV.

    Query Parameters:
        subject: Filter by subject (optional)
        days: Number of days to look back (default: 30)

    Returns:
        CSV file with performance data
    """
    validate_student_id(user_id)

    subject = request.args.get('subject')
    days = request.args.get('days', 30, type=int)

    service = _performance_service()
    if service is None:
        return jsonify({'error': 'Performance service not available'}), 503

    historical = service.get_historical_metrics(user_id, subject, days)

    if not historical:
        return jsonify({'error': 'No performance data available for export'}), 404

    # Convert to CSV
    import pandas as pd
    df = pd.DataFrame(historical)

    # Create CSV response
    from flask import Response
    csv_data = df.to_csv(index=False)

    return Response(
        csv_data,
        mimetype='text/csv',
        headers={
            'Content-Disposition': f'attachment; filename=performance_{user_id}_{datetime.now().strftime("%Y%m%d")}.csv'
        }
    ), 200
    
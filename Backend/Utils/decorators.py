# Utils/decorators.py - Fix the log_request decorator

"""Decorators for request handling and error management."""

import functools
import logging
import time
from flask import request, jsonify

logger = logging.getLogger(__name__)


def handle_errors(func):
    """Decorator to handle exceptions in route handlers."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            logger.warning(f"Validation error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 400
        except Exception as e:
            logger.error(f"Unexpected error in {func.__name__}: {e}", exc_info=True)
            return jsonify({'success': False, 'error': 'Internal server error'}), 500
    return wrapper


def log_request(func):
    """Decorator to log incoming requests."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()

        # Log request
        log_data = {
            'endpoint': func.__name__,
            'method': request.method,
            'path': request.path,
            'remote_addr': request.remote_addr
        }

        # ==================== FIX: Only access request.json if content-type is application/json ====================
        if request.method in ['POST', 'PUT', 'PATCH'] and request.is_json:
            try:
                # Don't log sensitive data
                safe_data = {k: v for k, v in request.json.items()
                            if k not in ['password', 'token', 'secret']}
                log_data['data'] = safe_data
            except Exception:
                pass
        # ==================== END OF FIX ====================

        logger.info(f"Request: {log_data}")

        # Execute function
        response = func(*args, **kwargs)

        # Log response time
        duration = (time.time() - start_time) * 1000
        if isinstance(response, tuple):
            response_data = response[0]
            status_code = response[1] if len(response) > 1 else 200
        else:
            response_data = response
            status_code = 200

        logger.info(f"Response: {func.__name__} - {status_code} - {duration:.2f}ms")

        return response
    return wrapper


def require_student_id(func):
    """Decorator to validate student_id in request."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        student_id = kwargs.get('student_id')
        if not student_id:
            # ==================== FIX: Handle both GET and POST requests ====================
            if request.method in ['POST', 'PUT', 'PATCH'] and request.is_json:
                data = request.get_json() or {}
                student_id = data.get('student_id') or data.get('user_id')
            else:
                # For GET requests, check query parameters
                student_id = request.args.get('student_id') or request.args.get('user_id')
            # ==================== END OF FIX ====================

        if not student_id:
            return jsonify({'success': False, 'error': 'student_id is required'}), 400

        kwargs['student_id'] = student_id
        return func(*args, **kwargs)
    return wrapper


def require_subject(func):
    """Decorator to validate subject in request."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        subject = kwargs.get('subject')
        if not subject:
            # ==================== FIX: Handle both GET and POST requests ====================
            if request.method in ['POST', 'PUT', 'PATCH'] and request.is_json:
                data = request.get_json() or {}
                subject = data.get('subject')
            else:
                # For GET requests, check query parameters
                subject = request.args.get('subject')
            # ==================== END OF FIX ====================

        if not subject:
            return jsonify({'success': False, 'error': 'subject is required'}), 400

        kwargs['subject'] = subject
        return func(*args, **kwargs)
    return wrapper


def cache_response(ttl_seconds: int = 60):
    """Decorator to cache responses."""
    cache = {}

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            key = f"{func.__name__}:{args}:{kwargs}"
            current_time = time.time()

            # Check cache
            if key in cache:
                cached_data, cached_time = cache[key]
                if current_time - cached_time < ttl_seconds:
                    return cached_data

            # Execute function and cache
            result = func(*args, **kwargs)
            cache[key] = (result, current_time)
            return result
        return wrapper
    return decorator


def measure_time(func):
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        duration = (time.time() - start_time) * 1000
        logger.debug(f"{func.__name__} took {duration:.2f}ms")
        return result
    return wrapper


def retry_on_failure(max_retries: int = 3, delay: float = 1.0):
    """Decorator to retry function on failure."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        time.sleep(delay * (attempt + 1))
                        logger.warning(f"Retry {attempt + 1}/{max_retries} for {func.__name__}")
                    else:
                        logger.error(f"All retries failed for {func.__name__}: {e}")
            raise last_exception
        return wrapper
    return decorator

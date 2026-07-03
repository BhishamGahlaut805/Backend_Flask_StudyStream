"""Logging configuration and utilities."""

import logging
import sys
from datetime import datetime
from flask import Flask


def setup_logging(app: Flask = None):
    """Setup logging configuration."""
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Set up application logger
    if app:
        app.logger.handlers = []
        for handler in root_logger.handlers:
            app.logger.addHandler(handler)
        app.logger.setLevel(logging.INFO)

    # Set logging levels for noisy libraries
    logging.getLogger('pymongo').setLevel(logging.WARNING)
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

    logging.info(f"Logging initialized at {datetime.now().isoformat()}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance."""
    return logging.getLogger(name)


class RequestLogger:
    """Request logging utility."""

    @staticmethod
    def log_request(request, extra: dict = None):
        """Log an incoming request."""
        logger = logging.getLogger('request')

        log_data = {
            'method': request.method,
            'path': request.path,
            'remote_addr': request.remote_addr,
            'user_agent': request.headers.get('User-Agent', 'Unknown')
        }

        if extra:
            log_data.update(extra)

        logger.info(f"Request: {log_data}")
        return log_data

    @staticmethod
    def log_response(response, duration_ms: float, extra: dict = None):
        """Log a response."""
        logger = logging.getLogger('response')

        log_data = {
            'status_code': response.status_code,
            'duration_ms': round(duration_ms, 2)
        }

        if extra:
            log_data.update(extra)

        logger.info(f"Response: {log_data}")
        return log_data


class PerformanceLogger:
    """Performance logging utility."""

    def __init__(self, name: str):
        self.name = name
        self.start_time = None
        self.logger = logging.getLogger('performance')

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, *args):
        duration = (datetime.now() - self.start_time).total_seconds()
        self.logger.info(f"{self.name} took {duration:.3f}s")

    def log(self, message: str):
        """Log a performance message."""
        if self.start_time:
            elapsed = (datetime.now() - self.start_time).total_seconds()
            self.logger.info(f"{self.name} - {message} (elapsed: {elapsed:.3f}s)")
        else:
            self.logger.info(f"{self.name} - {message}")
            
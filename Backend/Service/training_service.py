"""Training Service - Manages training of all models with MongoDB storage."""

import logging
import threading
import time
import traceback
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd

from Service.data_manager import StudentDataManager
from Service.feature_engineering import FeatureEngineeringService
from Model.practice_difficulty import PracticeDifficultyModel
from Model.exam_difficulty import ExamDifficultyModel
from Model.learning_velocity import LearningVelocityModel
from Model.burnout_risk import BurnoutRiskModel
from Model.adaptive_scheduling import AdaptiveSchedulingModel
from Model.global_readiness import GlobalReadinessModel
from Utils.helpers import extract_last_training_info

logger = logging.getLogger(__name__)


class TrainingService:
    """Enhanced service for training all models with MongoDB storage."""

    def __init__(self, config):
        self.config = config
        self.training_jobs = {}
        self.training_history = {}
        logger.info("TrainingService initialized")

    def _get_data_manager(self, student_id: str) -> StudentDataManager:
        """Get or create data manager for student."""
        return StudentDataManager(student_id)

    def _get_retrain_interval(self) -> int:
        """Return row-based retrain interval."""
        return int(getattr(self.config, 'MODEL_RETRAIN_INTERVAL_ROWS',
                          getattr(self.config, 'PRACTICE_RETRAIN_INTERVAL', 100)))

    def _should_train_for_new_rows(self, model_name: str, current_rows: int,
                                   min_samples: int, data_manager: StudentDataManager,
                                   training_id: str) -> Tuple[bool, Optional[int]]:
        """Gate training until enough new rows are available."""
        if current_rows < min_samples:
            return False, None

        metadata_history = data_manager.load_model_metadata(model_name)
        last_rows = extract_last_training_info(metadata_history)

        if last_rows is None:
            return True, None

        retrain_interval = self._get_retrain_interval()
        should_train = (current_rows - last_rows) >= retrain_interval

        if not should_train:
            logger.info(f"[{training_id}] Skipping {model_name} training. rows={current_rows}, "
                       f"last_rows={last_rows}, interval={retrain_interval}")

        return should_train, last_rows

    def cancel_practice_training(self, student_id: str):
        """Mark current/future practice training job as cancelled."""
        self.training_jobs[student_id] = self.training_jobs.get(student_id, {})
        self.training_jobs[student_id]['practice_cancelled'] = True
        self.training_jobs[student_id]['practice_cancelled_at'] = datetime.now().isoformat()
        logger.info(f"Practice training cancelled for {student_id}")

    def _is_practice_cancelled(self, student_id: str) -> bool:
        job_info = self.training_jobs.get(student_id, {})
        return bool(job_info.get('practice_cancelled', False))

    # ==================== PRACTICE DIFFICULTY MODEL ====================

    def train_practice_model(self, student_id: str, force_retrain: bool = False) -> Dict[str, Any]:
        """Train practice difficulty model."""
        training_id = f"practice_{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"[{training_id}] Starting practice model training for student {student_id}")

        try:
            if self._is_practice_cancelled(student_id):
                return {'success': False, 'cancelled': True, 'error': 'Training cancelled'}

            data_manager = self._get_data_manager(student_id)
            practice_df = data_manager.load_practice_features()

            if practice_df.empty:
                return {'success': False, 'error': 'No practice features found'}

            logger.info(f"[{training_id}] Loaded {len(practice_df)} practice feature rows")

            training_data = data_manager.prepare_practice_training_data(
                self.config.MIN_PRACTICE_SAMPLES
            )

            if training_data is None:
                return {
                    'success': False,
                    'error': f'Insufficient training data. Need at least {self.config.MIN_PRACTICE_SAMPLES} samples'
                }

            if not force_retrain:
                should_train, last_rows = self._should_train_for_new_rows(
                    'practice_difficulty',
                    int(len(practice_df)),
                    int(self.config.MIN_PRACTICE_SAMPLES),
                    data_manager,
                    training_id
                )
                if not should_train:
                    return {
                        'success': False,
                        'skipped': True,
                        'error': 'Retrain interval not reached',
                        'feature_rows': int(len(practice_df)),
                        'last_trained_rows': int(last_rows) if last_rows is not None else None,
                        'retrain_interval': self._get_retrain_interval()
                    }

            # Initialize and train model
            model = PracticeDifficultyModel(
                sequence_length=self.config.SEQUENCE_LENGTH_PRACTICE,
                n_features=self.config.PRACTICE_FEATURES_COUNT
            )
            model.feature_names = list(training_data.get('feature_names', []))
            model.build_model()

            start_time = time.time()
            history = model.train(
                X_train=training_data['X_train'],
                y_train=training_data['y_train'],
                X_val=training_data.get('X_val'),
                y_val=training_data.get('y_val'),
                epochs=self.config.EPOCHS,
                batch_size=self.config.BATCH_SIZE,
                verbose=0
            )
            training_time = time.time() - start_time

            # Save model to MongoDB via GridFS
            model_id = data_manager.save_model('practice_difficulty', model, {
                'student_id': student_id,
                'training_id': training_id,
                'samples': len(training_data['X_train'])
            })

            # Evaluate on test set
            test_loss, test_mae = None, None
            if training_data.get('X_test') is not None and len(training_data['X_test']) > 0:
                evaluation = model.evaluate(training_data['X_test'], training_data['y_test'])
                if isinstance(evaluation, (list, tuple)):
                    test_loss = float(evaluation[0])
                    test_mae = float(evaluation[1]) if len(evaluation) > 1 else None

            metadata = {
                'training_id': training_id,
                'timestamp': datetime.now().isoformat(),
                'training_time_seconds': round(training_time, 2),
                'feature_rows_at_training': int(len(practice_df)),
                'samples': int(len(training_data['X_train'])),
                'final_loss': float(history.history['loss'][-1]),
                'final_mae': float(history.history['mae'][-1]) if 'mae' in history.history else None,
                'test_loss': float(test_loss) if test_loss else None,
                'test_mae': float(test_mae) if test_mae else None,
                'epochs_completed': int(len(history.history['loss'])),
                'model_id': model_id,
                'model_architecture': 'RandomForestRegressor'
            }

            data_manager.save_model_metadata('practice_difficulty', metadata)
            logger.info(f"[{training_id}] Practice model training completed")

            return {'success': True, 'metadata': metadata}

        except Exception as e:
            logger.error(f"[{training_id}] Practice model training error: {e}\n{traceback.format_exc()}")
            return {'success': False, 'error': str(e)}

    def train_practice_model_async(self, student_id: str, force_retrain: bool = False) -> bool:
        """Train practice model asynchronously."""
        logger.info(f"Requesting async practice training for student {student_id}")

        self.training_jobs[student_id] = self.training_jobs.get(student_id, {})
        self.training_jobs[student_id]['practice_cancelled'] = False

        if student_id in self.training_jobs and self.training_jobs[student_id].get('practice_in_progress', False):
            logger.info(f"Practice training already in progress for {student_id}")
            return False

        def _train():
            thread_id = threading.current_thread().name
            logger.info(f"[Thread:{thread_id}] Starting async practice training for {student_id}")

            try:
                self.training_jobs[student_id]['practice_in_progress'] = True
                self.training_jobs[student_id]['practice_start_time'] = datetime.now().isoformat()

                result = self.train_practice_model(student_id, force_retrain=force_retrain)
                self.training_jobs[student_id]['practice_result'] = result
                self.training_jobs[student_id]['practice_completed_time'] = datetime.now().isoformat()

                if result.get('success'):
                    logger.info(f"[Thread:{thread_id}] Practice training completed successfully for {student_id}")
                else:
                    logger.warning(f"[Thread:{thread_id}] Practice training failed: {result.get('error')}")

            except Exception as e:
                logger.error(f"[Thread:{thread_id}] Unhandled exception: {e}")
                self.training_jobs[student_id]['practice_result'] = {'success': False, 'error': str(e)}
            finally:
                self.training_jobs[student_id]['practice_in_progress'] = False

        thread = threading.Thread(target=_train, name=f"Train-{student_id}-{datetime.now().strftime('%H%M%S')}")
        thread.daemon = True
        thread.start()

        return True

    # ==================== GLOBAL FEATURES GENERATION ====================

    def generate_global_features(self, student_id: str) -> Dict[str, Any]:
        """Generate global features from practice data."""
        logger.info(f"Generating global features for student {student_id}")

        try:
            data_manager = self._get_data_manager(student_id)
            practice_df = data_manager.load_practice_features()

            if practice_df.empty:
                return {'success': False, 'error': 'No practice data found', 'generated': False}

            min_global_samples = getattr(self.config, 'MIN_PRACTICE_SAMPLES_FOR_GLOBAL', 40)
            if len(practice_df) < min_global_samples:
                return {
                    'success': False,
                    'error': f'Need at least {min_global_samples} samples',
                    'current_samples': len(practice_df),
                    'generated': False
                }

            feature_service = FeatureEngineeringService()
            global_df = feature_service.compute_global_features(practice_df)

            if global_df.empty:
                return {'success': False, 'error': 'Global feature computation failed', 'generated': False}

            data_manager.save_global_features(global_df)

            if len(global_df) >= self.config.SEQUENCE_LENGTH_GLOBAL + 2:
                self.train_global_model_async(student_id)

            return {
                'success': True,
                'generated': True,
                'rows_generated': len(global_df),
                'sessions_processed': len(global_df['session_id'].unique()) if 'session_id' in global_df.columns else 0
            }

        except Exception as e:
            logger.error(f"Error generating global features: {e}\n{traceback.format_exc()}")
            return {'success': False, 'error': str(e), 'generated': False}

    def generate_global_features_async(self, student_id: str) -> bool:
        """Generate global features asynchronously."""
        def _generate():
            thread_id = threading.current_thread().name
            logger.info(f"[Thread:{thread_id}] Starting async global feature generation for {student_id}")
            try:
                result = self.generate_global_features(student_id)
                logger.info(f"[Thread:{thread_id}] Global feature generation completed: {result}")
            except Exception as e:
                logger.error(f"[Thread:{thread_id}] Error: {e}")

        thread = threading.Thread(target=_generate, name=f"Global-{student_id}-{datetime.now().strftime('%H%M%S')}")
        thread.daemon = True
        thread.start()
        return True

    # ==================== GLOBAL MODEL TRAINING ====================

    def train_global_model(self, student_id: str) -> Dict[str, Any]:
        """Train global readiness model."""
        training_id = f"global_{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"[{training_id}] Starting global model training")

        try:
            data_manager = self._get_data_manager(student_id)
            global_df = data_manager.load_global_features()

            if global_df.empty or len(global_df) < self.config.SEQUENCE_LENGTH_GLOBAL + 2:
                return {'success': False, 'error': 'Insufficient global data'}

            should_train, last_rows = self._should_train_for_new_rows(
                'global_readiness',
                int(len(global_df)),
                int(self.config.SEQUENCE_LENGTH_GLOBAL + 2),
                data_manager,
                training_id
            )
            if not should_train:
                return {
                    'success': False,
                    'skipped': True,
                    'error': 'Retrain interval not reached',
                    'feature_rows': int(len(global_df)),
                    'last_trained_rows': int(last_rows) if last_rows is not None else None
                }

            from config import Config
            feature_cols = Config.GLOBAL_FEATURES
            available_cols = [col for col in feature_cols if col in global_df.columns]

            if len(available_cols) < 8:
                return {'success': False, 'error': 'Insufficient feature columns'}

            data_values = global_df[available_cols].values.astype(np.float32)
            target_values = global_df[Config.GLOBAL_TARGET].values.astype(np.float32)

            X, y = [], []
            seq_length = self.config.SEQUENCE_LENGTH_GLOBAL

            for i in range(len(data_values) - seq_length):
                X.append(data_values[i:i + seq_length])
                y.append(target_values[i + seq_length])

            if len(X) < 3:
                return {'success': False, 'error': 'Insufficient sequences'}

            X, y = np.array(X), np.array(y)
            n = len(X)
            split = int(n * 0.8)
            X_train, X_test = X[:split], X[split:]
            y_train, y_test = y[:split], y[split:]

            model = GlobalReadinessModel(sequence_length=seq_length, n_features=len(available_cols))
            model.build_model()

            history = model.train(
                X_train=X_train, y_train=y_train,
                X_val=X_test, y_val=y_test,
                epochs=min(self.config.EPOCHS, 30),
                batch_size=16,
                verbose=0
            )

            test_loss, test_mae = model.evaluate(X_test, y_test)

            # Save model to MongoDB
            model_id = data_manager.save_model('global_readiness', model, {
                'student_id': student_id,
                'training_id': training_id
            })

            metadata = {
                'training_id': training_id,
                'timestamp': datetime.now().isoformat(),
                'samples': len(X_train),
                'test_samples': len(X_test),
                'feature_rows_at_training': int(len(global_df)),
                'final_loss': float(history.history['loss'][-1]),
                'test_loss': float(test_loss),
                'test_mae': float(test_mae) if test_mae else None,
                'epochs_completed': len(history.history['loss']),
                'model_id': model_id,
                'model_architecture': 'RandomForestRegressor'
            }

            data_manager.save_model_metadata('global_readiness', metadata)
            return {'success': True, 'metadata': metadata}

        except Exception as e:
            logger.error(f"[{training_id}] Global model training error: {e}\n{traceback.format_exc()}")
            return {'success': False, 'error': str(e)}

    def train_global_model_async(self, student_id: str) -> bool:
        """Train global model asynchronously."""
        def _train():
            thread_id = threading.current_thread().name
            logger.info(f"[Thread:{thread_id}] Starting async global training")
            try:
                self.training_jobs[student_id] = self.training_jobs.get(student_id, {})
                self.training_jobs[student_id]['global_in_progress'] = True
                result = self.train_global_model(student_id)
                self.training_jobs[student_id]['global_result'] = result
            except Exception as e:
                logger.error(f"[Thread:{thread_id}] Error: {e}")
            finally:
                self.training_jobs[student_id]['global_in_progress'] = False

        thread = threading.Thread(target=_train, name=f"GlobalTrain-{student_id}-{datetime.now().strftime('%H%M%S')}")
        thread.daemon = True
        thread.start()
        return True

    # ==================== EXAM DIFFICULTY MODEL ====================

    def train_exam_model(self, student_id: str) -> Dict[str, Any]:
        """Train exam difficulty model."""
        training_id = f"exam_{student_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        logger.info(f"[{training_id}] Starting exam model training")

        try:
            data_manager = self._get_data_manager(student_id)
            exam_df = data_manager.load_exam_features()

            if exam_df.empty:
                return {'success': False, 'error': 'No exam features found'}

            should_train, last_rows = self._should_train_for_new_rows(
                'exam_difficulty',
                int(len(exam_df)),
                int(self.config.MIN_EXAM_SAMPLES),
                data_manager,
                training_id
            )
            if not should_train:
                return {
                    'success': False,
                    'skipped': True,
                    'error': 'Retrain interval not reached',
                    'feature_rows': int(len(exam_df)),
                    'last_trained_rows': int(last_rows) if last_rows is not None else None
                }

            training_data = data_manager.prepare_exam_training_data(self.config.MIN_EXAM_SAMPLES)
            if training_data is None:
                return {'success': False, 'error': 'Insufficient exam data'}

            model = ExamDifficultyModel(
                sequence_length=self.config.SEQUENCE_LENGTH_EXAM,
                n_features=self.config.EXAM_FEATURES
            )
            model.build_model()

            history = model.train(
                X_train=training_data['X_train'],
                y_train=training_data['y_train'],
                X_val=training_data.get('X_val'),
                y_val=training_data.get('y_val'),
                epochs=min(self.config.EPOCHS, 50),
                batch_size=16,
                verbose=0
            )

            model_id = data_manager.save_model('exam_difficulty', model, {
                'student_id': student_id,
                'training_id': training_id
            })

            metadata = {
                'training_id': training_id,
                'timestamp': datetime.now().isoformat(),
                'feature_rows_at_training': int(len(exam_df)),
                'samples': len(training_data['X_train']),
                'final_loss': float(history.history['loss'][-1]),
                'epochs_completed': len(history.history['loss']),
                'model_id': model_id,
                'model_architecture': 'RandomForestRegressor'
            }

            data_manager.save_model_metadata('exam_difficulty', metadata)
            return {'success': True, 'metadata': metadata}

        except Exception as e:
            logger.error(f"[{training_id}] Exam model training error: {e}\n{traceback.format_exc()}")
            return {'success': False, 'error': str(e)}

    def train_exam_model_async(self, student_id: str) -> bool:
        """Train exam model asynchronously."""
        def _train():
            thread_id = threading.current_thread().name
            logger.info(f"[Thread:{thread_id}] Starting async exam training")
            try:
                self.training_jobs[student_id] = self.training_jobs.get(student_id, {})
                self.training_jobs[student_id]['exam_in_progress'] = True
                result = self.train_exam_model(student_id)
                self.training_jobs[student_id]['exam_result'] = result
            except Exception as e:
                logger.error(f"[Thread:{thread_id}] Error: {e}")
            finally:
                self.training_jobs[student_id]['exam_in_progress'] = False

        thread = threading.Thread(target=_train, name=f"ExamTrain-{student_id}-{datetime.now().strftime('%H%M%S')}")
        thread.daemon = True
        thread.start()
        return True

    # ==================== MODEL STATUS ====================

    def get_training_status(self, student_id: str) -> Dict[str, Any]:
        """Get comprehensive training status."""
        try:
            data_manager = self._get_data_manager(student_id)
            status = self.training_jobs.get(student_id, {})

            status['data_summary'] = {
                'practice_features': len(data_manager.load_practice_features()),
                'global_features': len(data_manager.load_global_features()),
                'exam_features': len(data_manager.load_exam_features()),
                'practice_threshold': self.config.MIN_PRACTICE_SAMPLES,
                'global_threshold': getattr(self.config, 'MIN_PRACTICE_SAMPLES_FOR_GLOBAL', 40),
                'exam_threshold': self.config.MIN_EXAM_SAMPLES
            }

            status['models'] = {}
            for model_name in ['practice_difficulty', 'exam_difficulty', 'global_readiness']:
                metadata = data_manager.load_model_metadata(model_name)
                status['models'][model_name] = {
                    'trained': len(metadata) > 0,
                    'last_trained': metadata[0].get('timestamp') if metadata else None,
                    'training_count': len(metadata)
                }

            return status
        except Exception as e:
            logger.error(f"Error getting training status: {e}")
            return {'error': str(e)}
        
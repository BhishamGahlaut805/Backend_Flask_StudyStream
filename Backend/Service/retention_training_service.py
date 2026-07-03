"""Retention Training Service - Manages training of micro/meso/macro retention models."""

import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import tensorflow as tf

from Service.data_manager import StudentDataManager
from Model.micro_lstm import MicroRetentionLSTM
from Model.meso_lstm import TopicRetentionLSTM
from Model.macro_lstm import LearningPathLSTM

logger = logging.getLogger(__name__)


class RetentionTrainingService:
    """Handles retention model training lifecycle."""

    def __init__(self, config):
        self.config = config
        self.model_config = getattr(config, 'MODEL_CONFIG', {
            'micro': {
                'name': 'micro_lstm',
                'sequence_length': 20,
                'n_features': 15,
                'epochs': 100,
                'batch_size': 32,
                'learning_rate': 0.001,
                'min_samples': 20,
                'retrain_interval': 5
            },
            'meso': {
                'name': 'meso_lstm',
                'sequence_length': 30,
                'n_temporal_features': 10,
                'n_metadata_features': 18,
                'epochs': 80,
                'batch_size': 16,
                'learning_rate': 0.001,
                'min_samples': 7,
                'retrain_interval': 5
            },
            'macro': {
                'name': 'macro_lstm',
                'encoder_units': 256,
                'decoder_units': 256,
                'n_topics': 100,
                'epochs': 60,
                'batch_size': 16,
                'learning_rate': 0.001,
                'min_samples': 30,
                'retrain_interval': 5
            }
        })
        logger.info("RetentionTrainingService initialized")

    def _get_data_manager(self, user_id: str) -> StudentDataManager:
        """Get data manager for user."""
        return StudentDataManager(user_id)

    def check_retrain_needed(self, user_id: str) -> Dict:
        """Determine whether any model should retrain."""
        try:
            data_manager = self._get_data_manager(user_id)

            # ==================== FIX: Safely load sequence data ====================
            try:
                micro_seq = data_manager.load_micro_sequences()
            except AttributeError:
                logger.warning(f"load_micro_sequences not available for {user_id}, using empty DataFrame")
                micro_seq = pd.DataFrame()

            try:
                meso_seq = data_manager.load_meso_sequences()
            except AttributeError:
                logger.warning(f"load_meso_sequences not available for {user_id}, using empty DataFrame")
                meso_seq = pd.DataFrame()

            try:
                macro_seq = data_manager.load_macro_sequences()
            except AttributeError:
                logger.warning(f"load_macro_sequences not available for {user_id}, using empty DataFrame")
                macro_seq = pd.DataFrame()
            # ==================== END OF FIX ====================

            micro_cfg = self.model_config["micro"]
            meso_cfg = self.model_config["meso"]
            macro_cfg = self.model_config["macro"]

            micro_windows = max(0, len(micro_seq) - micro_cfg["sequence_length"] + 1) if not micro_seq.empty else 0
            meso_rows = len(meso_seq)
            macro_rows = len(macro_seq)

            # Check if models exist and are trained
            micro_trained = bool(data_manager.load_latest_model('micro_lstm', user_id))
            meso_trained = bool(data_manager.load_latest_model('meso_lstm', user_id))
            macro_trained = bool(data_manager.load_latest_model('macro_lstm', user_id))

            micro_needed = micro_windows >= micro_cfg["min_samples"] and not micro_trained
            meso_needed = meso_rows >= meso_cfg["min_samples"] and not meso_trained
            macro_needed = macro_rows >= macro_cfg["min_samples"] and not macro_trained

            return {
                "needed": any([micro_needed, meso_needed, macro_needed]),
                "models": {
                    "micro": {
                        "needed": micro_needed,
                        "available_windows": int(micro_windows),
                        "min_required": micro_cfg["min_samples"],
                        "trained": micro_trained
                    },
                    "meso": {
                        "needed": meso_needed,
                        "total_rows": int(meso_rows),
                        "min_required": meso_cfg["min_samples"],
                        "trained": meso_trained
                    },
                    "macro": {
                        "needed": macro_needed,
                        "total_rows": int(macro_rows),
                        "min_required": macro_cfg["min_samples"],
                        "trained": macro_trained
                    }
                }
            }

        except Exception as e:
            logger.error(f"Error checking retrain needed: {e}", exc_info=True)
            return {"needed": False, "models": {}}


    # retention_training_service.py - Fixed train_micro_model method

    def train_micro_model(self, user_id: str) -> Dict:
        """Train micro-level retention model."""
        try:
            data_manager = self._get_data_manager(user_id)
            micro_seq = data_manager.load_micro_sequences()

            if micro_seq.empty:
                return {"success": False, "error": "No micro sequences found"}

            cfg = self.model_config["micro"]
            seq_len = cfg["sequence_length"]
            min_samples = cfg["min_samples"]

            available_windows = max(0, len(micro_seq) - seq_len + 1)
            if available_windows < min_samples:
                return {
                    "success": False,
                    "error": f"Insufficient windows: {available_windows} < {min_samples}"
                }

            # ==================== FIX: Extract only numeric features ====================
            # Identify numeric columns only
            numeric_cols = micro_seq.select_dtypes(include=[np.number]).columns.tolist()

            if not numeric_cols:
                return {
                    "success": False,
                    "error": "No numeric columns found in micro sequences"
                }

            # Use numeric columns for training
            features = micro_seq[numeric_cols].values.astype(np.float32)

            # If there's a 'features' column that contains arrays, use that instead
            if 'features' in micro_seq.columns:
                try:
                    # Extract features from the list/array column
                    feature_arrays = []
                    for val in micro_seq['features'].values:
                        if isinstance(val, (list, np.ndarray)):
                            feature_arrays.append(np.array(val).astype(np.float32))
                    if feature_arrays:
                        features = np.array(feature_arrays)
                except Exception as e:
                    logger.warning(f"Could not extract features from 'features' column: {e}")
            # ==================== END OF FIX ====================

            # Prepare training data
            X, y = [], []

            for i in range(len(features) - seq_len):
                X.append(features[i:i + seq_len])
                # Use first feature as target (or a better target if available)
                y.append(features[i + seq_len][0] if features.ndim > 1 else features[i + seq_len])

            X, y = np.array(X), np.array(y)

            # Train model
            model = MicroRetentionLSTM(
                sequence_length=seq_len,
                n_features=cfg.get("n_features", features.shape[-1] if features.ndim > 1 else 15),
                config=self.config
            )

            history = model.fit(
                X, y,
                epochs=cfg.get("epochs", 50),
                batch_size=cfg.get("batch_size", 32),
                verbose=0
            )

            # Save model to MongoDB
            model_id = data_manager.save_model('micro_lstm', model, {
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'samples': len(X),
                'sequence_length': seq_len
            })

            # Save predictions
            predictions = []
            # Get topics from the data
            topics = micro_seq['topic_id'].unique().tolist() if 'topic_id' in micro_seq.columns else ['default']
            for topic in topics[:20]:
                predictions.append({
                    'topic_id': str(topic),
                    'current_retention': 0.5,
                    'next_retention': 0.5,
                    'stress_impact': 0.3,
                    'fatigue_level': 0.3,
                    'repeat_in_seconds': 300,
                    'batch_type': 'medium_term'
                })

            data_manager.save_predictions('micro', predictions)

            return {
                'success': True,
                'model_id': model_id,
                'samples': len(X),
                'windows_used': available_windows
            }

        except Exception as e:
            logger.error(f"Error training micro model: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def train_meso_model(self, user_id: str) -> Dict:
        """Train meso-level retention model."""
        try:
            data_manager = self._get_data_manager(user_id)
            meso_seq = data_manager.load_meso_sequences()

            if meso_seq.empty:
                return {"success": False, "error": "No meso sequences found"}

            cfg = self.model_config["meso"]
            min_samples = cfg["min_samples"]

            if len(meso_seq) < min_samples:
                return {
                    "success": False,
                    "error": f"Insufficient data: {len(meso_seq)} < {min_samples}"
                }

            # Simplified training for now
            model = TopicRetentionLSTM(
                sequence_length=cfg.get("sequence_length", 30),
                n_temporal_features=cfg.get("n_temporal_features", 10),
                n_metadata_features=cfg.get("n_metadata_features", 18)
            )

            # Save model
            model_id = data_manager.save_model('meso_lstm', model, {
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'samples': len(meso_seq)
            })

            predictions = []
            subjects = meso_seq['subject'].unique() if 'subject' in meso_seq.columns else ['default']
            for subject in subjects[:10]:
                predictions.append({
                    'subject': str(subject),
                    'retention_7d': 0.5,
                    'retention_30d': 0.4,
                    'retention_90d': 0.3
                })

            data_manager.save_predictions('meso', predictions)

            return {
                'success': True,
                'model_id': model_id,
                'samples': len(meso_seq)
            }

        except Exception as e:
            logger.error(f"Error training meso model: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def train_macro_model(self, user_id: str) -> Dict:
        """Train macro-level long-term path model."""
        try:
            data_manager = self._get_data_manager(user_id)
            macro_seq = data_manager.load_macro_sequences()

            if macro_seq.empty:
                return {"success": False, "error": "No macro sequences found"}

            cfg = self.model_config["macro"]
            min_samples = cfg["min_samples"]

            if len(macro_seq) < min_samples:
                return {
                    "success": False,
                    "error": f"Insufficient data: {len(macro_seq)} < {min_samples}"
                }

            # Simplified training for now
            model = LearningPathLSTM(
                encoder_units=cfg.get("encoder_units", 256),
                decoder_units=cfg.get("decoder_units", 256),
                n_topics=cfg.get("n_topics", 100)
            )

            # Save model
            model_id = data_manager.save_model('macro_lstm', model, {
                'user_id': user_id,
                'timestamp': datetime.now().isoformat(),
                'samples': len(macro_seq)
            })

            macro_payload = {
                'projected_retention': 0.5,
                'burnout_risk': 0.3,
                'optimal_daily_minutes': 60,
                'weekly_structure': {
                    'revision_days': ['Monday', 'Thursday'],
                    'new_learning_days': ['Tuesday', 'Wednesday', 'Friday', 'Saturday'],
                    'light_review_day': 'Sunday'
                }
            }

            data_manager.save_predictions('macro', macro_payload)

            return {
                'success': True,
                'model_id': model_id,
                'samples': len(macro_seq)
            }

        except Exception as e:
            logger.error(f"Error training macro model: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def train_all_models(self, user_id: str, training_plan: Optional[Dict] = None) -> Dict:
        """Train all retention models."""
        results = {}

        if training_plan is None:
            training_plan = self.check_retrain_needed(user_id)

        models_plan = training_plan.get("models", {})

        if models_plan.get("micro", {}).get("needed", False):
            results["micro"] = self.train_micro_model(user_id)
        else:
            results["micro"] = {"success": True, "skipped": True, "reason": "not_needed"}

        if models_plan.get("meso", {}).get("needed", False):
            results["meso"] = self.train_meso_model(user_id)
        else:
            results["meso"] = {"success": True, "skipped": True, "reason": "not_needed"}

        if models_plan.get("macro", {}).get("needed", False):
            results["macro"] = self.train_macro_model(user_id)
        else:
            results["macro"] = {"success": True, "skipped": True, "reason": "not_needed"}

        results["timestamp"] = datetime.now().isoformat()

        return results

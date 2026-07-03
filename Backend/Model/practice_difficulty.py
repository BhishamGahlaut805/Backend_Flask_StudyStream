"""Practice Difficulty Model - Random Forest for difficulty prediction."""

import json
import os
import pickle
import logging
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from datetime import datetime

from Model.base import BaseModel

logger = logging.getLogger(__name__)


class PracticeDifficultyModel(BaseModel):
    """Random Forest model for practice difficulty prediction."""

    def __init__(self, sequence_length=10, n_features=12):
        super().__init__('practice_difficulty')
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.target = 'next_difficulty'
        self.feature_names = []

    def build_model(self):
        """Initialize RandomForestRegressor."""
        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_split=6,
            min_samples_leaf=3,
            max_features=max(1, int(self.n_features)),
            random_state=42,
            n_jobs=-1
        )
        self.built = True
        return self.model

    def _prepare_flat_features(self, X):
        """Convert sequence input into 2D features."""
        X = np.asarray(X, dtype=np.float32)

        if len(X.shape) == 2:
            if X.shape == (self.sequence_length, self.n_features):
                X = X.reshape(1, self.sequence_length, self.n_features)
            elif X.shape[1] == self.n_features:
                X = X.reshape(X.shape[0], 1, self.n_features)

        if len(X.shape) != 3:
            raise ValueError(f"Expected 3D input, got {X.shape}")

        if X.shape[2] != self.n_features:
            raise ValueError(f"Feature mismatch: expected {self.n_features}, got {X.shape[2]}")

        return X.reshape(X.shape[0], -1)

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=1, batch_size=32, model_path=None, verbose=0):
        """Train Random Forest and return history."""
        try:
            if self.model is None:
                self.build_model()

            X_train_flat = self._prepare_flat_features(X_train)
            y_train = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)

            X_train_scaled = self.scaler_X.fit_transform(X_train_flat)
            y_train_scaled = self.scaler_y.fit_transform(y_train).flatten()

            self.model.fit(X_train_scaled, y_train_scaled)

            train_pred_scaled = self.model.predict(X_train_scaled).reshape(-1, 1)
            train_pred = self.scaler_y.inverse_transform(train_pred_scaled).flatten()
            y_train_unscaled = self.scaler_y.inverse_transform(y_train_scaled.reshape(-1, 1)).flatten()

            train_loss = float(mean_squared_error(y_train_unscaled, train_pred))
            train_mae = float(mean_absolute_error(y_train_unscaled, train_pred))

            class History:
                def __init__(self):
                    self.history = {'loss': [train_loss], 'mae': [train_mae]}

            history_entry = {
                'timestamp': datetime.now().isoformat(),
                'epochs_completed': 1,
                'final_loss': train_loss,
                'samples': int(len(X_train_flat))
            }

            if X_val is not None and y_val is not None and len(X_val) > 0:
                X_val_flat = self._prepare_flat_features(X_val)
                X_val_scaled = self.scaler_X.transform(X_val_flat)
                y_val = np.asarray(y_val, dtype=np.float32).reshape(-1, 1)
                y_val_scaled = self.scaler_y.transform(y_val).flatten()

                val_pred_scaled = self.model.predict(X_val_scaled).reshape(-1, 1)
                val_pred = self.scaler_y.inverse_transform(val_pred_scaled).flatten()
                y_val_unscaled = self.scaler_y.inverse_transform(y_val_scaled.reshape(-1, 1)).flatten()
                history_entry['final_val_loss'] = float(mean_squared_error(y_val_unscaled, val_pred))

            self.training_history.append(history_entry)

            if model_path:
                self.save(model_path)

            return History()

        except Exception as e:
            logger.error(f"Training error: {e}")
            raise

    def predict(self, X):
        """Predict next difficulty values."""
        if self.model is None:
            raise ValueError("Model not trained")

        X_flat = self._prepare_flat_features(X)
        X_scaled = self.scaler_X.transform(X_flat)
        pred_scaled = self.model.predict(X_scaled).reshape(-1, 1)
        pred = self.scaler_y.inverse_transform(pred_scaled).flatten()
        return pred

    def predict_next(self, recent_features):
        """Predict with confidence and smoothing."""
        recent_features = np.asarray(recent_features, dtype=np.float32)

        if len(recent_features.shape) == 2:
            X = recent_features.reshape(1, self.sequence_length, self.n_features)
        else:
            X = recent_features

        prediction = float(self.predict(X)[0])

        last_difficulty = prediction
        if len(recent_features.shape) == 2 and recent_features.shape[0] > 0:
            last_difficulty = float(recent_features[-1][7])

        stability = 1.0 - abs(prediction - last_difficulty)
        confidence = float(np.clip(stability * 0.8 + 0.2, 0.5, 0.95))

        smooth_factor = 0.7 if confidence > 0.8 else 0.5
        smoothed = smooth_factor * prediction + (1 - smooth_factor) * last_difficulty

        return {
            'predicted_difficulty': float(np.clip(prediction, 0.2, 0.95)),
            'smoothed_difficulty': float(np.clip(smoothed, 0.2, 0.95)),
            'confidence': confidence
        }

    def evaluate(self, X, y):
        """Evaluate model and return loss/mae."""
        if self.model is None:
            raise ValueError("Model not trained")

        X_flat = self._prepare_flat_features(X)
        X_scaled = self.scaler_X.transform(X_flat)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        y_scaled = self.scaler_y.transform(y).flatten()

        pred_scaled = self.model.predict(X_scaled).reshape(-1, 1)
        pred = self.scaler_y.inverse_transform(pred_scaled).flatten()
        y_unscaled = self.scaler_y.inverse_transform(y_scaled.reshape(-1, 1)).flatten()

        loss = float(mean_squared_error(y_unscaled, pred))
        mae = float(mean_absolute_error(y_unscaled, pred))
        return loss, mae

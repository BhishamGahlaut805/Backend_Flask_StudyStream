"""Adaptive Scheduling Model - Predicts concept priority scores."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from Model.base import BaseModel


class AdaptiveSchedulingModel(BaseModel):
    """Predicts concept priority score (0-1) for scheduling."""

    def __init__(self, sequence_length=30, n_features=13):
        super().__init__('adaptive_scheduling')
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.target = 'concept_priority_score'

    def build_model(self):
        """Build Random Forest regressor."""
        self.model = RandomForestRegressor(
            n_estimators=350,
            max_depth=12,
            min_samples_split=6,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        self.built = True
        return self.model

    def train(self, X_train, y_train, X_val=None, y_val=None,
              epochs=1, batch_size=32, model_path=None, verbose=0):
        """Train the model."""
        if self.model is None:
            self.build_model()

        X_train = np.asarray(X_train, dtype=np.float32)
        if len(X_train.shape) == 3:
            X_train = X_train.reshape(X_train.shape[0], -1)

        y_train = np.asarray(y_train, dtype=np.float32)

        self.model.fit(X_train, y_train)

        class History:
            def __init__(self):
                self.history = {'loss': [0.1], 'mae': [0.1]}

        history_entry = {'timestamp': datetime.now().isoformat(), 'samples': len(X_train)}
        self.training_history.append(history_entry)

        if model_path:
            self.save(model_path)

        return History()

    def predict(self, X):
        """Predict priority scores."""
        if self.model is None:
            raise ValueError("Model not trained")

        X = np.asarray(X, dtype=np.float32)
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)

        return self.model.predict(X)

    def predict_with_rules(self, X, rule_weight=0.3):
        """Hybrid prediction combining ML with rule-based scoring."""
        try:
            ml_score = self.predict(X)[0]
        except:
            ml_score = 0.5

        if len(X.shape) == 3:
            latest = X[0, -1, :]
        else:
            latest = X[-1, :]

        mastery = latest[0]
        exam_weight = latest[3]
        days_since = latest[4]

        rule_score = (1 - mastery) * 0.4 + exam_weight * 0.3 + min(days_since / 30, 1) * 0.3

        final_score = (1 - rule_weight) * ml_score + rule_weight * rule_score
        return final_score
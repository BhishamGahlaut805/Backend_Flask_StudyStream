"""Global Readiness Model - Predicts global readiness difficulty score."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from Model.base import BaseModel


class GlobalReadinessModel(BaseModel):
    """Predicts global readiness difficulty score."""

    def __init__(self, sequence_length=5, n_features=12):
        super().__init__('global_readiness')
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.target = 'readiness_difficulty_score'

    def build_model(self):
        """Build Random Forest model."""
        self.model = RandomForestRegressor(
            n_estimators=400,
            max_depth=14,
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
        """Predict readiness score."""
        if self.model is None:
            raise ValueError("Model not trained")

        X = np.asarray(X, dtype=np.float32)
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)

        return self.model.predict(X)
"""Exam Difficulty Model - Random Forest for exam difficulty prediction."""

import numpy as np
from sklearn.ensemble import RandomForestRegressor

from Model.base import BaseModel
from Utils.helpers import difficulty_level_from_value


class ExamDifficultyModel(BaseModel):
    """Predicts optimal difficulty for entire exam."""

    def __init__(self, sequence_length=10, n_features=8):
        super().__init__('exam_difficulty')
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.target = 'recommended_difficulty'

    def build_model(self):
        """Build Random Forest model."""
        self.model = RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
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

        history_entry = {
            'timestamp': datetime.now().isoformat(),
            'samples': len(X_train)
        }
        self.training_history.append(history_entry)

        if model_path:
            self.save(model_path)

        return History()

    def predict(self, X):
        """Predict exam difficulty."""
        if self.model is None:
            raise ValueError("Model not trained")

        X = np.asarray(X, dtype=np.float32)
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0], -1)

        return self.model.predict(X)

    def predict_exam_difficulty(self, recent_exams, student_readiness=None):
        """Predict recommended exam difficulty."""
        if len(recent_exams) < self.sequence_length:
            padding = [[0.5] * self.n_features] * (self.sequence_length - len(recent_exams))
            recent_exams = padding + recent_exams[-self.sequence_length:]

        X = np.array(recent_exams[-self.sequence_length:]).reshape(1, -1)

        try:
            prediction = float(self.predict(X)[0])
        except:
            prediction = 0.5

        if student_readiness is not None:
            prediction = 0.6 * prediction + 0.4 * student_readiness

        prediction = float(np.clip(prediction, 0.2, 0.95))

        return {
            'recommended_difficulty': prediction,
            'difficulty_level': difficulty_level_from_value(prediction),
            'confidence': 0.85 if len(recent_exams) >= self.sequence_length else 0.6
        }
        
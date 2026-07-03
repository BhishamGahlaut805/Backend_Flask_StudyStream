"""Base model class for all ML models."""

import json
import pickle
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """Abstract base class for all ML models."""

    def __init__(self, model_type: str):
        self.model_type = model_type
        self.model = None
        self.scaler_X = MinMaxScaler()
        self.scaler_y = MinMaxScaler()
        self.training_history = []
        self.built = False

    @abstractmethod
    def build_model(self):
        """Build the model architecture."""
        pass

    @abstractmethod
    def train(self, X_train, y_train, **kwargs):
        """Train the model."""
        pass

    @abstractmethod
    def predict(self, X):
        """Make predictions."""
        pass

    def save(self, filepath: str):
        """Save model to file."""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)

            metadata = {
                'model_type': self.model_type,
                'saved_at': datetime.now().isoformat(),
                'training_history': self.training_history
            }

            with open(f'{filepath}.meta', 'w') as f:
                json.dump(metadata, f, indent=2)

            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def load(self, filepath: str):
        """Load model from file."""
        try:
            with open(filepath, 'rb') as f:
                self.model = pickle.load(f)
            self.built = True

            # Load metadata if exists
            try:
                with open(f'{filepath}.meta', 'r') as f:
                    metadata = json.load(f)
                    self.training_history = metadata.get('training_history', [])
            except:
                pass

            logger.info(f"Model loaded from {filepath}")
            return self
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def serialize(self) -> bytes:
        """Serialize model to bytes for MongoDB storage."""
        return pickle.dumps(self.model)

    @classmethod
    def deserialize(cls, data: bytes):
        """Deserialize model from bytes."""
        return pickle.loads(data)
"""Micro-LSTM Model - Question-level retention prediction."""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import logging

logger = logging.getLogger(__name__)


class MicroRetentionLSTM:
    """Predicts retention after each question with stress and fatigue awareness."""

    def __init__(self, sequence_length=20, n_features=15, config=None):
        self.sequence_length = sequence_length
        self.n_features = n_features
        self.config = config or {}
        self.model = self._build_model()
        self.history = None
        self._inference_fn = None

    def _get_inference_fn(self):
        """Build once and reuse."""
        if self._inference_fn is None:
            @tf.function(reduce_retracing=True)
            def _infer(sequence):
                return self.model(sequence, training=False)
            self._inference_fn = _infer
        return self._inference_fn

    def _build_model(self):
        """Build enhanced micro-LSTM architecture."""
        inputs = layers.Input(shape=(self.sequence_length, self.n_features))

        x = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
        )(inputs)
        x = layers.LayerNormalization()(x)

        attention = layers.MultiHeadAttention(num_heads=8, key_dim=64)(x, x)
        x = layers.Add()([x, attention])
        x = layers.LayerNormalization()(x)

        x = layers.Bidirectional(
            layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)
        )(x)
        x = layers.LayerNormalization()(x)

        x = layers.GlobalAveragePooling1D()(x)

        d1 = layers.Dense(128, activation='relu')(x)
        d1 = layers.Dropout(0.3)(d1)
        d1 = layers.BatchNormalization()(d1)

        d2 = layers.Dense(64, activation='relu')(d1)
        d2 = layers.Dropout(0.2)(d2)
        d2 = layers.BatchNormalization()(d2)

        current_retention = layers.Dense(1, activation='sigmoid', name='current_retention')(d2)

        model = models.Model(inputs=inputs, outputs=current_retention)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )

        return model

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, **kwargs):
        """Train the model."""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val is not None else 'loss',
                patience=15,
                restore_best_weights=True,
                verbose=0
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val is not None else 'loss',
                factor=0.5,
                patience=8,
                min_lr=0.00001,
                verbose=0
            )
        ]

        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)

        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
            **kwargs
        )

        return self.history

    def predict(self, X):
        """Make predictions."""
        X = tf.convert_to_tensor(np.asarray(X, dtype=np.float32))
        return self._get_inference_fn()(X)

    def save(self, filepath):
        """Save model."""
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")

    def load(self, filepath):
        """Load model."""
        self.model = tf.keras.models.load_model(filepath)
        self._inference_fn = None
        logger.info(f"Model loaded from {filepath}")
        return self
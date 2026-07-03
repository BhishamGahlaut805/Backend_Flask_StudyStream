"""Meso-LSTM Model - Chapter-level retention prediction."""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import logging

logger = logging.getLogger(__name__)


class TopicRetentionLSTM:
    """Predicts chapter-level retention at 7, 30, 90 days."""

    def __init__(self, sequence_length=30, n_temporal_features=10, n_metadata_features=18):
        self.sequence_length = sequence_length
        self.n_temporal_features = n_temporal_features
        self.n_metadata_features = n_metadata_features
        self.model = self._build_model()
        self.history = None
        self._inference_fn = None

    def _get_inference_fn(self):
        """Build once and reuse."""
        if self._inference_fn is None:
            @tf.function(reduce_retracing=True)
            def _infer(X_temporal, X_metadata):
                return self.model([X_temporal, X_metadata], training=False)
            self._inference_fn = _infer
        return self._inference_fn

    def _build_model(self):
        """Build the meso-LSTM architecture."""
        temporal_input = layers.Input(shape=(self.sequence_length, self.n_temporal_features), name='temporal_sequence')

        x = layers.Conv1D(filters=64, kernel_size=7, padding='same', activation='relu')(temporal_input)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

        x = layers.Conv1D(filters=128, kernel_size=5, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling1D(pool_size=2)(x)

        x = layers.LSTM(128, return_sequences=True, dropout=0.2)(x)
        x = layers.LSTM(64, dropout=0.2)(x)

        metadata_input = layers.Input(shape=(self.n_metadata_features,), name='topic_metadata')
        y = layers.Dense(32, activation='relu')(metadata_input)
        y = layers.Dropout(0.2)(y)

        combined = layers.Concatenate()([x, y])

        z = layers.Dense(128, activation='relu')(combined)
        z = layers.Dropout(0.3)(z)
        z = layers.Dense(64, activation='relu')(z)
        z = layers.Dropout(0.2)(z)

        retention_7d = layers.Dense(1, activation='sigmoid', name='retention_7d')(z)
        retention_30d = layers.Dense(1, activation='sigmoid', name='retention_30d')(z)
        retention_90d = layers.Dense(1, activation='sigmoid', name='retention_90d')(z)

        model = models.Model(
            inputs=[temporal_input, metadata_input],
            outputs=[retention_7d, retention_30d, retention_90d]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            loss_weights={'retention_7d': 0.2, 'retention_30d': 0.3, 'retention_90d': 0.5},
            metrics=['mae']
        )

        return model

    def fit(self, X_temporal, X_metadata, y, X_val_temporal=None, X_val_metadata=None,
            y_val=None, epochs=50, batch_size=32, **kwargs):
        """Train the model."""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if X_val_temporal is not None else 'loss',
                patience=10,
                restore_best_weights=True,
                verbose=0
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if X_val_temporal is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=0
            )
        ]

        validation_data = None
        if X_val_temporal is not None and X_val_metadata is not None and y_val is not None:
            validation_data = ([X_val_temporal, X_val_metadata], y_val)

        self.history = self.model.fit(
            [X_temporal, X_metadata], y,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
            **kwargs
        )

        return self.history

    def predict(self, X_temporal, X_metadata):
        """Predict retention."""
        X_temporal = tf.convert_to_tensor(np.asarray(X_temporal, dtype=np.float32))
        X_metadata = tf.convert_to_tensor(np.asarray(X_metadata, dtype=np.float32))
        predictions = self._get_inference_fn()(X_temporal, X_metadata)
        return {
            'retention_7d': predictions[0].numpy(),
            'retention_30d': predictions[1].numpy(),
            'retention_90d': predictions[2].numpy()
        }

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
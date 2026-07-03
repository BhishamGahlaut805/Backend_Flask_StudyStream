"""Macro-LSTM Model - Long-term learning path optimization."""

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import logging

logger = logging.getLogger(__name__)


class LearningPathLSTM:
    """Generates optimal long-term learning sequences."""

    def __init__(self, encoder_units=256, decoder_units=256, n_topics=100):
        self.encoder_units = encoder_units
        self.decoder_units = decoder_units
        self.n_topics = n_topics
        self.model = self._build_model()
        self.history = None
        self._inference_fn = None

    def _get_inference_fn(self):
        """Build once and reuse."""
        if self._inference_fn is None:
            @tf.function(reduce_retracing=True)
            def _infer(encoder_input, decoder_input):
                return self.model([encoder_input, decoder_input], training=False)
            self._inference_fn = _infer
        return self._inference_fn

    def _build_model(self):
        """Build encoder-decoder LSTM with attention."""
        encoder_inputs = layers.Input(shape=(None, 20), name='encoder_input')
        encoder_lstm = layers.LSTM(
            self.encoder_units,
            return_state=True,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2
        )
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
        encoder_states = [state_h, state_c]

        decoder_inputs = layers.Input(shape=(None, 15), name='decoder_input')
        decoder_lstm = layers.LSTM(
            self.decoder_units,
            return_sequences=True,
            return_state=True,
            dropout=0.2,
            recurrent_dropout=0.2
        )
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)

        attention = layers.Attention()([decoder_outputs, encoder_outputs])
        decoder_concat = layers.Concatenate(axis=-1)([decoder_outputs, attention])

        topic_output = layers.TimeDistributed(
            layers.Dense(self.n_topics, activation='softmax'),
            name='next_topics'
        )(decoder_concat)

        retention_output = layers.TimeDistributed(
            layers.Dense(1, activation='sigmoid'),
            name='retention_targets'
        )(decoder_concat)

        model = models.Model(
            inputs=[encoder_inputs, decoder_inputs],
            outputs=[topic_output, retention_output]
        )

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss={'next_topics': 'categorical_crossentropy', 'retention_targets': 'mse'},
            loss_weights={'next_topics': 0.7, 'retention_targets': 0.3},
            metrics={'next_topics': ['accuracy'], 'retention_targets': ['mae']}
        )

        return model

    def fit(self, encoder_input, decoder_input, targets,
            val_encoder=None, val_decoder=None, val_targets=None,
            epochs=50, batch_size=32, **kwargs):
        """Train the model."""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss' if val_encoder is not None else 'loss',
                patience=10,
                restore_best_weights=True,
                verbose=0
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss' if val_encoder is not None else 'loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=0
            )
        ]

        validation_data = None
        if val_encoder is not None and val_decoder is not None and val_targets is not None:
            validation_data = ([val_encoder, val_decoder], val_targets)

        self.history = self.model.fit(
            [encoder_input, decoder_input], targets,
            validation_data=validation_data,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=0,
            **kwargs
        )

        return self.history

    def predict(self, encoder_input, decoder_input):
        """Generate learning path."""
        encoder_input = tf.convert_to_tensor(np.asarray(encoder_input, dtype=np.float32))
        decoder_input = tf.convert_to_tensor(np.asarray(decoder_input, dtype=np.float32))
        return self._get_inference_fn()(encoder_input, decoder_input)

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
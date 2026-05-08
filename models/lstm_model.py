"""
models/lstm_model.py
════════════════════════════════════════════════════════════════
Bidirectional LSTM for sequential race-form modelling.
Falls back to exponential-weighted average if TF unavailable.
════════════════════════════════════════════════════════════════
"""

import numpy as np
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks, regularizers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class LSTMRacePredictor:
    def __init__(self, seq_len=5, n_features=7, lstm_units=64, dropout=0.3):
        self.seq_len = seq_len
        self.n_features = n_features
        self.lstm_units = lstm_units
        self.dropout = dropout
        self.model = None
        self.is_fitted = False
        if TF_AVAILABLE:
            self._build()

    def _build(self):
        inp = layers.Input(shape=(self.seq_len, self.n_features))
        x = layers.Bidirectional(
            layers.LSTM(self.lstm_units, return_sequences=True,
                        kernel_regularizer=regularizers.l2(1e-4)))(inp)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.LSTM(self.lstm_units // 2, return_sequences=False,
                        kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dropout(self.dropout)(x)
        x = layers.Dense(32, activation="relu")(x)
        x = layers.Dropout(self.dropout / 2)(x)
        out = layers.Dense(1)(x)
        self.model = models.Model(inputs=inp, outputs=out)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="huber", metrics=["mae"]
        )

    def fit(self, X_seq, y, epochs=60, batch_size=64, validation_split=0.1, verbose=0):
        if not TF_AVAILABLE:
            print("[LSTM] TensorFlow unavailable – using fallback.")
            self.is_fitted = True
            return self
        cb = [
            callbacks.EarlyStopping(patience=12, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(patience=6, factor=0.5, min_lr=1e-5),
        ]
        self.model.fit(X_seq, y, epochs=epochs, batch_size=batch_size,
                       validation_split=validation_split, callbacks=cb, verbose=verbose)
        self.is_fitted = True
        return self

    def predict(self, X_seq) -> np.ndarray:
        if not TF_AVAILABLE or not self.is_fitted or self.model is None:
            return np.full(len(X_seq), 10.0, dtype=np.float32)
        return self.model.predict(X_seq, verbose=0).flatten()

    def save(self, path="models/saved/lstm_f1.keras"):
        if self.model:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self.model.save(path)
            print(f"[LSTM] Saved → {path}")

    @classmethod
    def load(cls, path="models/saved/lstm_f1.keras"):
        if not TF_AVAILABLE:
            obj = cls.__new__(cls)
            obj.model = None
            obj.is_fitted = False
            return obj
        obj = cls.__new__(cls)
        obj.model = tf.keras.models.load_model(path)
        s = obj.model.input_shape
        obj.seq_len, obj.n_features = s[1], s[2]
        obj.is_fitted = True
        return obj


class SimpleLSTMFallback:
    """Exponential-weighted fallback when TensorFlow is not installed."""
    def __init__(self, span=5):
        self.span = span
        self.is_fitted = True

    def fit(self, *args, **kwargs):
        return self

    def predict(self, X_seq) -> np.ndarray:
        alpha = 2 / (self.span + 1)
        weights = np.array([(1-alpha)**(self.span-1-i) * alpha for i in range(X_seq.shape[1])])
        weights /= weights.sum()
        pos_seq = X_seq[:, :, 0]
        return (pos_seq * weights[np.newaxis, :]).sum(axis=1)

    def save(self, *a, **kw): pass

    @classmethod
    def load(cls, *a, **kw): return cls()


def get_lstm_model(seq_len=5, n_features=7):
    if TF_AVAILABLE:
        return LSTMRacePredictor(seq_len=seq_len, n_features=n_features)
    return SimpleLSTMFallback()

"""Blank (TensorFlow / Keras): a small network on random data - replace the layers, the data and the training."""
import keras
import numpy as np

INPUT_SHAPE = (1, 16)                # one input: batch 1, 16 features


def model():
    return keras.Sequential([keras.Input((16,)), keras.layers.Dense(32, activation="relu"),
                             keras.layers.Dense(4, activation="softmax")])


def _data(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 16)).astype(np.float32)
    return x, x[:, :4].argmax(axis=1)                       # the class: the largest of the first four features


def build():
    keras.utils.set_random_seed(0)
    m = model()
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    m.fit(*_data(2000, 1), epochs=20, verbose=0)
    return m


def calibration():
    return [x[None] for x in _data(200, 2)[0]]


def test_inputs():
    return [x[None] for x in _data(5, 3)[0]]

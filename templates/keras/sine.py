"""Sine regression (TensorFlow / Keras): 1 -> 16 -> 16 -> 1 learning sin(x) on [0, 2 pi]; no download."""
import math

import keras
import numpy as np

INPUT_SHAPE = (1, 1)


def model():
    return keras.Sequential([keras.Input((1,)), keras.layers.Dense(16, activation="relu"),
                             keras.layers.Dense(16, activation="relu"), keras.layers.Dense(1)])


def build():
    keras.utils.set_random_seed(0)
    m = model()
    m.compile(optimizer=keras.optimizers.Adam(0.01), loss="mse")
    x = np.random.default_rng(1).random((4000, 1)).astype(np.float32) * 2 * math.pi
    m.fit(x, np.sin(x), epochs=60, batch_size=256, verbose=0)
    return m


def calibration():
    return [np.array([[v]], dtype=np.float32) for v in np.linspace(0, 2 * math.pi, 200)]


def test_inputs():
    return [np.array([[v]], dtype=np.float32) for v in (0.0, 0.5, 1.57, 3.14, 4.71, 6.2)]

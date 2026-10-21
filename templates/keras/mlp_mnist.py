"""MLP on MNIST (TensorFlow / Keras): 784 -> 64 -> 10 with a softmax; one epoch, MNIST downloaded by Keras."""
import keras
import numpy as np

INPUT_SHAPE = (1, 784)


def model():
    return keras.Sequential([keras.Input((784,)), keras.layers.Dense(64, activation="relu"),
                             keras.layers.Dense(10, activation="softmax")])


def _data(train):
    (x, y), (xt, yt) = keras.datasets.mnist.load_data()
    x, y = (x, y) if train else (xt, yt)
    return ((x.reshape(-1, 784).astype(np.float32) / 255.0 - 0.1307) / 0.3081), y


def build():
    keras.utils.set_random_seed(0)
    m = model()
    m.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    m.fit(*_data(True), epochs=1, batch_size=128, verbose=0)
    return m


def calibration():
    return [x[None] for x in _data(True)[0][0:20000:100]]


def test_inputs():
    return [x[None] for x in _data(False)[0][:5]]

"""LeNet-5 on MNIST (TensorFlow / Keras): two convolutions and three dense layers; one epoch, MNIST downloaded by
Keras.  Images are NHWC (28 x 28 x 1)."""
import keras
import numpy as np

INPUT_SHAPE = (1, 28, 28, 1)


def model():
    return keras.Sequential([
        keras.Input((28, 28, 1)),
        keras.layers.Conv2D(6, 5, padding="same", activation="relu"), keras.layers.MaxPooling2D(2),
        keras.layers.Conv2D(16, 5, activation="relu"), keras.layers.MaxPooling2D(2),
        keras.layers.Flatten(), keras.layers.Dense(120, activation="relu"), keras.layers.Dense(84, activation="relu"),
        keras.layers.Dense(10)])


def _data(train):
    (x, y), (xt, yt) = keras.datasets.mnist.load_data()
    x, y = (x, y) if train else (xt, yt)
    return ((x[..., None].astype(np.float32) / 255.0 - 0.1307) / 0.3081), y


def build():
    keras.utils.set_random_seed(0)
    m = model()
    m.compile(optimizer="adam", loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=["accuracy"])
    m.fit(*_data(True), epochs=1, batch_size=128, verbose=0)
    return m


def calibration():
    return [x[None] for x in _data(True)[0][0:20000:100]]


def test_inputs():
    return [x[None] for x in _data(False)[0][:5]]

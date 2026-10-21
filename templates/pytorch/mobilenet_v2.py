"""MobileNetV2 (PyTorch, pretrained): torchvision's ImageNet model (224 x 224 RGB, 1000 classes) with its published
weights; synthetic inputs (the SoC is checked against the host exactly; ImageNet accuracy is not measured).  Large:
needs DDR or a big memory, and hours in RTL simulation."""
import numpy as np

INPUT_SHAPE = (1, 3, 224, 224)


def model():
    import torchvision
    return torchvision.models.mobilenet_v2()


def build():
    import torchvision
    return torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1).eval()


def _inputs(n, seed):
    g = np.random.default_rng(seed)
    return [g.standard_normal(INPUT_SHAPE).astype(np.float32) for _ in range(n)]


def calibration():
    return _inputs(32, 0)


def test_inputs():
    return _inputs(2, 1)

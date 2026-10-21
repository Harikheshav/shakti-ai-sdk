"""resnet18: torchvision's ImageNet model (224x224 RGB, 1000 classes) with its published IMAGENET1K_V1 weights
(downloaded by torchvision, which checks the file's hash).  ImageNet itself is not used: calibration and test inputs
are synthetic (normalised noise from a fixed seed), so the target is checked against the host exactly, but int8
accuracy is not measured."""
import numpy as np
import torch

INPUT_SHAPE = (1, 3, 224, 224)


def build():
    import torchvision
    net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
    # the max pool's padding: onnx2tf pads a MaxPool with the type's minimum (PadV2), and int8 calibration then
    # gives that tensor a scale of ~1e36, which TFLite (Micro) refuses in the next Add.  The pool follows a ReLU
    # (inputs >= 0), so padding with zeros gives the same outputs.
    net.maxpool = torch.nn.Sequential(torch.nn.ZeroPad2d(1), torch.nn.MaxPool2d(3, 2))
    return net.eval()


def _inputs(n, seed):
    g = np.random.default_rng(seed)
    return [g.standard_normal(INPUT_SHAPE).astype(np.float32) for _ in range(n)]


def calibration():
    return _inputs(32, 0)


def test_inputs():
    return _inputs(2, 1)

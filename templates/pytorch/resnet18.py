"""ResNet-18 (PyTorch, pretrained): torchvision's ImageNet model (224 x 224 RGB, 1000 classes) with its published
weights; synthetic inputs (checked against the host exactly; ImageNet accuracy is not measured).  Large: needs DDR
or a big memory, and hours in RTL simulation."""
import numpy as np
import torch

INPUT_SHAPE = (1, 3, 224, 224)


def _zero_padded_pool(net):
    # the max pool's padding as zeros (it follows a ReLU, so the outputs are the same): TFLite's int8 conversion
    # otherwise pads with the type's minimum and gets an unusable scale
    net.maxpool = torch.nn.Sequential(torch.nn.ZeroPad2d(1), torch.nn.MaxPool2d(3, 2))
    return net


def model():
    import torchvision
    return _zero_padded_pool(torchvision.models.resnet18())


def build():
    import torchvision
    return _zero_padded_pool(torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)).eval()


def _inputs(n, seed):
    g = np.random.default_rng(seed)
    return [g.standard_normal(INPUT_SHAPE).astype(np.float32) for _ in range(n)]


def calibration():
    return _inputs(32, 0)


def test_inputs():
    return _inputs(2, 1)

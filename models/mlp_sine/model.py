"""mlp_sine: a small multilayer perceptron that learns sin(x) on [0, 2*pi] (the same task as TensorFlow Lite
Micro's hello_world, here defined once in PyTorch for every framework).  1 -> 16 -> 16 -> 1, ReLU.  Trained here
from a fixed seed in a few seconds; no download."""
import math

import numpy as np
import torch

INPUT_SHAPE = (1, 1)


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.ReLU(), torch.nn.Linear(16, 16),
                                     torch.nn.ReLU(), torch.nn.Linear(16, 1))

    def forward(self, x):
        return self.f(x)


def build():
    torch.manual_seed(0)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=0.01)
    g = torch.Generator().manual_seed(1)
    for _ in range(3000):
        x = torch.rand(256, 1, generator=g) * 2 * math.pi
        loss = torch.nn.functional.mse_loss(net(x), torch.sin(x))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net.eval()


def calibration():
    """Inputs for int8 calibration (representative data)."""
    return [np.array([[v]], dtype=np.float32) for v in np.linspace(0, 2 * math.pi, 200)]


def test_inputs():
    """The inputs the target runs; its outputs are compared with the host's."""
    return [np.array([[v]], dtype=np.float32) for v in (0.0, 0.5, 1.0, 1.57, 2.5, 3.14, 4.0, 4.71, 5.5, 6.2)]

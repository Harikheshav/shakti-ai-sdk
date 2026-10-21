"""Sine regression (PyTorch): 1 -> 16 -> 16 -> 1 learning sin(x) on [0, 2 pi]; seconds to train, no download."""
import math

import numpy as np
import torch

INPUT_SHAPE = (1, 1)


def model():
    return torch.nn.Sequential(torch.nn.Linear(1, 16), torch.nn.ReLU(), torch.nn.Linear(16, 16), torch.nn.ReLU(),
                               torch.nn.Linear(16, 1))


def build():
    torch.manual_seed(0)
    net = model()
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
    return [np.array([[v]], dtype=np.float32) for v in np.linspace(0, 2 * math.pi, 200)]


def test_inputs():
    return [np.array([[v]], dtype=np.float32) for v in (0.0, 0.5, 1.57, 3.14, 4.71, 6.2)]

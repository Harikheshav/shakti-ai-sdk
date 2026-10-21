"""Blank (PyTorch): a small network on random data - replace the layers, the data and the training."""
import numpy as np
import torch

INPUT_SHAPE = (1, 16)                # one input: batch 1, 16 features


def model():
    return torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 4))


def _data(n, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, 16)).astype(np.float32)
    return x, (x[:, :4].argmax(axis=1)).astype(np.int64)     # the class: the largest of the first four features


def build():
    torch.manual_seed(0)
    net = model()
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    x, y = (torch.from_numpy(a) for a in _data(2000, 1))
    for _ in range(200):
        loss = torch.nn.functional.cross_entropy(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net.eval()


def calibration():
    return [x[None] for x in _data(200, 2)[0]]


def test_inputs():
    return [x[None] for x in _data(5, 3)[0]]

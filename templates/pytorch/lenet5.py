"""LeNet-5 on MNIST (PyTorch): two convolutions and three dense layers; one epoch, MNIST downloaded by torchvision."""
import os

import torch

INPUT_SHAPE = (1, 1, 28, 28)
CACHE = os.environ.get("SHAKTI_AI_CACHE", os.path.expanduser("~/.cache/shakti-ai-sdk"))


def model():
    return torch.nn.Sequential(
        torch.nn.Conv2d(1, 6, 5, padding=2), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Conv2d(6, 16, 5), torch.nn.ReLU(), torch.nn.MaxPool2d(2),
        torch.nn.Flatten(), torch.nn.Linear(16 * 5 * 5, 120), torch.nn.ReLU(), torch.nn.Linear(120, 84),
        torch.nn.ReLU(), torch.nn.Linear(84, 10))


def _data(train):
    import torchvision
    t = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize((0.1307,), (0.3081,))])
    return torchvision.datasets.MNIST(os.path.join(CACHE, "mnist"), train=train, download=True, transform=t)


def build():
    torch.manual_seed(0)
    net = model()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loader = torch.utils.data.DataLoader(_data(True), batch_size=128, shuffle=True,
                                         generator=torch.Generator().manual_seed(0))
    for x, y in loader:
        loss = torch.nn.functional.cross_entropy(net(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net.eval()


def calibration():
    d = _data(True)
    return [d[i][0].unsqueeze(0).numpy() for i in range(0, 20000, 100)]


def test_inputs():
    d = _data(False)
    return [d[i][0].unsqueeze(0).numpy() for i in range(5)]

"""mlp_mnist: a multilayer perceptron classifying MNIST digits into class probabilities (softmax output):
784 -> 64 -> 10, ReLU, softmax.  Trained here for one epoch from a fixed seed; MNIST is
downloaded by torchvision (its own md5 check) into SHAKTI_AI_CACHE/mnist, as for lenet5."""
import os

import torch

INPUT_SHAPE = (1, 784)
CACHE = os.environ.get("SHAKTI_AI_CACHE", os.path.expanduser("~/.cache/shakti-ai-sdk"))


class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Linear(784, 64), torch.nn.ReLU(), torch.nn.Linear(64, 10),
                                     torch.nn.Softmax(dim=1))

    def forward(self, x):
        return self.f(x)


def _data(train):
    import torchvision
    t = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                        torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                                        torchvision.transforms.Lambda(lambda x: x.reshape(784))])
    return torchvision.datasets.MNIST(os.path.join(CACHE, "mnist"), train=train, download=True, transform=t)


def build():
    torch.manual_seed(0)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    loader = torch.utils.data.DataLoader(_data(True), batch_size=128, shuffle=True,
                                         generator=torch.Generator().manual_seed(0))
    net.train()
    for x, y in loader:
        loss = torch.nn.functional.nll_loss(torch.log(net(x) + 1e-9), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    test = _data(False)
    with torch.no_grad():
        right = sum(int(net(x.unsqueeze(0)).argmax()) == y for x, y in (test[i] for i in range(1000)))
    print(f"mlp_mnist: {right / 10:.1f} % of 1000 MNIST test digits")
    return net


def calibration():
    d = _data(True)
    return [d[i][0].unsqueeze(0).numpy() for i in range(0, 20000, 100)]


def test_inputs():
    d = _data(False)
    return [d[i][0].unsqueeze(0).numpy() for i in range(10)]

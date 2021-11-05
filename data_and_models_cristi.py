import torchvision
import torch.nn as nn
import torch.nn.functional as F

def get_mnist_dataset(train):
    return torchvision.datasets.MNIST(root='./data', train=train, download=True,
                                 transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                     (0.1307,), (0.3081,))
                                 ]))

class mnist_conv_model(nn.Module):
    def __init__(self):
        super(mnist_conv_model, self).__init__()
        self.conv1 = nn.Conv2d(1, 5, kernel_size=5)
        self.conv2 = nn.Conv2d(5, 2, kernel_size=5)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(800, 50)
        self.fc2 = nn.Linear(50, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return self.softmax(x)
import torch
from torch import nn

from config import DEVICE


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.flatten = nn.Flatten()

        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),    # input layer: 28 x 28 = 784 neurons
            nn.ReLU(),                  # activation function hidden layer: Rectified Linear Unit
            nn.Linear(512, 512),        # hidden layer: 512 neurons
            nn.ReLU(),                  # activation function output layer: Rectified Linear Unit
            nn.Linear(512, 10),         # output layer: 10 neurons
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.linear_relu_stack(x)


def create_model(device=DEVICE):
    return NeuralNetwork().to(device)


def create_training_components(model):
    loss_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=1e-3,
    )

    return loss_fn, optimizer
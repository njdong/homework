import torch
import torch.nn as nn
from typing import Callable


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        hidden_count: int = 1,
        activation: Callable = nn.ReLU,
        initializer: Callable = nn.init.ones_,
    ) -> None:
        """
        Initialize the MLP.

        Arguments:
            input_size: The dimension D of the input data.
            hidden_size: The number of neurons H in the hidden layer.
            num_classes: The number of classes C.
            activation: The activation function to use in the hidden layer.
            initializer: The initializer to use for the weights.
        """
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.hidden_count = hidden_count
        self.activation = activation

        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_size)])

        for i in range(hidden_count - 1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.layers.append(nn.Linear(hidden_size, num_classes))

        for layer in self.layers:
            initializer(layer.weight)

    def forward(self, x):
        """
        Forward pass of the network.

        Arguments:
            x: The input data.

        Returns:
            The output of the network.
        """
        for layer in self.layers[:-1]:
            x = self.activation()(layer(x))

        x = self.layers[-1](x)

        return x

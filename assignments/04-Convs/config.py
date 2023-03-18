from typing import Callable
import torch
import torch.optim
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor


class CONFIG:
    """
    larger batch_size to enable faster network
    """

    batch_size = 200
    num_epochs = 5

    optimizer_factory: Callable[
        [nn.Module], torch.optim.Optimizer
    ] = lambda model: torch.optim.Adam(model.parameters(), lr=5e-3)

    transforms = Compose([ToTensor()])

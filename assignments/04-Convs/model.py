import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(torch.nn.Module):
    """
    CNN
    """

    def __init__(self, num_channels, num_classes) -> None:
        """
        CNN
        """
        super().__init__()

        self.conv1 = nn.Conv2d(num_channels, 8, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16 * 8 * 8, num_classes)

    def forward(self, x) -> None:
        """
        CNN
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 16 * 8 * 8)
        x = self.fc1(x)
        return x

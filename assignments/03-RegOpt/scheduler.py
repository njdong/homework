from typing import List

from torch.optim.lr_scheduler import _LRScheduler

import numpy as np


class CustomLRScheduler(_LRScheduler):
    def __init__(self, optimizer, step_size=2, gamma=0.005, last_epoch=-1):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        # ... Your Code Here ...
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)

        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.last_epoch = last_epoch

    def get_lr(self):
        """
        getting learning rate
        """

        if self.last_epoch == 0:
            return [lr for lr in self.base_lrs]

        decay = np.exp((-1) * self.gamma)

        return [lr * decay for lr in self.base_lrs]

    # def get_lr(self) -> List[float]:
    # Note to students: You CANNOT change the arguments or return type of
    # this function (because it is called internally by Torch)

    # ... Your Code Here ...
    # Here's our dumb baseline implementation:


#
# return [lr for lr in self.scheduler.get_last_lr()]

# return [i for i in self.base_lrs]

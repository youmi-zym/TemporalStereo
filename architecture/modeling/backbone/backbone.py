from abc import ABCMeta, abstractmethod
import torch.nn as nn

__all__ = ["Backbone"]

class Backbone(nn.Module, metaclass=ABCMeta):
    """
    Abstract base class for network backbones.
    """

    def __init__(self):
        """
        The `__init__` method of any subclass can specify its own set of arguments.
        """
        super().__init__()

    @abstractmethod
    def forward(self, *inputs):
        """
        Subclasses must override this method, but adhere to the same return type.
        """
        pass
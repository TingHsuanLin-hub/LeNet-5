import numpy as np
from handcrafted_nn.tensor import tensor

class parameter(tensor):
    def __init__(self, matrix=..., trainable=True):
        super().__init__(matrix, trainable)

    def __call__(self):
        return super().__call__()

    def __add__(self, o):
        return super().__add__(o)
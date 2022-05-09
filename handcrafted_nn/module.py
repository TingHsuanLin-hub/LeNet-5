from abc import abstractclassmethod
import numpy as np
from handcrafted_nn.parameter import parameter
from handcrafted_nn.tensor import tensor

class module(tensor):
    def __init__(self, matrix=..., trainable=False):
        super().__init__(matrix, trainable)

    def __call__(self,x):
        return self.forward(x)

    @abstractclassmethod
    def forward(self,x):
        ...

    def backward(self):
        self.input.append(parameter(self.matrix))
        for index, layer in enumerate(self.input.reverse()):
            if index == 0:
                dZ = layer()
            else:
                dZ = pre_grad
            pre_grad = layer.backward(dZ)

    def update(self):
        for index, layer in enumerate(self.input.reverse()):
            if layer.trainable:
                layer.update()
            else:
                pass


import numpy as np
class tensor:
    def __init__(self, matrix=np.array([]), trainable=False):
        self.trainable = trainable
        self.matrix = matrix
        self.matrix_grad = np.zeros_like(matrix)
        self.input = []
        self.shape = self.matrix.shape

    def __call__(self):
        return self.matrix

    def __add__(self, o):
        if self.trainable==True or o.trainable == True:
            return tensor(self.matrix+o.matrix, trainable=True)
        else:
            return tensor(self.matrix+o.matrix)

    def __sub__(self, o):
        if self.trainable==True or o.trainable == True:
            return tensor(self.matrix - o.matrix, trainable=True)
        else:
            return tensor(self.matrix - o.matrix)

    def __mul__(self, o):
        if self.trainable==True or o.trainable == True:
            return tensor(self.matrix*o, trainable=True)
        else:
            return tensor(self.matrix*o)

    def multiply_constant(self, c):
        self.matrix = self.matrix*c
        return self

if __name__ == "__main__":
    a = tensor()
    a = a()
    
    print(a)

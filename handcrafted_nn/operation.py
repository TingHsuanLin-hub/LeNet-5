from abc import abstractmethod
import numpy as np
from handcrafted_nn.tensor import tensor
from handcrafted_nn.parameter import parameter
from handcrafted_nn.pure_data import pure_data

class operation(tensor):
    def __init__(self):
        super().__init__()
        pass

    def __call__(self,x):
        # self.input.append(x.input)
        # self.input.append(self)
        return self.forward(x)
    
    @abstractmethod
    def forward():
        ...

    @abstractmethod
    def backward():
        ...

class conv2D(operation):
    def __init__(self, filter_size=(3,3),in_channel = 1, out_channel = 1, stride = 1, padding=0):
        super().__init__()
        self.b = parameter(matrix=np.random.rand(1,1,1, out_channel)/10)
        self.W = parameter(matrix=np.random.rand(in_channel,filter_size[0], filter_size[1],  out_channel)/10)
        self.db = parameter(matrix=np.zeros([1,1,1, out_channel]))
        self.dW = parameter(matrix=np.zeros([in_channel,filter_size[0], filter_size[1], out_channel]))
        self.stride = stride
        self.padding = padding
        self.v_w = parameter(matrix=np.zeros([in_channel,filter_size[0], filter_size[1], out_channel]))
        self.v_b = parameter(matrix=np.zeros([1,1,1, out_channel]))

    def conv_single_step(self, A, W, b):
        s = np.multiply(A, W)
        z = np.sum(s)
        z = z + b.astype(float)
        return z

    def zero_pad(self, X, pad):
        X_pad = np.pad(X, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant', constant_values = (0,0))
        return X_pad
    
    def forward(self, input_A):
        self.A = input_A
        input_A = input_A.matrix
        (batch, n_C_prev, n_H_prev, n_W_prev) = input_A.shape
        # Retrieve dimensions from W's shape (≈1 line)
        (n_C_prev, f, f, n_C) = self.W.shape
        
        # Compute the dimensions of the CONV output volume using the formula given above. Hint: use int() to floor. (≈2 lines)
        n_H = int((n_H_prev + 2*self.padding - f)/self.stride) + 1
        n_W =int((n_W_prev + 2*self.padding - f)/self.stride) + 1
        
        # Initialize the output volume Z with zeros. (≈1 line)
        Z = np.zeros([batch,n_C, n_H, n_W])
        # Create A_prev_pad by padding A_prev
        A_prev_pad = self.zero_pad(input_A, self.padding)
        
        for i in range(batch):                               # loop over the batch of training examples
            a_prev_pad = A_prev_pad[i,:,:,:]                              # Select ith training example's padded activation
            for h in range(n_H):                           # loop over vertical axis of the output volume
                for w in range(n_W):                       # loop over horizontal axis of the output volume
                    for c in range(n_C):                   # loop over channels (= #filters) of the output volume
                        
                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = h*self.stride
                        vert_end = h*self.stride + f
                        horiz_start = w*self.stride 
                        horiz_end = w*self.stride + f
                        
                        # Use the corners to define the (3D) slice of a_prev_pad (See Hint above the cell). (≈1 line)
                        a_slice_prev = a_prev_pad[:,vert_start:vert_end,horiz_start:horiz_end]
                        
                        
                        # Convolve the (3D) slice with the correct filter W and bias b, to get back one output neuron. (≈1 line)
                        Z[i,c, h, w] = self.conv_single_step(a_slice_prev, self.W.matrix[:, :, :, c], self.b.matrix[:,:,:,c])
                                            
        ### END CODE HERE ###
        
        # Making sure your output shape is correct
        assert(Z.shape == (batch, n_C, n_H, n_W))
        self.matrix = Z
        return parameter(Z)

    def backward(self, up_grad):
        # Retrieve dimensions from A_prev's shape
        (batch, n_C_prev, n_H_prev, n_W_prev) = self.A.shape
        
        # Retrieve dimensions from W's shape
        (n_C_prev, f, f, n_C) = self.W.shape
        
  
        # Retrieve dimensions from dZ's shape
        (batch, n_C, n_H, n_W) = up_grad.shape
        
        # Initialize dA_prev, dW, db with the correct shapes
        dA_prev = np.zeros((batch, n_C_prev, n_H_prev, n_W_prev))                           
        dW = np.zeros((n_C_prev, f, f, n_C))
        db = np.zeros((1, 1, 1, n_C))

        # Pad A_prev and dA_prev

        A_prev_pad = self.zero_pad(self.A.matrix, self.padding)
        dA_prev_pad = self.zero_pad(dA_prev, self.padding)
        
        for i in range(batch):                       # loop over the training examples
            
            # select ith training example from A_prev_pad and dA_prev_pad
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]
            
            for h in range(n_H):                   # loop over vertical axis of the output volume
                for w in range(n_W):               # loop over horizontal axis of the output volume
                    for c in range(n_C):           # loop over the channels of the output volume
                        
                        # Find the corners of the current "slice"
                        vert_start = h
                        vert_end = vert_start + f
                        horiz_start = w
                        horiz_end = horiz_start + f
                        
                        # Use the corners to define the slice from a_prev_pad
                        a_slice = a_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end]

                        # Update gradients for the window and the filter's parameters using the code formulas given above
                        da_prev_pad[:, vert_start:vert_end, horiz_start:horiz_end] += self.W.matrix[:,:,:,c] * up_grad.matrix[i, c, h, w]
                        dW[:,:,:,c] += a_slice * up_grad.matrix[i, c, h, w]
                        db[:,:,:,c] += up_grad.matrix[i, c, h, w]
                        
            # Set the ith training example's dA_prev to the unpaded da_prev_pad (Hint: use X[pad:-pad, pad:-pad, :])
            if self.padding == 0:
                dA_prev[i, :, :, :] = da_prev_pad
            else:
                dA_prev[i, :, :, :] = da_prev_pad[:, self.padding:-self.padding, self.padding:-self.padding]
        ### END CODE HERE ###
        
        # Making sure your output shape is correct
        assert(dA_prev.shape == (batch, n_C_prev, n_H_prev, n_W_prev))
        self.dA = parameter(dA_prev)
        self.db = self.db + parameter(db)
        self.dW = self.dW + parameter(dW)

        return self.dA

    def update(self,lr, beta = 1):

        self.A  = self.A - self.dA.multiply_constant(lr)
        self.dA = parameter(matrix=np.array([]))
        self.b = self.b - self.db.multiply_constant(lr)
        self.db = parameter(matrix=np.zeros_like(self.db.matrix))
        self.W = self.W - self.dW.multiply_constant(lr)
        self.dW = parameter(matrix=np.zeros_like(self.dW.matrix))
        return 

class pooling_2D(operation):
    def __init__(self,filter = 2, stride = 2, mode='max'):
        super().__init__()
        self.mode = mode
        self.f = filter
        self.stride = stride

    def create_mask_from_window(self,x):
        mask = x == np.max(x)
        return mask

    def distribute_value(self, dz, shape):
        (n_H, n_W) = shape
        average = dz / (n_H * n_W)
        a = np.ones(shape) * average
        
        return a

    def forward(self, input_A):
        self.A = input_A
        (m, n_C_prev, n_H_prev, n_W_prev) = input_A.shape
        
        # Define the dimensions of the output
        n_H = int(1 + (n_H_prev - self.f) / self.stride)
        n_W = int(1 + (n_W_prev - self.f) / self.stride)
        n_C = n_C_prev
        
        # Initialize output matrix A
        A = np.zeros((m, n_C, n_H, n_W))              
        
        ### START CODE HERE ###
        for i in range(m):                         # loop over the training examples
            for h in range(n_H):                     # loop on the vertical axis of the output volume
                for w in range(n_W):                 # loop on the horizontal axis of the output volume
                    for c in range (n_C):            # loop over the channels of the output volume
                        
                        # Find the corners of the current "slice" (≈4 lines)
                        vert_start = h*self.stride
                        vert_end = h*self.stride +self.f
                        horiz_start = w*self.stride
                        horiz_end = w*self.stride + self.f
                        
                        # Use the corners to define the current slice on the ith training example of A_prev, channel c. (≈1 line)
                        a_prev_slice = input_A.matrix[i, c, vert_start:vert_end, horiz_start:horiz_end]
                        
                        # Compute the pooling operation on the slice. Use an if statment to differentiate the modes. Use np.max/np.mean.
                        if self.mode == "max":
                            A[i, c, h, w] = np.max(a_prev_slice)
                        elif self.mode == "average":
                            A[i, c, h, w] = np.mean(a_prev_slice)
        
        ### END CODE HERE ###
        
        # Making sure your output shape is correct
        assert(A.shape == (m, n_C, n_H, n_W))
        self.matrix = A
        return parameter(A)

    def backward(self, up_grad):
        m,n_C_prev, n_H_prev, n_W_prev = self.A.shape
        m, n_C, n_H, n_W = up_grad.shape
        dA_prev = np.zeros(self.A.shape)
        
        for i in range(m):                       # loop over the training examples
            a_prev = self.A()[i]
            for h in range(n_H):                   # loop on the vertical axis
                for w in range(n_W):               # loop on the horizontal axis
                    for c in range(n_C):           # loop over the channels (depth)
                        vert_start = h
                        vert_end = vert_start + self.f
                        horiz_start = w
                        horiz_end = horiz_start + self.f
                        # Compute the backward propagation in both modes.
                        if self.mode == "max":
                            a_prev_slice = a_prev[c, vert_start:vert_end, horiz_start:horiz_end]
                            mask = self.create_mask_from_window(a_prev_slice)
                            dA_prev[i, c, vert_start:vert_end, horiz_start:horiz_end] += np.multiply(mask, up_grad.matrix[i, c, h, w])
                            
                        elif self.mode == "average":
                            da = up_grad.matrix[i, c, h, w]
                            shape = (self.f, self.f)
                            dA_prev[i, c, vert_start:vert_end, horiz_start:horiz_end] += self.distribute_value(da, shape)
                            
        ### END CODE ###
        
        # Making sure your output shape is correct
        assert(dA_prev.shape == self.A.shape)
        self.dA = parameter(matrix=dA_prev)
        return self.dA

    def update(self, lr = 1):
        self.A = self.A - self.dA.multiply_constant(lr)
        self.dA = parameter(matrix=np.array([]))
        return 

class relu(operation):
    def __init__(self):
        super().__init__()

    def forward(self, input_A):
        self.A = input_A
        self.matrix = np.maximum(0,input_A.matrix)
        return parameter(self.matrix)

    def backward(self, up_grad):
        self.dA = parameter(np.multiply(np.where(self.A.matrix < 0, 0,1), up_grad.matrix))
        return self.dA

    def update(self,lr = 1):
        self.A = self.A - self.dA.multiply_constant(lr)
        self.dA = parameter(matrix=np.array([]))
        return

class fully_connection(operation):
    def __init__(self, in_num, out_num):
        super().__init__()
        self.W = parameter(matrix=np.random.rand(in_num, out_num)/100)
        self.b = parameter(matrix=np.random.rand(out_num)/100)
        self.dW = parameter(matrix=np.zeros((in_num, out_num)))
        self.db = parameter(matrix=np.zeros(out_num))

    def forward(self, input_A):
        self.A = input_A
        batch = input_A.shape[0]
        reshape_input = input_A.matrix.reshape(batch, -1)
        Z = np.dot(reshape_input, self.W.matrix) + self.b.matrix.T
        self.matrix = Z
        return parameter(Z)

    def backward(self, up_grad):
        batch = self.A.shape[0]
        self.dA = parameter(np.dot(up_grad.matrix, self.W.matrix.T).reshape(self.A.shape))
        self.dW = self.dW + parameter(matrix=np.dot(self.A.matrix.reshape(batch,-1).T,up_grad.matrix))
        self.db = self.db + parameter(matrix=np.sum(up_grad.matrix, axis=0))
        return self.dA

    def update(self,lr):
        self.A  = self.A - self.dA.multiply_constant(lr)
        self.dA = parameter(matrix=np.array([]))
        self.b = self.b - self.db.multiply_constant(lr)
        self.db = parameter(matrix=np.zeros_like(self.db))
        self.W = self.W - self.dW.multiply_constant(lr)
        self.dW = parameter(matrix=np.zeros_like(self.dW.matrix))
        return 

class softmax_crossentropy(operation):
    def __init__(self):
        super().__init__()

    def __call__(self, input_A, target):
        self.A = input_A
        self.target = target
        return super().__call__(input_A)

    def softmax(self,x):
        x = x.matrix
        for batch in range(x.shape[0]):
            sum = np.sum(np.exp(x[batch].astype('float')))
            x[batch,:] = np.exp(x[batch,:].astype('float'))/sum
        return x

    def forward(self,input_A):
        loss = -np.sum(np.multiply(self.target.matrix==1, np.log(self.softmax(input_A).astype('float'))))
        self.matrix = loss
        return parameter(loss)

    def backward(self):
        da_prev = self.A - self.target
        self.dA =da_prev
        return self.dA

    def update(self, lr):
        self.A = self.A - self.dA.multiply_constant(lr)
        self.dA = parameter(matrix=np.array([0]))
        return



if __name__ == "__main__":
    a = conv2D()
    b = pure_data(matrix=np.random.rand(2,3))
    target = pure_data(matrix=np.array([[1,0,0],[0,1,0]]))
    pool = pooling_2D()
    relu1 = relu()
    fc = fully_connection(9, 10)
    loss_fn = softmax_crossentropy()
    print(loss_fn(b,target).matrix)
    print(loss_fn.backward().shape)
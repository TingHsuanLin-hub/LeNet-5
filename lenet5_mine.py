from copyreg import pickle
from handcrafted_nn.operation import conv2D, pooling_2D, relu, fully_connection,softmax_crossentropy
from handcrafted_nn.data_load import read_mnist, img_resize, one_hot, dataloader
import numpy as np
import time

class LeNet5():
    def __init__(self):
        self.conv1 = conv2D(filter_size=(5,5), in_channel=1, out_channel=6)
        self.relu1 = relu()
        self.pool1 = pooling_2D(filter=2,stride=2)
        self.conv2 = conv2D(filter_size=(5,5), in_channel=6, out_channel=16)
        self.relu2 = relu()
        self.pool2 = pooling_2D(filter=2,stride=2)
        self.fc1 = fully_connection(16*5*5, 120)
        self.relu3 = relu()
        self.fc2 = fully_connection(120, 84)
        self.relu4 = relu()
        self.fc3 = fully_connection(84, 10)
        self.loss_fn = softmax_crossentropy()

    def forward(self,x,target):
        x = self.conv1(x)
        x = self.relu1(x) # sigmoid previously
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        # x = x.matrix.reshape(-1, 16*5*5)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        out = self.loss_fn(x, target)
        self.out =out
        return x.matrix, out.matrix

    def backward(self):
        self.out = self.loss_fn.backward()
        self.out = self.fc3.backward(self.out)
        self.out = self.relu4.backward(self.out)
        self.out = self.fc2.backward(self.out)
        self.out = self.relu3.backward(self.out)
        self.out = self.fc1.backward(self.out)
        self.out = self.pool2.backward(self.out)
        self.out = self.relu2.backward(self.out)
        self.out = self.conv2.backward(self.out)
        self.out = self.pool1.backward(self.out)
        self.out = self.relu1.backward(self.out)
        dx = self.conv1.backward(self.out)
        return dx

    def update(self, lr):
        self.loss_fn.update(lr)
        self.fc3.update(lr)
        self.relu4.update(lr)
        self.fc2.update(lr)
        self.relu3.update(lr)
        self.fc1.update(lr)
        self.pool2.update(lr)
        self.relu2.update(lr)
        self.conv2.update(lr)
        self.pool1.update(lr)
        self.relu1.update(lr)

    def save_model(self, path = "./", name = "checkpoint"):
        dict = {
            0 : self.conv1,
            1 : self.relu1,
            2 : self.pool1,
            3 : self.conv2,
            4 : self.relu2,
            5 : self.pool2,
            6 : self.fc1,
            7 : self.relu3,
            8 : self.fc2,
            9 : self.relu4,
            10 : self.fc3,
            11: self.loss_fn
        }
        file = open(path+name+".pkl", "wb")
        pickle.dump(dict, file)
        file.close()



if __name__ == "__main__":
    batch_size = 100
    epoch = 500
    model = LeNet5()
    img, label = read_mnist("../data/train-images.idx3-ubyte", "../data/train-labels.idx1-ubyte")
    img = img_resize(img,(32,32))
    label = one_hot(label,10)
    train_loader = dataloader(img, label, batch_size=batch_size)
    img, label = read_mnist("../data/t10k-images.idx3-ubyte", "../data/t10k-labels.idx1-ubyte",train=False)
    img = img_resize(img,(32,32))
    label = one_hot(label,10)
    test_loader = dataloader(img, label, batch_size=batch_size)
    
    for epo in range(epoch):
        running_loss = 0.0
        train_correct = 0.0
        train_num = 0
        for batch_idx, (x_train, label_train) in enumerate(train_loader, start=0):
            start_time = time.time()
            outputs, loss = model.forward(x_train, label_train)
            predict = np.argmax(outputs,axis=1)
            train_num += label_train.shape[0]
            train_correct += (predict == np.argmax(label_train.matrix,axis=1)).sum().item()
            model.backward()
            model.update(lr=1e-3)
            running_loss += loss
            training_time = time.time()- start_time
            print("Average Training Time:{:.3f} \t Train Batch {} \t , Training loss: {:.3f}, Training accuracy: {:.3f}%".format(training_time, batch_idx, running_loss/train_num, 100*(train_correct/train_num)))

        print("Train Epoch {} \t , Training loss: {:.3f}, Training accuracy: {:.3f}%".format(epo, running_loss, 100*(train_correct/train_num)))

        if epo % 5 == 0:
            test_correct = 0.0
            test_num = 0
            test_loss = 0.0
            start_time = time.time()
            for batch_idx, (x_test, label_test) in enumerate(test_loader):
                outputs, loss = model.forward(x_test, label_test)
                predict = np.argmax(outputs, axis=1)
                test_num += label_test.shape[0]
                test_correct += (predict == np.argmax(label_test.matrix, axis=1)).sum().item()
                test_loss += loss
            val_time = time.time()- start_time
            print("Average Testing Time:{:.3f} \t Test Epoch {} \t , Testing loss: {:.3f}, Testing accuracy: {:.3f}%".format(val_time, epo, test_loss, 100*(test_correct/test_num)))
            
            if best_acc < (test_correct/test_num):
                best_acc = (test_correct/test_num)
                model.save_model(path="./checkpoint/handcrafted/",name = 'epoch_%d_acc_%.3f.pth' %(epo, (test_correct/test_num)))


        if epo % 10 == 0 and epo > 1:
            model.save_model(path="./checkpoint/handcrafted/",name = 'epoch_%d_acc_%.3f.pth' %(epo, (test_correct/test_num)))
        

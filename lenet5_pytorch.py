from matplotlib import transforms
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.optim as optim
import os
import time
from tensorboardX import SummaryWriter 

from thop import profile
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x) # sigmoid previously
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = x.view(-1, 16*5*5)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x,dim=1) #dimension 0 is the batch size
        return output

def trainer():
    epoch = 500
    best_acc = 0.0
    pipeline_train = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
    pipeline_test = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
    train_set = datasets.MNIST(root='./data',train=True, download=True,transform=pipeline_train)
    test_set = datasets.MNIST(root='./data',train=False, download=True,transform=pipeline_test)
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    model = LeNet5().to(device)
    model.train()
    crossloss = torch.nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),lr=1e-3,momentum=0.9)
    if not os.path.isdir('checkpoint/pytorch'):
        os.mkdir('checkpoint/pytorch')
    writer = SummaryWriter('log/pytorch')
    for epo in range(epoch):
        running_loss = 0.0
        train_correct = 0.0
        train_num = 0
        for batch_idx, (x_train, label_train) in enumerate(train_loader, start=0):
            optimizer.zero_grad()
            x_train= x_train.to(device)
            label_train = label_train.to(device)
            outputs = model(x_train)
            loss = crossloss(outputs, label_train)
            predict = outputs.argmax(dim=1)
            train_num += label_train.size(0)
            train_correct += (predict == label_train).sum().item()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print("Train Epoch {} \t , Training loss: {:.3f}, Training accuracy: {:.3f}%".format(epo, running_loss, 100*(train_correct/train_num)))
        writer.add_scalar('Training loss', running_loss, epo)
        writer.add_scalar("Training accuracy", train_correct/train_num, epo)

        if epo % 5 == 0:
            test_correct = 0.0
            test_num = 0
            test_loss = 0.0
            with torch.no_grad():
                for batch_idx, (x_test, label_test) in enumerate(test_loader):
                    model.eval()
                    x_test= x_test.to(device)
                    label_test = label_test.to(device)
                    outputs = model(x_test)
                    test_loss += crossloss(outputs, label_test).item()
                    predict = outputs.argmax(dim=1)
                    test_num += label_test.size(0)
                    test_correct += (predict == label_test).sum().item()
                print("Test Epoch {} \t , Testing loss: {:.3f}, Testing accuracy: {:.3f}%".format(epo, test_loss, 100*(test_correct/test_num)))
                writer.add_scalar("Validation accuracy", test_correct/test_num, epo)
                writer.add_scalar("Validation loss", test_loss, epo)
            
            if best_acc < (test_correct/test_num):
                best_acc = (test_correct/test_num)
                torch.save(model.state_dict(), 'checkpoint/pytorch/epoch_%d_acc_%.3f.pth' %(epo, (test_correct/test_num)))

            model.train()

        if epo % 20 == 0 and epo > 1:
            torch.save(model.state_dict(), 'checkpoint/pytorch/epoch_%d_acc_%.3f.pth' %(epo, (test_correct/test_num)))


def tester(pretrained_model):
    pipeline_test = transforms.Compose([transforms.Resize((32,32)),transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
    test_set = datasets.MNIST(root='./data',train=False, download=False,transform=pipeline_test)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=False)
    model = LeNet5().to(device)
    state_dict = torch.load(pretrained_model, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    test_num = 0
    test_correct = 0.0
    with torch.no_grad():
        start_time = time.time()
        for batch_idx, (x_test, label_test) in enumerate(test_loader):
            x_test= x_test.to(device)
            label_test = label_test.to(device)
            outputs = model(x_test)
            predict = outputs.argmax(dim=1)
            test_num += label_test.size(0)
            test_correct += (predict == label_test).sum().item()
        infer_time = time.time()-start_time
        infer_time = infer_time/test_num
        print("Average Inference Time:{:.3f}, Testing accuracy: {:.3f}%".format(infer_time, 100*(test_correct/test_num)))   


if __name__ == "__main__":
    trainer()
    # model = LeNet5()
    # x = torch.ones((1,1,1,32,32))
    # flops, params = profile(model, inputs=x)
    # print('FLOPs = ' + str(flops/1000**2) + 'M')
    # print('Params = ' + str(params) )


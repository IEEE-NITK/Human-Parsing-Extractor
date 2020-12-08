import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch import cuda
import numpy as np
import time
import matplotlib.pyplot as plt

device = 'cuda' if cuda.is_available() else 'cpu'
batch_size = 64
mnist_train = MNIST('mnist', train=True, download=True, transform=ToTensor())
train_data_loader = T.utils.data.DataLoader(mnist_train, batch_size=64, shuffle = True)
mnist_test = MNIST('mnist', train=False, download=True, transform=ToTensor())
test_data_loader = T.utils.data.DataLoader(mnist_test, batch_size=64, shuffle = True)

class InceptionA(nn.Module):

    def __init__(self, ip):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(ip, 16, kernel_size=1)
        self.branch5x5_1 = nn.Conv2d(ip, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)
        self.branch3x3dbl_1 = nn.Conv2d(ip, 16, kernel_size=1)
        self.branch3x3dbl_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3dbl_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
        self.branch_pool = nn.Conv2d(ip, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3dbl = self.branch3x3dbl_1(x)
        branch3x3dbl = self.branch3x3dbl_2(branch3x3dbl)
        branch3x3dbl = self.branch3x3dbl_3(branch3x3dbl)
        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)
        op = [branch1x1, branch5x5, branch3x3dbl, branch_pool]
        op = T.cat(op, 1)
        return op

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)
        self.incept1 = InceptionA(ip=10)
        self.incept2 = InceptionA(ip=20)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(1408, 10)
    def forward(self, x):
        input_size = x.size(0)
        x = self.conv1(x)
        x = self.mp(x)
        x = F.relu(x)
        x = self.incept1(x)
        x = self.conv2(x)
        x = self.mp(x)
        x = F.relu(x)
        x = self.incept2(x)
        x = x.view(input_size, -1)  # flatten the tensor
        x = self.fc(x)
        x = F.log_softmax(x)
        return x


model = CNN()
l = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
model.to(device)
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_data_loader):
        data, target = data.to(device), target.to(device)
        y_pred = model(data)
        loss = l(y_pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} | Batch Status: {}/{} ({:.0f}%) | Loss: {:.6f}'.format(epoch, batch_idx * len(data), len(train_data_loader.dataset),100. * batch_idx / len(train_data_loader), loss.item()))

def test():
    model.eval()
    test_loss=0
    correct = 0
    for data, target in test_data_loader:
        data, target = data.to(device), target.to(device)
        output = model(data)
        test_loss+=l(output, target)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    loss_avg = test_loss/(len(test_data_loader.dataset))
    accuracy = 100*(correct//len(test_data_loader.dataset))
    print("=====================")
    print("Average Loss: ", loss_avg, "Accuracy: ", accuracy)

for epoch in range(1, 2):
    train(epoch)
    test()

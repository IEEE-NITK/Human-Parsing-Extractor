import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from torch import cuda
import numpy as np
import matplotlib.pyplot as plt

device = 'cuda' if cuda.is_available() else 'cpu'
batch_size = 64
mnist_train = MNIST('mnist', train=True, download=True, transform=ToTensor())
train_data_loader = T.utils.data.DataLoader(mnist_train, batch_size=64, shuffle = True)
mnist_test = MNIST('mnist', train=False, download=True, transform=ToTensor())
test_data_loader = T.utils.data.DataLoader(mnist_test, batch_size=64, shuffle = True)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv3 = nn.Conv2d(20, 40, kernel_size=3)
        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(160, 10)

    def forward(self, x):
        input_size = x.size(0)
        x = self.conv1(x)
        x = F.relu(self.mp(x))
        x = self.conv2(x)
        x = F.relu(self.mp(x))
        x = F.relu(self.conv3(x))
        x = x.view(input_size, -1)
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

for epoch in range(1, 10):
    train(epoch)
    test()

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
class mnist_class(nn.Module):
    def __init__(self):
        super(mnist_class, self).__init__()
        self.l1 = nn.Linear(784, 520)
        self.l2 = nn.Linear(520, 320)
        self.l3 = nn.Linear(320, 240)
        self.l4 = nn.Linear(240, 120)
        self.l5 = nn.Linear(120, 10)

    def forward(self, x):
        x = x.view(-1, 784) #Flatten the data
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        y_pred = F.relu(self.l5(x))
        return y_pred;

model = mnist_class()
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

for epoch in range(1, 4):
    train(epoch)
    test()

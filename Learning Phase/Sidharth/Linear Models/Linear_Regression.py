import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import tensor
from torch.autograd import Variable
from torch import nn

x_data = Variable(tensor([[1.0], [2.0], [3.0],[4.0],[5.0],[6.0]]))
y_data = Variable(tensor([[2.0], [4.0], [6.0],[7.0],[9.0],[12.0]]))

x_data = x_data.float()
y_data = y_data.float()
list_1 = []

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1,1)
    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = Model()

criterion = nn.MSELoss(size_average=None)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
# optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# optimizer = torch.optim.Adamax(model.parameters(), lr=0.001)
# optimizer = torch.optim.Adamax(model.parameters(), lr=0.001)
# optimizer = torch.optim.Adamax(model.parameters(), lr=0.001)

for epoch in range(500):
    y_pred = model(x_data)
    print("x_data= ", x_data)
    print("y_pred= ", y_pred)
    loss = criterion(y_pred, y_data)
    print("Epoch = ", epoch, "Loss = ", loss.data)
    list_1.append(loss.data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

plt.plot(torch.arange(500), list_1)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()    

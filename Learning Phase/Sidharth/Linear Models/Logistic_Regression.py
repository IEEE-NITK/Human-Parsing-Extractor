import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import tensor
from torch.autograd import Variable
from torch import nn

x_data = Variable(tensor([[1.0], [2.0], [3.0],[4.0],[5.0],[6.0],[12.0],[9.0]]))
y_data = Variable(tensor([[0.0], [1.0], [1.0],[1.0],[0.0],[0.0],[1.0],[0.0]]))

x_data = x_data.float()
y_data = y_data.float()
list_1 = []

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear = nn.Linear(1,1)
    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

m = y_data.shape[0]
model = Model()

criterion = nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

for epoch in range(250):
    y_pred = model(x_data)
    print("x_data= ", x_data, "y_pred= ", y_pred)
    loss = criterion(y_pred, y_data)
    print(f'Epoch {epoch + 1}/250 | Loss: {loss.item():.4f}')
    list_1.append(loss.data)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

hour_variable = model(tensor([[1.0]]))
print(f'Prediction after 1 hour of training: {hour_var.item():.4f} | Above 50%: {hour_var.item() > 0.5}')
hour_variable = model(tensor([[7.0]]))
print(f'Prediction after 7 hours of training: {hour_var.item():.4f} | Above 50%: { hour_var.item() > 0.5}')

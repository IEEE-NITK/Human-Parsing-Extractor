from torch import nn
import torch
from torch import tensor
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader

class DiabetesDataset(Dataset):
    def __init__(self):
        data = np.loadtxt('diabetes.csv', delimiter = ',', dtype = np.float32)
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data[:, 0 : -1])
        self.y_data = torch.from_numpy(data[:, -1]).reshape(-1, 1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset,batch_size=32,shuffle=True,num_workers=2)

class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Linear(8, 6)
        self.layer2 = nn.Linear(6, 4)
        self.layer3 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        output1 = self.sigmoid(self.layer1(x))
        output2 = self.sigmoid(self.layer2(output1))
        y_pred = self.sigmoid(self.layer3(output2))
        return y_pred

model = Model()
criterion = torch.nn.BCELoss(reduction='sum')
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

for epoch in range(2):
    for i, data in enumerate(train_loader, 0):
        inp, op = data
        y_pred = model(inp)
        loss = criterion(y_pred, op)
        print(f'Epoch {epoch + 1} | Batch: {i+1} | Loss: {loss.item():.4f}')
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

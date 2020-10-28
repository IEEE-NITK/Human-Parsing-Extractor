import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

# Const Variables
FILE_NAME = "diabetes.csv"
N_FEATURES = 8
EPOCHS = 3000
LEARNING_RATE = 0.01
VERBOSE = 30

class DiabetesDataset(Dataset):
    def __init__(self):
        # Import Dataset
        data = np.loadtxt(FILE_NAME, delimiter = ',', dtype = np.float32)

        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data[:, 0 : -1])
        self.y_data = torch.from_numpy(data[:, -1]).reshape(-1, 1)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
        

class Net(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.fc1 = torch.nn.Linear(n_features, 8)
        self.fc2 = torch.nn.Linear(8, 20)
        self.fc3 = torch.nn.Linear(20, 15)
        self.fc4 = torch.nn.Linear(15, 10)
        self.fc5 = torch.nn.Linear(10, 8)
        self.fc6 = torch.nn.Linear(8, 1)

    def forward(self, X):
        X = torch.relu(self.fc1(X))
        X = torch.relu(self.fc2(X))
        X = torch.relu(self.fc3(X))
        X = torch.relu(self.fc4(X))
        X = torch.relu(self.fc5(X))
        X = torch.sigmoid(self.fc6(X))

        return X

    def calculate_accuracy(self, X, y):

        # Number of training examples 
        m = y.shape[0]

        # Declaring tensors for torch.where()
        one = torch.ones([m, 1])
        zero = torch.zeros([m, 1])

        # Forward Pass
        y_pred = self(X)
        y_pred = torch.where(y_pred >= 0.5, one, zero)

        return torch.sum(y_pred == y).item() / m * 100

    def train(self, data_set, data_loader, EPOCHS, learning_rate, verbose):
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)

        for epoch in range(EPOCHS):
            for data in data_loader:
                # Unpack data
                X, y = data

                # Forward Pass
                y_pred = self(X)
                accuracy = self.calculate_accuracy(X, y)
                loss = criterion(y_pred, y)

                # Backward Pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            epoch_loss = criterion(self(data_set.x_data), data_set.y_data)
            if epoch % (EPOCHS // verbose) == 0:
                print(f"Epoch : {epoch} | loss : {round(epoch_loss.item(), 5)}")

        print(f"Epoch : {EPOCHS} | loss : {round(epoch_loss.item(), 5)}") 
        
# Data Loader
data = DiabetesDataset()
train_loader = DataLoader(dataset=data, batch_size=100, shuffle=True)

# Model Instance

torch.manual_seed(0)

net = Net(N_FEATURES)
net.train(data, train_loader, EPOCHS, LEARNING_RATE, VERBOSE)

accuracy = net.calculate_accuracy(data.x_data, data.y_data)
print(f"\nModel accuracy after {EPOCHS} epochs : {round(accuracy, 2)}\n") 





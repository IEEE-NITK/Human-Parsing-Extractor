import torch
import numpy as np

# Const Variables
FILE_NAME = "diabetes.csv"
N_FEATURES = 8
EPOCHS = 100000
LEARNING_RATE = 0.1
VERBOSE = 10

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

    def train(self, X, y, EPOCHS, learning_rate, verbose):
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr = learning_rate)

        for epoch in range(EPOCHS):
            # Forward Pass
            y_pred = self(X)
            accuracy = self.calculate_accuracy(X, y)
            loss = criterion(y_pred, y)

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % (EPOCHS // verbose) == 0:
                print(f"Epoch : {epoch} | loss : {round(loss.item(), 5)} | accuracy : {round(accuracy, 2)}")

        print(f"Epoch : {EPOCHS} | loss : {round(loss.item(), 5)} | accuracy : {round(accuracy, 2)}") 
        

# Import Dataset
data = np.loadtxt(FILE_NAME, delimiter = ',', dtype = np.float32)

X = torch.from_numpy(data[:, 0 : -1])
y = torch.from_numpy(data[:, -1]).reshape(-1, 1)

# Model Instance

torch.manual_seed(0)

net = Net(N_FEATURES)
net.train(X, y, EPOCHS, LEARNING_RATE, VERBOSE)





# Importing Libraries
import torch
import pandas as pd

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

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

    def get_accuracy(self, y_pred, y):

        # Number of training examples
        m = y.shape[0]
        
        # Declaring tensors for torch.where()
        one = torch.ones([m, 1])
        zero = torch.zeros([m, 1])

        y_pred = torch.where(y_pred >= 0.5, one, zero)
        return torch.sum(y_pred == y).item() / m * 100
  
    def train(self, X, y, EPOCHS, learning_rate, verbose):
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr = learning_rate)

        for epoch in range(EPOCHS):
            # Forward Pass
            y_pred = self(X)
            accuracy = self.get_accuracy(y_pred, y)
            loss = criterion(y_pred, y)

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % (EPOCHS // verbose) == 0: 
                print(f"Epoch : {epoch} | loss : {loss} | accuracy : {accuracy}")

        print(f"Epoch : {EPOCHS} | loss : {loss} | accuracy : {accuracy}")


# Import Dataset
df = pd.read_csv('ex2data1.csv')

feature_columns = df.columns[0:-1]
label_columns = df.columns[-1]

X = torch.tensor(df[feature_columns].values)
y = torch.tensor(df[label_columns].values).reshape(df[label_columns].shape[0], 1)

X = X.float()
y = y.float()


# Model Instance
torch.manual_seed(0)
logistic_regression_model = LogisticRegressionModel(2)
logistic_regression_model.train(X, y, EPOCHS = 2000, learning_rate = 0.01, verbose = 10)

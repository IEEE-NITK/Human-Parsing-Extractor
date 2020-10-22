# Importing Libraries
import torch
import math
import pandas as pd
import matplotlib.pyplot as plt


class PolynomialRegressionModel(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = torch.nn.Linear(n_features, 1)
    
    def forward(self, x):
        return self.linear(x)
  
    def train(self, X, y, EPOCHS, learning_rate, verbose):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.SGD(self.parameters(), lr = learning_rate)

        for epoch in range(EPOCHS):
            # Forward Pass
            y_pred = self(X)
            loss = criterion(y_pred, y)

            # Backward Pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if epoch % (EPOCHS // verbose) == 0: 
                print(f"Epoch : {epoch} | loss : {loss}")

        print(f"Epoch : {EPOCHS} | loss : {loss}")


# Import Dataset
df = pd.read_csv('ex1data1.csv')

X = torch.tensor(df['W'].values).reshape(df['W'].shape[0], 1).T
X = torch.cat((X, torch.pow(X, 2)))
X = X.T
y = torch.tensor(df['Y'].values).reshape(df['Y'].shape[0], 1)

X = X.float()
y = y.float()


def plot_curve(X, y, model):
    y_pred = []

    curve_start, _ = torch.min(X.T[0], 0)
    curve_end, _ = torch.max(X.T[0], 0) 

    curve_start = int(curve_start.item()) - 2
    curve_end = int(curve_end.item()) + 2

    for x in range(curve_start, curve_end):
        x = torch.Tensor([[x, x * x]])
        y_pred.append(model(x))

    plt.title("Labels vs Features")
    plt.xlabel("Features")
    plt.ylabel("Labels / Predictions")
    plt.scatter(X.T[0].T, y)
    plt.plot(range(curve_start, curve_end), y_pred, 'k')
    plt.show()


# Model Instance
polynomial_regression_model = PolynomialRegressionModel(2)
polynomial_regression_model.train(X, y, EPOCHS = 100000, learning_rate = 0.000001, verbose = 10)



# Plotting to see the curve
plot_curve(X, y, polynomial_regression_model)

# Note : Training Time can be significantly reduced by normalizing the features.

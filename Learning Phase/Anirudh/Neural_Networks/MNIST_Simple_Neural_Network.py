import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt

# Const Variables
FILE_NAME = "diabetes.csv"
N_FEATURES = 28 * 28
EPOCHS = 100
LEARNING_RATE = 0.01
VERBOSE = 10

def import_data():
    train = datasets.MNIST("", train=True, download=True, 
                            transform = transforms.Compose([transforms.ToTensor()]))
    test = datasets.MNIST("", train=False, download=True, 
                            transform = transforms.Compose([transforms.ToTensor()]))
    train_set = DataLoader(train, batch_size=256, shuffle=True)
    test_set = DataLoader(test, batch_size=256, shuffle=True)
    return train_set, test_set

def plot_image(image):
    plt.imshow(image.view(28, 28))
    plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.fc1 = torch.nn.Linear(n_features, 64)
        self.fc2 = torch.nn.Linear(64, 16)
        # self.fc3 = torch.nn.Linear(25, 16)
        # self.fc4 = torch.nn.Linear(16, 16)
        # self.fc5 = torch.nn.Linear(16, 16)
        self.fc6 = torch.nn.Linear(16, 10)

    def forward(self, X):
        X = torch.relu(self.fc1(X))
        X = torch.relu(self.fc2(X))
        # X = torch.relu(self.fc3(X))
        # X = torch.relu(self.fc4(X))
        # X = torch.relu(self.fc5(X))
        X = torch.log_softmax(self.fc6(X), dim = 1)

        return X

    def calculate_accuracy(self, data_set):
        correct_count = 0
        total_count = 0

        for data in data_set:
            X, y = data

            # Number of training examples 
            m = y.shape[0]

            # Declaring tensors for torch.where()
            one = torch.ones([m, 1])
            zero = torch.zeros([m, 1])

            # Forward Pass
            y_pred = self(X.view(-1, 28 * 28))
            _, y_pred = torch.max(y_pred, 1)

            correct_count += torch.sum(y_pred == y).item()
            total_count += m
        print(correct_count, total_count)
        return correct_count / total_count * 100

    def train(self, data_loader, EPOCHS, learning_rate, verbose):
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(EPOCHS):
            for data in data_loader:
                # Unpack data
                X, y = data

                # Forward Pass
                y_pred = self(X.view(-1, 28 * 28))
                loss = F.nll_loss(y_pred, y)

                # Backward Pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % (EPOCHS // verbose) == 0:
                print(f"Epoch : {epoch} | loss : {round(loss.item(), 5)}")

        if epoch % (EPOCHS // verbose) == 0:
            print(f"Epoch : {epoch} | loss : {round(loss.item(), 5)}")
        
# Import Data
train_set, test_set = import_data()

# Model Instance
torch.manual_seed(0)

net = Net(N_FEATURES)
net.train(train_set, EPOCHS, LEARNING_RATE, VERBOSE)

train_accuracy = net.calculate_accuracy(train_set)
print(f"Train accuracy after {EPOCHS} epochs : {round(train_accuracy)}")

test_accuracy = net.calculate_accuracy(test_set)
print(f"Test accuracy after {EPOCHS} epochs : {round(test_accuracy)}")





import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt


# Const Variables
EPOCHS = 10
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


class DenseBlock(torch.nn.Module):
    def __init__(self, n_channels, f=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(n_channels, n_channels, kernel_size=f, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(n_channels)

        self.conv2 = nn.Conv2d(n_channels, n_channels, kernel_size=f, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(n_channels)

        self.conv3 = nn.Conv2d(n_channels, n_channels, kernel_size=f, stride=stride, padding=padding)
        self.bn3 = nn.BatchNorm2d(n_channels)

    def forward(self, X):
        X_in = X # Storing input X

        # Layer 1
        X = self.conv1(X)
        X = self.bn1(X)
        X_layer1 = X # Storing Layer 1 output
        X = X + X_in # Dense Connection 1
        X = torch.relu(X)

        # Layer 2
        X = self.conv2(X)
        X = self.bn2(X)
        X_layer2 = X # Storing Layer 2 output
        X = X + X_in + X_layer1 # Dense Connection 2
        X = torch.relu(X)

        # Layer 3
        X = self.conv3(X)
        X = self.bn3(X)
        X = X + X_in + X_layer1 + X_layer2 # Dense Connection 3
        X = torch.relu(X)

        return X

class DenseNet(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Stage 1
        self.stage1_conv1 = nn.Conv2d(1, 64, kernel_size=5)
        self.stage1_mp1 = nn.MaxPool2d(2)

        # Stage 2 (2 x DenseBlock + 1 x MaxPooling)
        self.stage2_db1 = DenseBlock(64, 3)
        self.stage2_db2 = DenseBlock(64, 3)
        self.mp = nn.MaxPool2d(2)

        # Stage 3
        self.stage3_fc1 = nn.Linear(2304, 256);
        self.stage3_fc2= nn.Linear(256, 10);

    def forward(self, X):
        training_examples = X.size(0) # Stored for flattening later

        # Stage 1
        X = self.stage1_conv1(X)
        X = self.stage1_mp1(X)
        X = torch.relu(X);

        # Stage 2 (2 x DenseBlock + 1 x MaxPooling)
        X = self.stage2_db1(X)
        X = self.stage2_db2(X)
        X = self.mp(X)
        X = torch.relu(X);

        # Stage 3
        X = X.view(training_examples, -1)
        X = self.stage3_fc1(X)
        X = torch.relu(X)
        X = self.stage3_fc2(X)
        X = torch.log_softmax(X, dim = 1)

        return X

    def calculate_accuracy(self, data_set):
        correct_count = 0
        total_count = 0

        for data in data_set:
            # Unpack data
            X, y = data

            # Number of training examples 
            m = y.shape[0]

            # Declaring tensors for torch.where()
            one = torch.ones([m, 1])
            zero = torch.zeros([m, 1])

            # Forward Pass
            y_pred = self(X)
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
                y_pred = self(X)
                loss = F.nll_loss(y_pred, y)

                # Backward Pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if epoch % (EPOCHS // verbose) == 0:
                print(f"Epoch : {epoch} | loss : {round(loss.item(), 5)}")

        print(f"Epoch : {epoch} | loss : {round(loss.item(), 5)}")
        

if __name__ == '__main__':
    # Import Data
    train_set, test_set = import_data()

    # Model Instance
    torch.manual_seed(0)
    densenet = DenseNet()
    densenet.train(train_set, EPOCHS, LEARNING_RATE, VERBOSE)

    train_accuracy = densenet.calculate_accuracy(train_set)
    print(f"Train accuracy after {EPOCHS} epochs : {round(train_accuracy, 4)}")

    test_accuracy = densenet.calculate_accuracy(test_set)
    print(f"Test accuracy after {EPOCHS} epochs : {round(test_accuracy, 4)}")





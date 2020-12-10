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


class InceptionBlockA(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Branch 1
        self.branch1_1x1 = nn.Conv2d(in_channels, 24, kernel_size=1)

        # Branch 2
        self.branch2_mp = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.branch2_1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        # Branch 3
        self.branch3_1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3_5x5 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        # Branch 4
        self.branch4_1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch4_3x3_1 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch4_3x3_2 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

    def forward(self, X):
        # Branch 1
        X_branch1 = self.branch1_1x1(X)

        # Branch 2
        X_branch2 = self.branch2_mp(X)
        X_branch2 = self.branch2_1x1(X_branch2)

        # Branch 3
        X_branch3 = self.branch3_1x1(X)
        X_branch3 = self.branch3_5x5(X_branch3)

        # Branch 4
        X_branch4 = self.branch4_1x1(X)
        X_branch4 = self.branch4_3x3_1(X_branch4)
        X_branch4 = self.branch4_3x3_2(X_branch4)

        # Concatinating branches
        output = [X_branch1, X_branch2, X_branch3, X_branch4]
        output = torch.cat(output, 1)

        output = torch.relu(output)

        return output


class InceptionModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Stage 1 ((Conv -> MaxPool -> BatchNorm) x 2)
        self.stage1_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2)
        self.stage1_mp = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.stage1_bn1 = nn.BatchNorm2d(64)
        self.stage1_conv2 = nn.Conv2d(64, 32, kernel_size=1)
        self.stage1_conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.stage1_bn2 = nn.BatchNorm2d(32)

        # Stage 2 (MaxPool -> InceptionBlock x 2)
        self.stage2_mp = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.stage2_IBA_1 = InceptionBlockA(32)
        self.stage2_IBA_2 = InceptionBlockA(88)

        # Stage 3
        self.stage3_ag = nn.AvgPool2d(2)
        self.stage3_fc = nn.Linear(2200, 10)

    def forward(self, X):
        training_examples = X.size(0) # Storing for flattening

        # Stage 1 ((Conv -> MaxPool -> BatchNorm) x 2)
        X = self.stage1_conv1(X)
        X = self.stage1_mp(X)
        X = self.stage1_bn1(X)
        X = self.stage1_conv2(X)
        X = self.stage1_conv3(X)
        X = self.stage1_bn2(X)
        X = torch.relu(X)

        # Stage 2 (MaxPool -> InceptionBlock x 2)
        X = self.stage2_mp(X)
        X = self.stage2_IBA_1(X)
        X = self.stage2_IBA_2(X)

        # Stage 3
        X = self.stage3_ag(X)
        X = X.view(training_examples, -1)
        X = self.stage3_fc(X)
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
    net = InceptionModel()
    net.train(train_set, EPOCHS, LEARNING_RATE, VERBOSE)

    train_accuracy = net.calculate_accuracy(train_set)
    print(f"Train accuracy after {EPOCHS} epochs : {round(train_accuracy, 4)}")

    test_accuracy = net.calculate_accuracy(test_set)
    print(f"Test accuracy after {EPOCHS} epochs : {round(test_accuracy, 4)}")




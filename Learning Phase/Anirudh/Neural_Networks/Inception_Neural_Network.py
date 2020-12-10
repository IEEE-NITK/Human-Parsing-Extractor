import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt


# Const Variables
EPOCHS = 10
LEARNING_RATE = 0.001
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


class InceptionBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        # 1x1 Branch
        self.branch1_1x1 = nn.Conv2d(in_channels, out_channels[0], kernel_size=1)

        # MaxPooling Branch
        self.branch2_mp = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)
        self.branch2_1x1 = nn.Conv2d(in_channels, out_channels[1], kernel_size=1)

        # 5x5 Branch
        self.branch3_1x1 = nn.Conv2d(in_channels, out_channels[2], kernel_size=1)
        self.branch3_5x5 = nn.Conv2d(out_channels[2], out_channels[3], kernel_size=5, padding=2)

        # 3x3 Branch
        self.branch4_1x1 = nn.Conv2d(in_channels, out_channels[4], kernel_size=1)
        self.branch4_3x3_1 = nn.Conv2d(out_channels[4], out_channels[5], kernel_size=3, padding=1)
        self.branch4_3x3_2 = nn.Conv2d(out_channels[5], out_channels[6], kernel_size=3, padding=1)

    def forward(self, X):

        # 1x1 Branch
        X_branch1 = self.branch1_1x1(X)

        # MaxPooling Branch
        X_branch2 = self.branch2_mp(X)
        X_branch2 = self.branch2_1x1(X_branch2)

        # 5x5 Branch
        X_branch3 = self.branch3_1x1(X)
        X_branch3 = self.branch3_5x5(X_branch3)

        # 3x3 Branch
        X_branch4 = self.branch4_1x1(X)
        X_branch4 = self.branch4_3x3_1(X_branch4)
        X_branch4 = self.branch4_3x3_2(X_branch4)

        # Concatinating Branches
        output = [X_branch1, X_branch2, X_branch3, X_branch4]
        output = torch.cat(output, 1)

        output = torch.relu(output)

        return output


class SideBlock(torch.nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.mp = nn.MaxPool2d(kernel_size=5, stride=2)
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=1)
        self.fc1 = nn.Linear(512, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, X):
        training_size = X.size(0) # Storing for flattening

        # Conv Layer
        X = self.mp(X)
        X = self.conv1(X)
        
        # Flattening
        X = X.view(training_size, -1) 

        # Fully Connected Layers
        X = self.fc1(X)
        X = self.fc2(X)
        X = torch.log_softmax(X, dim=1)

        return X

class InceptionV3(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Stage 1
        self.stage1_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2)
        self.stage1_mp = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.stage1_bn1 = nn.BatchNorm2d(64)
        self.stage1_conv2 = nn.Conv2d(64, 32, kernel_size=1)
        self.stage1_conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.stage1_bn2 = nn.BatchNorm2d(32)

        # Stage 2
        self.stage2_mp = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.stage2_IBA_1 = InceptionBlock(32, [24, 16, 16, 24, 16, 24, 24])
        self.stage2_IBA_2 = InceptionBlock(88, [24, 16, 16, 24, 16, 24, 24])

        # Stage 3
        self.stage3_mp = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.stage3_IBA_1 = InceptionBlock(88, [48, 32, 32, 48, 32, 48, 48])
        self.stage3_IBA_2 = InceptionBlock(176, [48, 32, 32, 48, 32, 48, 48])
        self.stage3_IBA_3 = InceptionBlock(176, [48, 32, 32, 48, 32, 48, 48])
        self.stage3_IBA_4 = InceptionBlock(176, [48, 32, 32, 48, 32, 48, 48])
        self.stage3_IBA_5 = InceptionBlock(176, [48, 32, 32, 48, 32, 48, 48])

        # Stage 4
        self.stage4_mp = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage4_IBA_1 = InceptionBlock(176, [96, 64, 64, 96, 64, 96, 96])
        self.stage4_IBA_2 = InceptionBlock(352, [96, 64, 64, 96, 64, 96, 96])

        # Stage 5
        self.stage5_ag = nn.AvgPool2d(2)
        self.stage5_fc = nn.Linear(1408, 10)

        # Side Blocks
        self.sideblock1 = SideBlock(176)
        self.sideblock2 = SideBlock(176)

    def forward(self, X):
        training_examples = X.size(0)

        # Stage 1
        X = self.stage1_conv1(X)
        X = self.stage1_mp(X)
        X = self.stage1_bn1(X)
        X = self.stage1_conv2(X)
        X = self.stage1_conv3(X)
        X = self.stage1_bn2(X)
        X = torch.relu(X)

        # Stage 2
        X = self.stage2_mp(X)
        X = self.stage2_IBA_1(X)
        X = self.stage2_IBA_2(X)

        # Stage 3
        X = self.stage3_mp(X)
        X = self.stage3_IBA_1(X)

        X_sidebranch1 = self.sideblock1(X)

        X = self.stage3_IBA_2(X)
        X = self.stage3_IBA_3(X)
        X = self.stage3_IBA_4(X)

        X_sidebranch2 = self.sideblock2(X)

        X = self.stage3_IBA_5(X)

        # Stage 4
        X = self.stage4_mp(X)
        X = self.stage4_IBA_1(X)
        X = self.stage4_IBA_2(X)

        # Stage 5
        X = self.stage5_ag(X)
        X = X.view(training_examples, -1)
        X = self.stage5_fc(X)
        X = torch.log_softmax(X, dim = 1)

        return X, X_sidebranch1, X_sidebranch2

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
            y_pred, y_pred1, y_pred2 = self(X)
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
                y_pred, y_pred1, y_pred2 = self(X)
                
                # Computing Losses
                loss = F.nll_loss(y_pred, y) # Main Loss
                loss1 = F.nll_loss(y_pred1, y) # SideBranch1
                loss2 = F.nll_loss(y_pred2, y) # SideBranch2
                total_loss = loss + loss1 + loss2

                # Backward Pass
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()        
                
            if epoch % (EPOCHS // verbose) == 0:
                print(f"Epoch : {epoch} | loss : {round(loss.item(), 5)}")

        print(f"Epoch : {epoch} | loss : {round(loss.item(), 5)}")
        

if __name__ == '__main__':
    # Import Data
    train_set, test_set = import_data()

    # Model Instance
    torch.manual_seed(0)
    net = InceptionV3()
    net.train(train_set, EPOCHS, LEARNING_RATE, VERBOSE)

    train_accuracy = net.calculate_accuracy(train_set)
    print(f"Train accuracy after {EPOCHS} epochs : {round(train_accuracy, 4)}")

    test_accuracy = net.calculate_accuracy(test_set)
    print(f"Test accuracy after {EPOCHS} epochs : {round(test_accuracy, 4)}")





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


class IdentityBlock(nn.Module):
    def __init__(self, in_channels, out_channels,f=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels[0], kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels[0])

        self.conv2 = nn.Conv2d(out_channels[0], out_channels[1], kernel_size=f, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels[1])

        self.conv3 = nn.Conv2d(out_channels[1], out_channels[2], kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels[2])

    def forward(self, X):
        X_cut = X
        
        # Layer 1
        X = self.conv1(X)
        X = self.bn1(X)
        X = torch.relu(X)

        # Layer 2
        X = self.conv2(X)
        X = self.bn2(X)
        X = torch.relu(X)

        # Layer 3
        X = self.conv3(X)
        X = self.bn3(X)

        assert(X.shape == X_cut.shape)

        # Skip Connection
        X += X_cut
        X = torch.relu(X)

        return  X


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels,f=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels[0], kernel_size=1, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(out_channels[0])

        self.conv2 = nn.Conv2d(out_channels[0], out_channels[1], kernel_size=f, stride=1, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels[1])

        self.conv3 = nn.Conv2d(out_channels[1], out_channels[2], kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(out_channels[2])

        self.convcut = nn.Conv2d(in_channels, out_channels[2], kernel_size=1, stride=stride, padding=0)
        self.bncut = nn.BatchNorm2d(out_channels[2])

    def forward(self, X):
        X_cut = X
        
        # Layer 1
        X = self.conv1(X)
        X = self.bn1(X)
        X = torch.relu(X)

        # Layer 2
        X = self.conv2(X)
        X = self.bn2(X)
        X = torch.relu(X)

        # Layer 3
        X = self.conv3(X)
        X = self.bn3(X)

        # Skip Connection
        X_cut = self.convcut(X_cut)
        X_cut = self.bncut(X_cut)

        assert(X.shape == X_cut.shape)

        X += X_cut
        X = torch.relu(X)

        return  X


class ResNet50(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Stage 1
        self.stage1_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2)
        self.stage1_bn1 = nn.BatchNorm2d(64)
        self.stage1_mp = nn.MaxPool2d(kernel_size=3, stride=2)

        # Stage 2 (1 x ConvBlock + 1 x IdentityBlock)
        self.stage2_conv1 = ConvBlock(64, [64, 64, 256])
        self.stage2_id1 = IdentityBlock(256, [64, 64, 256])

        # Stage 3 (1 x ConvBlock + 3 x IdentityBlock)
        self.stage3_conv1 = ConvBlock(256, [128, 128, 512])
        self.stage3_id1 = IdentityBlock(512, [128, 128, 512])
        self.stage3_id2 = IdentityBlock(512, [128, 128, 512])
        self.stage3_id3 = IdentityBlock(512, [128, 128, 512])

        # Stage 4 (1 x ConvBlock + 5 x IdentityBlock)
        self.stage4_conv1 = ConvBlock(512, [256, 256, 1024])
        self.stage4_id1 = IdentityBlock(1024, [256, 256, 1024])
        self.stage4_id2 = IdentityBlock(1024, [256, 256, 1024])
        self.stage4_id3 = IdentityBlock(1024, [256, 256, 1024])
        self.stage4_id4 = IdentityBlock(1024, [256, 256, 1024])
        self.stage4_id5 = IdentityBlock(1024, [256, 256, 1024])

        # Stage 5 (1 x ConvBlock + 3 x IdentityBlock)
        self.stage5_conv1 = ConvBlock(1024, [512, 512, 2048])
        self.stage5_id1 = IdentityBlock(2048, [512, 512, 2048])
        self.stage5_id2 = IdentityBlock(2048, [512, 512, 2048])
        self.stage5_id3 = IdentityBlock(2048, [512, 512, 2048])

        # Stage 6
        self.stage6_ap = nn.AvgPool2d(2)
        self.stage6_fc = torch.nn.Linear(8192, 10)

    def forward(self, X):
        # Stored for flattening later
        batch_size = X.size(0)

        # Stage 1
        X = self.stage1_conv1(X)
        X = self.stage1_bn1(X)
        X = self.stage1_mp(X)

        # Stage 2 (1 x ConvBlock + 1 x IdentityBlock)
        X = self.stage2_conv1(X)
        X = self.stage2_id1(X)

        # Stage 3 (1 x ConvBlock + 3 x IdentityBlock)
        X = self.stage3_conv1(X)
        X = self.stage3_id1(X)
        X = self.stage3_id2(X)
        X = self.stage3_id3(X)
        
        # Stage 4 (1 x ConvBlock + 5 x IdentityBlock)
        X = self.stage4_conv1(X)
        X = self.stage4_id1(X)
        X = self.stage4_id2(X)
        X = self.stage4_id3(X)
        X = self.stage4_id4(X)
        X = self.stage4_id5(X)
        
        # Stage 5 (1 x ConvBlock + 3 x IdentityBlock)
        X = self.stage5_conv1(X)
        X = self.stage5_id1(X)
        X = self.stage5_id2(X)
        X = self.stage5_id3(X)
        
        # Stage 6
        X = self.stage6_ap(X)
        X = X.view(batch_size, -1) # Flatten
        X = self.stage6_fc(X) 
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
        criterion = torch.nn.CrossEntropyLoss()

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
    resnet = ResNet50()
    resnet.train(train_set, EPOCHS, LEARNING_RATE, VERBOSE)

    train_accuracy = resnet.calculate_accuracy(train_set)
    print(f"Train accuracy after {EPOCHS} epochs : {round(train_accuracy, 4)}")

    test_accuracy = resnet.calculate_accuracy(test_set)
    print(f"Test accuracy after {EPOCHS} epochs : {round(test_accuracy, 4)}")





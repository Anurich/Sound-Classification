from re import M
import torch.nn as nn
import torch
class MLP(nn.ModuleList):
    def __init__(self, numClass):
        super().__init__()
        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1760, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=256, out_features= 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_features=64, out_features=numClass)
        )

        self.softmax = nn.LogSoftmax(dim=-1)
    def forward(self, x):
        x = self.dense(x)
        return x


class CNNetwork(nn.ModuleList):
    def __init__(self, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels= 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels= 64, kernel_size=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.BatchNorm2d(64)
        )

        self.dense = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=1600, out_features=128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=128, out_features= num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.conv1(x)
        x = self.conv2(x)

        return self.dense(x)

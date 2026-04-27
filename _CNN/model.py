import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, channels_in, num_classes = 10):
        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(channels_in, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(32, 128, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size = 2, stride = 2),

            nn.Conv2d(128, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size = 2, stride = 2)

        )

        self.flatten = nn.Flatten()
        self.FullConn_1 = nn.Linear(256 * 3 * 3, 256)
        self.ReLU_FC = nn.ReLU()
        self.FullConn_2 = nn.Linear(256, num_classes)

    def forward(self, x):

        x = self.features(x)
        x = self.flatten(x)
        x = self.FullConn_1(x)
        x = self.ReLU_FC(x)
        x = self.FullConn_2(x)
        
        return x
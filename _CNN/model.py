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

# Idan's CNN; checking to see if any difference in results between network architectures

"""
class Net(nn.Module):
    def __init__(self,N_channel,kernel_s):
        super().__init__()

        self.conv1 = nn.Conv2d(N_channel, 32, kernel_size = kernel_s, stride=1, padding=np.ceil((k-1)/2).astype(int))
        self.pool1 = nn.MaxPool2d(6, 6)
        self.pool2 = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(32, 128, kernel_size = 3, stride=1, padding=1, device = "cuda")
        self.conv3 = nn.Conv2d(128, 256, kernel_size = 3, stride=1, padding=1, device = "cuda")

        self.fc1 = nn.Linear(512, 256, device = "cuda")
        self.fc2 = nn.Linear(256, 2, device = "cuda")

    def forward(self, x):
        # input image
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        # 256x1
        x = self.fc2(x)
        # 2x1
        
        return x
"""
import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, channels_in):
        super(CNN, self).__init__()

        self.features = nn.Sequential(

            nn.Conv1d(channels_in, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size = 2, stride = 2),

            nn.Conv1d(64, 256, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size = 2, stride = 2),

            nn.Conv1d(256, 512, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),

            nn.MaxPool1d(kernel_size = 2, stride = 2)

        )

        self.flatten = nn.Flatten()
        self.FullConn_1 = nn.Linear(1499 * 512, 1500)
        self.ReLU_FC = nn.ReLU()
        self.FullConn_2 = nn.Linear(1500, 512)
        self.ReLU_FC = nn.ReLU()
        self.FullConn_3 = nn.Linear(512, 6)

    def forward(self, x):

        x = self.features(x)
        x = self.flatten(x)
        x = self.FullConn_1(x)
        x = self.ReLU_FC(x)
        x = self.FullConn_2(x)
        x = self.ReLU_FC(x)
        x = self.FullConn_3(x)

        return x

# # Check the size of each intermediate tensor

# Model_Check = CNN(channels_in = 8)

# x = torch.randn(32, 8, 11993)

# with torch.no_grad():
#   print("Inputs:", x.shape)

#   x = Model_Check.features[0](x)
#   x = Model_Check.features[1](x)
#   print("After First Convolution + ReLU:", x.shape)

#   x = Model_Check.features[2](x)
#   print("After First Max Pooling:", x.shape)

#   x = Model_Check.features[3](x)
#   x = Model_Check.features[4](x)
#   print("After Second Convolution + ReLU:", x.shape)

#   x = Model_Check.features[5](x)
#   print("After Second Max Pooling:", x.shape)

#   x = Model_Check.features[6](x)
#   x = Model_Check.features[7](x)
#   print("After Third Convolution + ReLU:", x.shape)

#   x = Model_Check.features[8](x)
#   print("After Third Max Pooling:", x.shape)

#   x = Model_Check.flatten(x)
#   print("After Flattening:", x.shape)

#   x = Model_Check.FullConn_1(x)
#   x = Model_Check.ReLU_FC(x)
#   print("After Full Connection 1 + ReLU:", x.shape)

#   x = Model_Check.FullConn_2(x)
#   print("After Full Connection 2:", x.shape)
  
#   x = Model_Check.FullConn_3(x)
#   print("After Full Connection 3:", x.shape)

import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layers = [10, 64, 64, 6]):
        super(MLP, self).__init__()

        self.layers = []

        self.act = nn.ReLU()

        for _ in range(len(layers) - 2):
            self.layers += [nn.Linear(layers[_], layers[_+1]), self.act]
        self.layers += [nn.Linear(layers[-2], layers[-1]), nn.Sigmoid()]
        
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)
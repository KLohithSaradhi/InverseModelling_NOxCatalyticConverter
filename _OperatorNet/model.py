import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, layers = [10, 64, 64, 6]):
        super(MLP, self).__init__()

        self.layers = []

        self.act = nn.GELU()

        for _ in range(len(layers) - 2):
            self.layers += [nn.Linear(layers[_], layers[_+1]), self.act]
        self.layers += [nn.Linear(layers[-2], layers[-1])]
        
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)
    

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        
        super(GRUModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # The GRU Layer
        self.gru = nn.GRU(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Fully Connected (Linear) Layer
        # This will be applied to the GRU output at EVERY time step
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size, device=x.device)
        
        out, _ = self.gru(x, h0)
        
        predictions = self.fc(out)
        
        return predictions
    
class OperatorNetwork(nn.Module):
    def __init__(self, trunk_input_size, branch_input_size, num_layers=2,  p = 16):
        super(OperatorNetwork, self).__init__()


        self.trunk = GRUModel(input_size=trunk_input_size, hidden_size=p, num_layers=num_layers, output_size=p)
        self.branch = MLP([branch_input_size, p, p])

    def forward(self, seq, A):
        basis = self.trunk(seq)

        coeff = self.branch(A)

        out = torch.einsum('btp,bp->bt', basis, coeff).unsqueeze(-1)

        return out
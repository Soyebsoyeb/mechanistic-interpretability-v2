import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, d_model, d_mlp, activation=torch.relu):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(d_model, d_mlp) * 0.02)
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_mlp, d_model) * 0.02)
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.activation = activation

    def forward(self, x):
        h = self.activation(x @ self.W_in + self.b_in)
        return h @ self.W_out + self.b_out

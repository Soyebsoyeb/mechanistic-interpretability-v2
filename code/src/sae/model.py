import torch
import torch.nn as nn

class SparseAutoencoder(nn.Module):
    def __init__(self, d_in, d_hidden, sparsity_lambda=1e-3):
        super().__init__()
        self.W_e = nn.Parameter(torch.randn(d_in, d_hidden) * 0.01)
        self.b_e = nn.Parameter(torch.zeros(d_hidden))
        self.W_d = nn.Parameter(torch.randn(d_hidden, d_in) * 0.01)
        self.b_d = nn.Parameter(torch.zeros(d_in))
        self.lambda_reg = sparsity_lambda

    def encode(self, x):
        return torch.relu(x @ self.W_e + self.b_e)

    def decode(self, z):
        return z @ self.W_d + self.b_d

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        recon = ((x_hat - x) ** 2).mean()
        sparse = self.lambda_reg * z.abs().mean()
        return {
            'x_hat': x_hat, 'z': z,
            'recon_loss': recon, 'sparsity_loss': sparse,
            'total_loss': recon + sparse,
            'mean_l0': (z > 0).float().mean().item(),
            'dead_features': (z.sum(dim=0) == 0).sum().item()
        }

    def normalize_decoder(self):
        with torch.no_grad():
            self.W_d.data /= (self.W_d.data.norm(dim=1, keepdim=True) + 1e-8)

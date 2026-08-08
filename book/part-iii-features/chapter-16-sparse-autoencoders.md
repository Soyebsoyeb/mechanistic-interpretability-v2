# Chapter 16 — Sparse Autoencoders

## Motivation
SAEs decompose activations into sparse, interpretable features.

## Formalization
Encoder: $z = f(W_e x + b_e)$  
Decoder: $\hat{x} = W_d z + b_d$  
Loss: $\mathcal{L} = \|x - \hat{x}\|_2^2 + \lambda S(z)$

## Implementation

```python
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

def train_sae(activations, d_in, d_hidden, epochs=100, lr=1e-3):
    sae = SparseAutoencoder(d_in, d_hidden)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    dataset = torch.utils.data.TensorDataset(activations)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1024, shuffle=True)
    for epoch in range(epochs):
        for batch in loader:
            opt.zero_grad()
            out = sae(batch[0])
            out['total_loss'].backward()
            opt.step()
            sae.normalize_decoder()
    return sae
```

## Practical Issues
- Dead features: Resampling, auxiliary losses
- Feature splitting: Merge analysis
- Feature absorption: Increase sparsity

## Falsification
Falsified if feature does not reconstruct when ablated.

## Exercises
- **Mathematical**: Derive gradient of L1 penalty w.r.t. $W_e$.
- **Implementation**: Implement top-k SAE.
- **Experimental**: Train SAE on GPT-2 layer; catalogue top-20 features.
- **Research**: Do SAE features align with path-patching circuits?

## References
- Bricken, T., et al. (2023). "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning."

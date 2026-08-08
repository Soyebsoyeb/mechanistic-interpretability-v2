# Chapter 16 — Sparse Autoencoders

## Motivation

Sparse autoencoders (SAEs) learn to decompose neural network activations into sparse, interpretable features. They are currently one of the most promising tools for feature discovery in large language models, scaling to billions of parameters and millions of features.

## Formalization

Encoder: $z = f(W_e x + b_e)$  
Decoder: $\hat{x} = W_d z + b_d$  
Objective: $\mathcal{L} = \|x - \hat{x}\|_2^2 + \lambda S(z)$

Common sparsity penalties:
- $S(z) = \|z\|_1$ (L1 sparsity)
- $S(z) = \text{top-}k(z)$ (k-sparse)
- $S(z) = \text{KL}(\hat{\rho} \| \rho)$ (target activation $ho$)

### Key Hyperparameters

| Parameter | Description | Typical Value |
|-----------|-------------|---------------|
| $d_{\text{hidden}}$ | SAE hidden dimension | $\sim$ 4-8$\times$ input dim |
| $\lambda$ | Sparsity coefficient | $10^{-3}$ to $10^{-5}$ |
| $S(z)$ | Sparsity penalty | L1 or Top-K |

## Implementation

```python
class SparseAutoencoder(nn.Module):
    def __init__(self, d_in, d_hidden, sparsity_lambda=1e-3, sparsity_penalty='l1'):
        super().__init__()
        self.d_in = d_in
        self.d_hidden = d_hidden
        self.lambda_reg = sparsity_lambda
        self.penalty = sparsity_penalty

        self.W_e = nn.Parameter(torch.randn(d_in, d_hidden) * 0.01)
        self.b_e = nn.Parameter(torch.zeros(d_hidden))
        self.W_d = nn.Parameter(torch.randn(d_hidden, d_in) * 0.01)
        self.b_d = nn.Parameter(torch.zeros(d_in))

        with torch.no_grad():
            self.W_d.data = self.W_d.data / self.W_d.data.norm(dim=1, keepdim=True)

    def encode(self, x):
        return torch.relu(x @ self.W_e + self.b_e)

    def decode(self, z):
        return z @ self.W_d + self.b_d

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        recon_loss = ((x_hat - x) ** 2).mean()
        sparsity_loss = self.lambda_reg * z.abs().mean()
        total_loss = recon_loss + sparsity_loss
        return {
            'x_hat': x_hat, 'z': z,
            'recon_loss': recon_loss, 'sparsity_loss': sparsity_loss,
            'total_loss': total_loss,
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

| Issue | Description | Mitigation |
|-------|-------------|------------|
| Dead features | Never activate | Resampling, auxiliary losses |
| Feature splitting | One true feature → multiple SAE features | Merge analysis |
| Feature absorption | Features absorb unrelated variance | Increase sparsity |
| Reconstruction error | High loss | Larger hidden dim, better training |

## Falsification
An SAE feature is falsified if:
- It does not reconstruct when ablated
- Its top activating examples are not semantically coherent
- It does not have a causal effect on model behavior

## Exercises
- **Mathematical**: Derive gradient of L1 penalty w.r.t. $W_e$.
- **Implementation**: Implement top-k SAE.
- **Experimental**: Train SAE on GPT-2 layer; catalogue top-20 features.
- **Research**: Do SAE features align with path-patching circuits?

## References

- Bricken, T., et al. (2023). "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning."
- Cunningham, H., et al. (2023). "Sparse Autoencoders Find Highly Interpretable Features in Language Models."

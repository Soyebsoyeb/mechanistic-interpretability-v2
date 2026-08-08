# Chapter 45 — Sparse Autoencoders From Scratch

## Full Implementation
```python
class SAEFromScratch(nn.Module):
    def __init__(self, d_in, d_hidden, l1_coeff=1e-3):
        super().__init__()
        self.W_enc = nn.Parameter(torch.randn(d_in, d_hidden) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(d_hidden))
        self.W_dec = nn.Parameter(torch.randn(d_hidden, d_in) * 0.01)
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        self.l1_coeff = l1_coeff

    def forward(self, x):
        z = torch.relu(x @ self.W_enc + self.b_enc)
        x_hat = z @ self.W_dec + self.b_dec
        loss = ((x_hat - x).pow(2).mean() + self.l1_coeff * z.abs().mean())
        return x_hat, z, loss

    def train_sae(self, activations, epochs=100, lr=1e-3, batch_size=1024):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        dataset = torch.utils.data.TensorDataset(activations)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            for batch in loader:
                opt.zero_grad()
                _, _, loss = self(batch[0])
                loss.backward()
                opt.step()
                with torch.no_grad():
                    self.W_dec.data /= (self.W_dec.data.norm(dim=1, keepdim=True) + 1e-8)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: {loss.item():.4f}")
```

## Logging
- Reconstruction loss
- Sparsity
- Active features
- Dead features
- Validation loss

## Exercises
- **Implementation**: Add dead feature resampling.
- **Experimental**: Train SAE on transformer layer; inspect features.

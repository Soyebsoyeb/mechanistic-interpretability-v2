# Chapter 14 — Superposition

## Motivation
When $n$ features are represented in dimension $d$ with $n > d$, models use superposition.

## Formalization
$$x = \sum_{i=1}^n f_i v_i + \varepsilon$$

Interference: $\frac{|v_i^\top v_j|}{\|v_i\| \|v_j\|}$. For sparse features, high interference is tolerable.

## Toy Experiment

```python
def superposition_experiment(n_features=100, d_model=20, sparsity=0.1):
    features = torch.randn(n_features, d_model)
    features = features / features.norm(dim=1, keepdim=True)
    acts = (torch.rand(10000, n_features) < sparsity).float()
    acts = acts * torch.randn(10000, n_features)
    targets = acts @ features
    model = nn.Linear(d_model, n_features)
    # Train...
    decoder_dirs = model.weight.T
    decoder_dirs = decoder_dirs / decoder_dirs.norm(dim=0, keepdim=True)
    overlap = (decoder_dirs.T @ decoder_dirs).abs()
    overlap.fill_diagonal_(0)
    return {"mean_overlap": overlap.mean().item()}
```

## Falsification
Falsified if orthogonal features achieve comparable performance.

## Exercises
- **Mathematical**: Prove bound on average pairwise inner product.
- **Implementation**: Extend to nonlinear autoencoders.
- **Experimental**: Measure feature overlap in trained transformer.
- **Research**: Superposition in early vs. late layers?

## References
- Elhage, N., et al. (2022). "Superposition, Memorization, and Double Descent."

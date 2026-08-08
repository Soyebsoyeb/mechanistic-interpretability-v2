# Chapter 14 — Superposition

## Motivation

When $n$ features are represented in dimension $d$ with $n > d$, the model cannot assign orthogonal directions to each feature. It must use superposition: representing more features than dimensions by accepting some interference. This is not a bug but a fundamental geometric necessity.

## Formalization

Representation: $x = \sum_{i=1}^n f_i v_i + \varepsilon$

If $v_i$ cannot all be orthogonal, interference occurs:

$$\text{Interference}(i,j) = \frac{|v_i^\top v_j|}{\|v_i\| \|v_j\|}$$

For sparse features (few $f_i$ active at once), high interference is tolerable because active features rarely collide.

### The Superposition Hypothesis

A model uses superposition when:
1. $n > d$ (more features than dimensions)
2. Features are sparse (low simultaneous activation probability)
3. The model achieves lower reconstruction error than a basis with $n = d$ orthogonal features

### Geometric Bounds

For $n$ unit vectors in $d$ dimensions, the average pairwise inner product satisfies:

$$\sum_{i \neq j} (v_i^\top v_j)^2 \geq \frac{n(n-d)}{d(n-1)}$$

This lower bound shows that superposition is unavoidable when $n > d$.

## Toy Experiment

```python
def superposition_experiment(n_features=100, d_model=20, sparsity=0.1, n_samples=10000):
    features = torch.randn(n_features, d_model)
    features = features / features.norm(dim=1, keepdim=True)
    acts = (torch.rand(n_samples, n_features) < sparsity).float()
    acts = acts * torch.randn(n_samples, n_features)
    targets = acts @ features

    model = nn.Linear(d_model, n_features)
    # Train...
    decoder_dirs = model.weight.T
    decoder_dirs = decoder_dirs / decoder_dirs.norm(dim=0, keepdim=True)
    overlap = (decoder_dirs.T @ decoder_dirs).abs()
    overlap.fill_diagonal_(0)
    return {"recon_error": loss.item(), "mean_overlap": overlap.mean().item()}
```

## Measurement
- Reconstruction error
- Feature overlap
- Sparsity
- Dimensionality ratio $n/d$

## Falsification
Falsified if orthogonal features achieve comparable performance, or if feature directions are nearly orthogonal.

## Exercises
- **Mathematical**: Prove the geometric bound on average pairwise inner product.
- **Implementation**: Extend to nonlinear autoencoders.
- **Experimental**: Measure feature overlap in trained transformer using SAE directions.
- **Research**: Is superposition more prevalent in early or late layers?

## References

- Elhage, N., et al. (2022). "Superposition, Memorization, and Double Descent."
- Schmidhuber, J. (1992). "Learning Factorial Codes by Predictability Minimization."

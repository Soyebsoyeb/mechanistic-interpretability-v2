# Chapter 15 — Sparse Coding

## Motivation

Sparse coding represents data as sparse linear combinations of dictionary atoms. It provides a mathematical foundation for understanding how neural networks might represent features efficiently, and it predates deep learning as a model of neural representation in neuroscience.

## Formalization

Represent $x \approx Dz$ where:
- $D \in \mathbb{R}^{d \times n}$ is the dictionary (columns are atoms)
- $z \in \mathbb{R}^n$ is the sparse code

Objective:
$$\mathcal{L} = \|x - Dz\|_2^2 + \lambda \|z\|_1$$

The first term encourages reconstruction. The second encourages sparsity.

### Properties

- **Overcomplete dictionary**: $n > d$ allows richer representations
- **Sparse prior**: Only few atoms active at once
- **Local competition**: Atoms compete to explain input variance

## Implementation

```python
def ista(x, D, lambda_reg, max_iter=100, lr=0.1):
    z = torch.zeros(D.shape[1], device=x.device)
    for _ in range(max_iter):
        grad = D.T @ (D @ z - x)
        z_temp = z - lr * grad
        z = torch.sign(z_temp) * torch.relu(torch.abs(z_temp) - lambda_reg * lr)
    return z

def learn_dictionary(data, n_atoms, lambda_reg, n_iter=1000):
    N, d = data.shape
    D = torch.randn(d, n_atoms)
    D = D / D.norm(dim=0, keepdim=True)
    for i in range(n_iter):
        idx = torch.randint(0, N, (1,)).item()
        z = ista(data[idx], D, lambda_reg)
        recon = D @ z
        error = recon - data[idx]
        D = D - 0.01 * error.unsqueeze(1) @ z.unsqueeze(0).T
        D = D / (D.norm(dim=0, keepdim=True) + 1e-8)
    return D
```

## Measurement
- Reconstruction error
- Sparsity ($\|z\|_0$ or $\|z\|_1$)
- Dictionary coherence: $\max_{i \neq j} |d_i^\top d_j|$

## Falsification
Falsified if data is not well-approximated by sparse combinations, or if dictionary atoms are uninterpretable.

## Exercises
- **Mathematical**: Analyze soft thresholding operator.
- **Implementation**: Implement OMP.
- **Experimental**: Learn sparse code for image patches; visualize atoms.
- **Research**: Compare sparse coding vs SAEs on transformer activations.

## References

- Olshausen, B. A., & Field, D. J. (1997). "Sparse Coding with an Overcomplete Basis Set."

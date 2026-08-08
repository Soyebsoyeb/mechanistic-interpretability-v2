# Chapter 3 — Linear Algebra for Interpretability

## Motivation
Neural networks operate in high-dimensional vector spaces. Every weight matrix is a linear map, every activation is a vector. Fluent command of linear algebra is essential.

## Vector Spaces and Inner Products

For $x, y \in \mathbb{R}^d$:
- Inner product: $\langle x, y \rangle = x^\top y = \sum_{i=1}^d x_i y_i$
- Euclidean norm: $\|x\|_2 = \sqrt{x^\top x}$
- Cosine similarity: $\cos(x, y) = \frac{x^\top y}{\|x\|_2 \|y\|_2}$

## Projection

For unit vector $v$:
$$P_v(x) = (x^\top v) v$$

Scalar coordinate: $a = v^\top x$. If $v$ represents a feature, $a$ measures feature strength.

## Subspaces

For orthonormal basis $U = [u_1, \ldots, u_k]$:
$$P_V(x) = UU^\top x$$

## Singular Value Decomposition

For $W \in \mathbb{R}^{m \times n}$:
$$W = U \Sigma V^\top$$

Applications: rank analysis, dominant directions, low-rank structure, compression, feature geometry.

## Change of Basis

Representation $x$ has coordinates $[x]_B = B^{-1}x$ under basis $B$. Neuron-level interpretation depends on parameterization. Mechanistic claims should be evaluated functionally.

## Implementation

```python
import torch
import torch.linalg as LA

def project_onto_direction(x, v):
    coordinate = x @ v
    projection = coordinate.unsqueeze(-1) * v
    return projection, coordinate

def analyze_weight_matrix(W, k=10):
    U, S, Vh = LA.svd(W, full_matrices=False)
    p = S / S.sum()
    effective_rank = torch.exp(-(p * torch.log(p + 1e-10)).sum())
    return {
        "singular_values": S[:k],
        "condition_number": (S.max() / (S.min() + 1e-10)).item(),
        "effective_rank": effective_rank.item(),
        "rank": (S > 1e-5).sum().item()
    }

def cosine_similarity_matrix(X, Y):
    X_norm = X / (LA.norm(X, dim=1, keepdim=True) + 1e-8)
    Y_norm = Y / (LA.norm(Y, dim=1, keepdim=True) + 1e-8)
    return X_norm @ Y_norm.T
```

## Measurement
1. Orthogonality: Gram matrix $G_{ij} = \cos(v_i, v_j)$
2. Feature overlap: Correlation of $a_i = v_i^\top x$
3. Subspace dimension: Explained variance ratio

## Intervention
```python
def ablate_subspace(x, basis_vectors):
    coeffs = x @ basis_vectors.T
    projection = coeffs @ basis_vectors
    return x - projection
```

## Falsification
Falsified if direction is not distinguishable from random, or if SVD does not show dominant singular vector aligned with $v$.

## Exercises
- **Mathematical**: Prove $\|W\|_2 = \sigma_{\max}(W)$. Show $P_v$ is orthogonal projection.
- **Implementation**: Compute principal angles between subspaces. Verify SVD reconstruction.
- **Experimental**: Compute SVD of each weight matrix in trained transformer. Plot spectra.
- **Research**: Do top SVD vectors of $W_{in}$ align with SAE-discovered features?

## References
- Strang, G. (2016). *Introduction to Linear Algebra*.
- Elhage, N., et al. (2022). "Superposition, Memorization, and Double Descent."

# Chapter 3 — Linear Algebra for Interpretability

## Motivation

Neural networks operate in high-dimensional vector spaces. Every weight matrix is a linear map, every activation is a vector, and every layer transforms representations through linear operations composed with nonlinearities. To reverse engineer these systems, we need fluent command of the linear algebra that governs them — not as abstract theory, but as a practical toolkit for analyzing, measuring, and intervening on internal states.

## Learning Objectives

- Compute projections onto interpretable directions and subspaces
- Apply SVD to analyze weight matrices, representations, and feature geometry
- Understand change-of-basis and its implications for feature interpretation
- Use subspace analysis to identify feature spaces and measure interference
- Compute principal angles between subspaces
- Apply linear algebra to design interventions

## Vector Spaces and Inner Products

We work primarily in finite-dimensional real vector spaces $\mathbb{R}^d$ equipped with the standard inner product.

For $x, y \in \mathbb{R}^d$:
- **Inner product**: $\langle x, y \rangle = x^\top y = \sum_{i=1}^d x_i y_i$
- **Euclidean norm**: $\|x\|_2 = \sqrt{x^\top x}$
- **Cosine similarity**: $\cos(x, y) = \frac{x^\top y}{\|x\|_2 \|y\|_2} \in [-1, 1]$

### Geometric Interpretation

The inner product measures alignment. When $\cos(x, y) = 1$, the vectors point in the same direction. When $\cos(x, y) = 0$, they are orthogonal. When $\cos(x, y) = -1$, they point in opposite directions.

In interpretability, cosine similarity is the standard metric for comparing feature directions, model directions, and representation directions.

## Projection

For a unit vector $v \in \mathbb{R}^d$ ($\|v\|_2 = 1$), the **orthogonal projection** of $x$ onto $v$ is:

$$P_v(x) = (x^\top v) v$$

The scalar **coordinate** (activation) along $v$ is:

$$a = v^\top x$$

This is the fundamental operation of feature-level interpretability: if $v$ represents a feature direction, then $a$ measures the strength of that feature in representation $x$.

### Projection Matrix

For a unit vector $v$, the projection matrix is $P = vv^\top$. Properties:
- $P^2 = P$ (idempotent)
- $P^\top = P$ (symmetric)
- $\text{rank}(P) = 1$

## Subspaces

A **subspace** $V \subseteq \mathbb{R}^d$ is a linear subspace. For an orthonormal basis $U = [u_1, \ldots, u_k] \in \mathbb{R}^{d \times k}$:

$$P_V(x) = UU^\top x$$

The residual is $x - P_V(x)$, which is orthogonal to $V$ by construction.

### Feature Subspaces

A concept may be represented not by a single direction but by a subspace. For example, "grammatical number" might be represented by a 2D subspace spanned by singular and plural directions. The projection onto this subspace measures the total "number information" in the representation.

## Singular Value Decomposition (SVD)

For $W \in \mathbb{R}^{m \times n}$:

$$W = U \Sigma V^\top$$

where:
- $U \in \mathbb{R}^{m \times m}$ orthogonal (left singular vectors)
- $\Sigma \in \mathbb{R}^{m \times n}$ diagonal with $\sigma_1 \geq \sigma_2 \geq \ldots \geq 0$
- $V \in \mathbb{R}^{n \times n}$ orthogonal (right singular vectors)

### Interpretability Applications

| Application | SVD Component | Interpretation |
|-------------|---------------|----------------|
| Rank analysis | Number of nonzero $\sigma_i$ | Effective dimensionality |
| Dominant directions | Top columns of $U$ and $V$ | Principal input-output mappings |
| Low-rank structure | $\sigma_k \gg \sigma_{k+1}$ | Matrix is approximately rank-$k$ |
| Compression | Truncated SVD | Fewer parameters, similar function |
| Feature geometry | Singular values | "Stretching" along feature directions |
| Condition number | $\sigma_{\max} / \sigma_{\min}$ | Sensitivity to perturbations |

### Effective Rank

The effective rank measures how many dimensions are "actively used":

$$\text{erank}(W) = \exp\left(-\sum_i p_i \log p_i\right)$$

where $p_i = \sigma_i / \sum_j \sigma_j$. For a perfectly low-rank matrix with $k$ equal singular values and the rest zero, $\text{erank} = k$.

## Change of Basis

A representation $x$ has coordinates $[x]_B = B^{-1}x$ under basis $B$. Since neural network weights parameterize specific bases, neuron-level interpretation can depend on parameterization.

**Key insight**: Mechanistic claims should be evaluated functionally (what does the network compute?) rather than structurally (what do individual neurons represent?) whenever possible. Two networks with identical functions but different parameterizations should have identical mechanistic explanations.

### Orthogonal Equivalence

For any orthogonal matrix $Q$, replacing $W \leftarrow WQ^\top$ and subsequent $W' \leftarrow QW'$ preserves the network function but changes neuron-level interpretations. This is why feature-level analysis is more robust than neuron-level analysis.

## Implementation

```python
import torch
import torch.linalg as LA

def project_onto_direction(x: torch.Tensor, v: torch.Tensor) -> tuple:
    """Project x onto unit direction v.

    Args:
        x: torch.Tensor [..., d]
        v: torch.Tensor [d], unit vector

    Returns:
        projection: torch.Tensor [..., d]
        coordinate: torch.Tensor [...]
    """
    v = v / (v.norm() + 1e-8)
    coordinate = x @ v  # [...]
    projection = coordinate.unsqueeze(-1) * v  # [..., d]
    return projection, coordinate


def project_onto_subspace(x: torch.Tensor, basis: torch.Tensor) -> tuple:
    """Project x onto orthonormal subspace.

    Args:
        x: torch.Tensor [..., d]
        basis: torch.Tensor [k, d], orthonormal rows

    Returns:
        projection: torch.Tensor [..., d]
        coefficients: torch.Tensor [..., k]
    """
    coefficients = x @ basis.T  # [..., k]
    projection = coefficients @ basis  # [..., d]
    return projection, coefficients


def analyze_weight_matrix(W: torch.Tensor, k: int = 10) -> dict:
    """Analyze weight matrix via SVD.

    Args:
        W: torch.Tensor [m, n]
        k: int, number of singular values to report

    Returns:
        dict with singular values, condition number, effective rank
    """
    U, S, Vh = LA.svd(W, full_matrices=False)

    # Effective rank (entropy-based)
    p = S / (S.sum() + 1e-10)
    effective_rank = torch.exp(-(p * torch.log(p + 1e-10)).sum())

    # Condition number
    cond = S.max() / (S[S > 1e-8].min() + 1e-10)

    return {
        "singular_values": S[:k].tolist(),
        "condition_number": cond.item(),
        "effective_rank": effective_rank.item(),
        "rank": (S > 1e-5).sum().item(),
        "left_singular_vectors": U[:, :k],
        "right_singular_vectors": Vh[:k, :]
    }


def cosine_similarity_matrix(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity between rows of X and Y.

    Args:
        X: torch.Tensor [n, d]
        Y: torch.Tensor [m, d]

    Returns:
        torch.Tensor [n, m]
    """
    X_norm = X / (LA.norm(X, dim=1, keepdim=True) + 1e-8)
    Y_norm = Y / (LA.norm(Y, dim=1, keepdim=True) + 1e-8)
    return X_norm @ Y_norm.T


def principal_angles(U1: torch.Tensor, U2: torch.Tensor) -> torch.Tensor:
    """Compute principal angles between subspaces spanned by U1 and U2.

    Args:
        U1: torch.Tensor [d, k1], orthonormal columns
        U2: torch.Tensor [d, k2], orthonormal columns

    Returns:
        torch.Tensor [min(k1, k2)], principal angles in radians
    """
    M = U1.T @ U2  # [k1, k2]
    _, s, _ = LA.svd(M)
    # s are cosines of principal angles
    s = torch.clamp(s, -1.0, 1.0)
    angles = torch.arccos(s)
    return angles
```

## Measurement: Quantifying Feature Geometry

Given a set of feature directions $\{v_1, \ldots, v_k\}$:

1. **Orthogonality**: Compute Gram matrix $G_{ij} = \cos(v_i, v_j)$. Near-identity indicates orthogonal features; off-diagonal entries indicate superposition.

2. **Feature overlap**: For representation $x$, measure $a_i = v_i^\top x$ and correlation $\text{Corr}(a_i, a_j)$ across a dataset.

3. **Subspace dimension**: Compute the explained variance ratio for the top-$k$ subspace.

4. **Principal angles**: Between two feature subspaces, measure how aligned they are.

## Intervention: Subspace Ablation

```python
def ablate_subspace(x: torch.Tensor, basis_vectors: torch.Tensor) -> torch.Tensor:
    """Remove component of x lying in subspace spanned by basis_vectors.

    Args:
        x: torch.Tensor [..., d]
        basis_vectors: torch.Tensor [k, d], orthonormal

    Returns:
        torch.Tensor [..., d], x with subspace removed
    """
    coeffs = x @ basis_vectors.T  # [..., k]
    projection = coeffs @ basis_vectors  # [..., d]
    return x - projection


def ablate_direction(x: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """Remove component of x along direction.

    Args:
        direction: torch.Tensor [d], unit vector
    """
    direction = direction / (direction.norm() + 1e-8)
    coord = x @ direction  # [...]
    projection = coord.unsqueeze(-1) * direction  # [..., d]
    return x - projection
```

## Falsification

A claimed feature direction $v$ is falsified if:
- $v$ is not statistically distinguishable from a random direction in prediction tasks
- Ablating $v$ does not change model behavior
- The SVD of the relevant weight matrix does not show a dominant singular vector aligned with $v$
- Principal angles between the claimed subspace and an independently discovered subspace are large

## Reproduction

All linear algebra operations must be:
- Deterministic (document random seed if any)
- Documented with input/output shapes
- Verified against reference implementations
- Tested for numerical stability

## Exercises

### Mathematical
1. Prove that for any weight matrix $W$, the operator norm $\|W\|_2 = \sigma_{\max}(W)$.
2. Show that if $v$ is a unit vector, then $P = vv^\top$ is an orthogonal projection ($P^2 = P$ and $P^\top = P$).
3. Prove that the effective rank satisfies $1 \leq \text{erank}(W) \leq \text{rank}(W)$.
4. Show that principal angles between subspaces are invariant under orthogonal transformations.

### Implementation
5. Implement a function that computes the principal angles between two subspaces and verify it against a reference.
6. Write a test that verifies SVD reconstruction: $U \Sigma V^\top = W$ for random matrices.
7. Implement a function that finds the nearest orthonormal basis to a given set of vectors using Gram-Schmidt.

### Experimental
8. Compute the SVD of each weight matrix in a trained transformer. Plot the singular value spectra on a log scale. Identify layers with low-rank structure (effective rank $\ll$ dimension).

### Research
9. Investigate whether the top singular vectors of $W_{in}$ in an MLP align with interpretable feature directions discovered by sparse autoencoders. Quantify alignment using principal angles.
10. Study how the condition number of attention weight matrices varies across layers and heads. Does high condition number correlate with interpretability difficulty?

## References

- Strang, G. (2016). *Introduction to Linear Algebra*, 5th ed.
- Golub, G. H., & Van Loan, C. F. (2013). *Matrix Computations*, 4th ed.
- Elhage, N., et al. (2022). "Superposition, Memorization, and Double Descent."

# Chapter 2 — Neural Networks as Computational Systems

## Motivation

Before we can reverse engineer a neural network, we must understand what kind of object it is. A neural network is not merely a function approximator; it is a **computational system** composed of primitive operations arranged in a directed graph. Mechanistic interpretability attempts to simplify this graph into a meaningful computational structure that reveals the algorithm being executed.

## Learning Objectives

- Represent neural networks as computational graphs with explicit node and edge semantics
- Understand internal representations as vectors in high-dimensional space with information-theoretic content
- Distinguish local from distributed representations and their implications for interpretation
- Identify features within representations using linear, nonlinear, and subspace decompositions
- Recognize that representation structure emerges from the interaction of architecture, optimization, data, and objective

## Formal Definition: Computational Graphs

A neural network can be represented as a directed acyclic graph (DAG) where each node $z_i$ computes:

$$z_i = f_i(z_{\text{parents}(i)})$$

The output $y$ is the composition of these transformations along all directed paths from input $x$ to output:

$$y = f_L \circ f_{L-1} \circ \cdots \circ f_1(x)$$

For a feedforward network with $L$ layers:

$$h^{(0)} = x$$
$$h^{(\ell)} = f^{(\ell)}(W^{(\ell)} h^{(\ell-1)} + b^{(\ell)})$$
$$y = h^{(L)}$$

### Computational Graph as a Mechanistic Object

The full computational graph contains every scalar multiplication and addition. For a transformer with $L$ layers, $d$ dimensions, and $n$ sequence length, this graph contains $O(L \cdot n^2 \cdot d^2)$ operations. Mechanistic interpretability seeks a **simplified graph** $G' = (V', E')$ where:
- Nodes $V'$ represent meaningful computational units (features, heads, circuits, layers)
- Edges $E'$ represent information flow between these units
- The simplified graph preserves the target behavior $B(x)$ for relevant inputs $x$
- The simplified graph is significantly smaller than the full graph

### Example: Simplifying an MLP

A 3-layer MLP with 512 hidden units has $O(512^2)$ edges in its computational graph. A mechanistic explanation might reduce this to: "Layer 1 detects edges, Layer 2 detects textures, Layer 3 detects objects." This is a vast simplification — but it is only valid if we can validate each claim with causal evidence.

## Internal Representations

A **representation** is a vector $x \in \mathbb{R}^d$. The representation may contain information about many variables simultaneously, encoded in different geometric structures.

### Linear Decomposition

For features $v_1, \ldots, v_k$, the representation can be decomposed as:

$$x = \sum_{i=1}^k a_i v_i + \varepsilon$$

where:
- $v_i \in \mathbb{R}^d$ are feature directions (not necessarily orthogonal)
- $a_i \in \mathbb{R}$ are feature coefficients (activations)
- $\varepsilon$ is residual noise (unmodeled variance)

### Information-Theoretic View

The representation $h$ carries information about target variable $Y$ if $I(h; Y) > 0$. However:
- $I(h; Y) > 0$ does **not** imply $h$ causes $Y$
- $I(h; Y) > 0$ does **not** imply $Y$ causes $h$
- $I(h; Y) > 0$ only implies statistical dependence

Causal relevance requires intervention, not just information.

### Distributed Representations

A concept need not correspond to a single neuron. It may correspond to:

| Structure | Mathematical Form | Interpretability Implication |
|-----------|-------------------|------------------------------|
| Single neuron | $f(x) = x_i$ | Easy to identify, often polysemantic |
| Direction | $f(x) = v^\top x$ | Requires finding $v$, more robust |
| Subspace | $f(x) = \|P_V x\|$ | Requires finding basis $V$ |
| Sparse combination | $f(x) = \sum_{i \in S} a_i (v_i^\top x)$ | Requires sparse coding or SAE |
| Nonlinear manifold | $f(x) = g(\phi(x))$ | Requires nonlinear probes |
| Circuit | Distributed across layers | Requires graph analysis |

This motivates feature-level analysis rather than neuron-level analysis. A neuron is a coordinate; a feature is a functional pattern. The distinction is fundamental.

## Implementation: Representation Analysis

```python
import torch
import torch.nn as nn
from typing import Dict, Tuple

def extract_representation(
    model: nn.Module,
    layer_name: str,
    inputs: torch.Tensor
) -> torch.Tensor:
    """Extract hidden representation from a specific layer.

    Args:
        model: nn.Module, the neural network
        layer_name: str, dot-path to layer, e.g. "encoder.layer2"
        inputs: torch.Tensor of shape [batch, ...]

    Returns:
        torch.Tensor of shape [batch, ..., d_model]
    """
    representations: Dict[str, torch.Tensor] = {}

    def hook_fn(module, input, output):
        # Handle tuple outputs (common in transformer blocks)
        if isinstance(output, tuple):
            representations["output"] = output[0].detach()
        else:
            representations["output"] = output.detach()

    # Navigate to target layer using dot notation
    target = model
    for part in layer_name.split("."):
        target = getattr(target, part)

    handle = target.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        _ = model(inputs)

    handle.remove()
    return representations["output"]


def decompose_representation(
    x: torch.Tensor,
    feature_directions: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decompose representation along given feature directions.

    Args:
        x: torch.Tensor [..., d_model]
        feature_directions: torch.Tensor [k, d_model], unit vectors

    Returns:
        coefficients: torch.Tensor [..., k]
        reconstruction: torch.Tensor [..., d_model]
    """
    # x: [..., d]
    # directions: [k, d]
    coefficients = x @ feature_directions.T  # [..., k]
    reconstruction = coefficients @ feature_directions  # [..., d]
    return coefficients, reconstruction


def compute_representation_geometry(representations: torch.Tensor) -> Dict[str, float]:
    """Compute geometric properties of a set of representations.

    Args:
        representations: [N, d]

    Returns:
        Dict with mean norm, norm std, pairwise cosine stats
    """
    norms = representations.norm(dim=1)
    normalized = representations / (norms.unsqueeze(1) + 1e-8)
    cosine_matrix = normalized @ normalized.T

    # Exclude diagonal
    mask = ~torch.eye(len(representations), dtype=torch.bool)
    off_diagonal = cosine_matrix[mask]

    return {
        "mean_norm": norms.mean().item(),
        "std_norm": norms.std().item(),
        "mean_cosine": off_diagonal.mean().item(),
        "std_cosine": off_diagonal.std().item(),
        "max_cosine": off_diagonal.max().item(),
        "min_cosine": off_diagonal.min().item()
    }
```

## Measurement: Quantifying Representation Content

Given representation $h$ and target variable $Y$:

1. **Linear predictability**: Train a linear probe $\hat{Y} = W h + b$. Report $R^2$ or accuracy. This measures whether $Y$ is linearly decodable from $h$.

2. **Nonlinear predictability**: Train an MLP probe. If nonlinear probe $\gg$ linear probe, $Y$ is encoded nonlinearly.

3. **Mutual information**: Estimate $I(h; Y)$ using binning, k-NN estimators, or neural estimators (MINE, InfoNCE).

4. **Causal effect**: Intervene on $h$ and measure change in $Y$. This is the gold standard but requires careful experimental design.

## Intervention: Modifying Representations

```python
def intervene_on_direction(
    model: nn.Module,
    inputs: torch.Tensor,
    layer_name: str,
    direction: torch.Tensor,
    scale: float
) -> torch.Tensor:
    """Add a vector along a specific direction in representation space.

    Args:
        direction: torch.Tensor [d_model], unit vector
        scale: float, magnitude of intervention

    Returns:
        Model output after intervention
    """
    direction = direction / (direction.norm() + 1e-8)

    def hook_fn(module, input, output):
        # output: [batch, seq, d_model] or [batch, d_model]
        intervention = scale * direction.view(1, 1, -1)
        # Handle different output shapes
        while intervention.dim() < output.dim():
            intervention = intervention.unsqueeze(0)
        return output + intervention.to(output.device)

    target = model
    for part in layer_name.split("."):
        target = getattr(target, part)

    handle = target.register_forward_hook(hook_fn)
    output = model(inputs)
    handle.remove()
    return output


def ablate_subspace(
    model: nn.Module,
    inputs: torch.Tensor,
    layer_name: str,
    basis_vectors: torch.Tensor
) -> torch.Tensor:
    """Remove component of representation lying in subspace spanned by basis_vectors.

    Args:
        basis_vectors: torch.Tensor [k, d_model], orthonormal basis

    Returns:
        Model output with subspace ablated
    """
    def hook_fn(module, input, output):
        # Project onto subspace and subtract
        coeffs = output @ basis_vectors.T  # [..., k]
        projection = coeffs @ basis_vectors  # [..., d]
        return output - projection

    target = model
    for part in layer_name.split("."):
        target = getattr(target, part)

    handle = target.register_forward_hook(hook_fn)
    output = model(inputs)
    handle.remove()
    return output
```

## Falsification

A representation hypothesis is falsified if:
- The feature direction does not predict the target variable on held-out data (poor generalization)
- Ablating the direction does not change the target behavior (no causal effect)
- An alternative direction explains more variance in the behavior (better explanation exists)
- The representation encodes the target variable only through a confounder (spurious correlation)

## Reproduction

To reproduce representation analysis:
1. Record model architecture and checkpoint hash
2. Record layer names and exact extraction points
3. Save feature directions or method for computing them
4. Document input distribution and preprocessing
5. Report random seeds and software versions
6. Save raw representations and analysis code

## Alternative Explanations

- The representation may encode the target variable *and* other variables simultaneously (multiplexing)
- The linear probe may exploit correlations rather than causal structure (spurious decoding)
- The feature direction may be a rotation of the true feature basis (basis ambiguity)
- The representation may be a nonlinear function of the target that a linear probe cannot capture

## Exercises

### Mathematical
1. Show that for any orthogonal matrix $Q$, replacing all weight matrices $W^{(\ell)}$ with $Q^{(\ell)} W^{(\ell)} (Q^{(\ell-1)})^\top$ preserves the network function but changes neuron-level interpretations. What does this imply for neuron-level mechanistic claims?
2. Prove that the set of all representations $\{h(x) : x \in \mathcal{X}\}$ forms a nonlinear manifold in $\mathbb{R}^d$. Under what conditions is this manifold approximately linear?

### Implementation
3. Implement a computational graph tracer for a simple MLP that returns the DAG as an adjacency matrix and a topological ordering.
4. Write a function that computes the principal angles between two subspaces using SVD.

### Experimental
5. Train a 3-layer MLP on a synthetic task with 5 known latent variables. Use PCA on hidden layers and measure how well each principal component aligns with each latent variable using $R^2$.

### Research
6. Prove or disprove: If a representation $h$ has high mutual information with $Y$, there exists an intervention on $h$ that changes $Y$. If false, provide a counterexample.
7. Investigate whether distributed representations in early layers of transformers are more or less interpretable than distributed representations in late layers.

## References

- Hinton, G. E. (1986). "Learning Distributed Representations."
- Bengio, Y., Courville, A., & Vincent, P. (2013). "Representation Learning: A Review." *IEEE TPAMI*.
- Elhage, N., et al. (2022). "Superposition, Memorization, and Double Descent."

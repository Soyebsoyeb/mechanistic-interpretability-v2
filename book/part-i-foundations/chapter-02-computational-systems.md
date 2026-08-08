# Chapter 2 — Neural Networks as Computational Systems

## Motivation

Before reverse engineering a neural network, we must understand what kind of object it is. A neural network is not merely a function approximator; it is a **computational system** composed of primitive operations arranged in a directed graph. Mechanistic interpretability attempts to simplify this graph into a meaningful computational structure.

## Computational Graphs

A neural network can be represented as a directed acyclic graph (DAG) where each node $z_i$ computes:

$$z_i = f_i(z_{\text{parents}(i)})$$

The output $y$ is the composition of transformations along paths from input $x$ to output.

For a feedforward network with $L$ layers:

$$h^{(0)} = x$$
$$h^{(\ell)} = f^{(\ell)}(W^{(\ell)} h^{(\ell-1)} + b^{(\ell)})$$
$$y = h^{(L)}$$

### Mechanistic Interpretability as Graph Simplification

The full computational graph contains every multiplication and addition. Mechanistic interpretability seeks a **simplified graph** $G' = (V', E')$ where nodes represent meaningful units and edges represent information flow, preserving target behavior $B(x)$.

## Internal Representations

A **representation** is a vector $x \in \mathbb{R}^d$. It may contain information about many variables simultaneously:

$$x = \sum_{i=1}^k a_i v_i + \varepsilon$$

where $v_i$ are feature directions, $a_i$ are coefficients, and $\varepsilon$ is residual noise.

### Distributed Representations

A concept may correspond to:
- **A direction**: $f(x) = v^\top x$
- **A subspace**: $f(x) = \|P_V x\|$
- **A sparse combination**: $f(x) = \sum_{i \in S} a_i (v_i^\top x)$
- **A nonlinear manifold**: $f(x) = g(\phi(x))$
- **A circuit**: Distributed across multiple layers

## Implementation

```python
import torch
import torch.nn as nn

def extract_representation(model, layer_name, inputs):
    representations = {}
    def hook_fn(module, input, output):
        representations["output"] = output.detach()
    target = model
    for part in layer_name.split("."):
        target = getattr(target, part)
    handle = target.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(inputs)
    handle.remove()
    return representations["output"]

def decompose_representation(x, feature_directions):
    coefficients = x @ feature_directions.T
    reconstruction = coefficients @ feature_directions
    return coefficients, reconstruction
```

## Measurement
1. Linear predictability: Train linear probe, report $R^2$
2. Mutual information: Estimate $I(h; Y)$
3. Causal effect: Intervene on $h$, measure change in $Y$

## Intervention
```python
def intervene_on_direction(model, inputs, direction, scale, layer_name):
    def hook_fn(module, input, output):
        intervention = scale * direction.view(1, 1, -1)
        return output + intervention.to(output.device)
    target = model
    for part in layer_name.split("."):
        target = getattr(target, part)
    handle = target.register_forward_hook(hook_fn)
    output = model(inputs)
    handle.remove()
    return output
```

## Falsification
Falsified if feature direction does not predict target on held-out data, or if ablation does not change behavior.

## Exercises
- **Mathematical**: Show that for orthogonal matrix $Q$, replacing $W$ with $WQ^\top$ preserves function but changes neuron interpretations.
- **Implementation**: Implement computational graph tracer for MLP returning adjacency matrix.
- **Experimental**: Train 3-layer MLP on synthetic task with 5 latent variables. Use PCA on hidden layers.
- **Research**: Prove or disprove: If $I(h; Y)$ is high, intervention on $h$ changes $Y$.

## References
- Hinton, G. E. (1986). "Learning Distributed Representations."
- Bengio, Y., et al. (2013). "Representation Learning: A Review." *IEEE TPAMI*.

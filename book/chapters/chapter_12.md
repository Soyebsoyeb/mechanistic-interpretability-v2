# Chapter 12 — Neurons, Features, and Directions

## Motivation

We must rigorously distinguish between a neuron (a coordinate in activation space) and a feature (a functional pattern). A concept may be distributed across many coordinates, making feature-level analysis more robust than neuron-level analysis. This chapter establishes the foundational vocabulary and methods for feature discovery.

## Learning Objectives

- Distinguish neurons from features and understand why the difference matters
- Identify feature directions using linear probes, contrastive examples, and statistical methods
- Understand distributed representations and their implications for interpretability
- Apply nonlinear feature discovery methods
- Recognize the basis ambiguity problem and its consequences

## Formalization

- **Neuron**: Coordinate $i$ in representation $x$, value $x_i$
- **Feature direction**: Unit vector $v$ where feature strength is $f(x) = v^\top x$
- **Nonlinear feature**: $f(x) = g(v^\top x + b)$ for nonlinear $g$
- **Feature subspace**: $f(x) = \|P_V x\|$ for projection $P_V$
- **Sparse feature**: $f(x) = \sum_{i \in S} a_i (v_i^\top x)$ where $|S| \ll d$

### The Neuron vs. Feature Distinction

| Aspect | Neuron | Feature |
|--------|--------|---------|
| Definition | Coordinate axis | Functional pattern |
| Interpretability | May be polysemantic | Should be monosemantic |
| Intervention | Easy (ablate one index) | Harder (requires subspace) |
| Discovery | Visual inspection | Requires analysis (PCA, SAE) |
| Robustness | Fragile to basis change | Invariant to basis change |

## Basis Ambiguity

For any orthogonal matrix $Q$, replacing all representations $x \rightarrow Qx$ and weights $W \rightarrow WQ^\top$ preserves the network function but changes neuron-level interpretations. This means:

> **Neuron-level interpretations are not functionally invariant.**

Feature directions, however, can be defined in a basis-invariant way (e.g., as directions that maximize some functional criterion).

## Implementation

```python
def extract_neuron_activations(model, dataset, layer_idx, neuron_idx):
    """Extract activation values for a specific neuron across dataset."""
    activations = []
    def hook_fn(module, input, output):
        activations.append(output[:, :, neuron_idx].detach().cpu())
    target = model.blocks[layer_idx]
    handle = target.register_forward_hook(hook_fn)
    model.eval()
    with torch.no_grad():
        for batch in dataset:
            _ = model(batch)
    handle.remove()
    return torch.cat([a.flatten() for a in activations])

def find_feature_direction(activations, labels, method='mean_diff'):
    """Find a direction in representation space that encodes a feature."""
    if method == 'mean_diff':
        pos_mean = activations[labels == 1].mean(dim=0)
        neg_mean = activations[labels == 0].mean(dim=0)
        direction = pos_mean - neg_mean
        return direction / (direction.norm() + 1e-8)
    elif method == 'linear_probe':
        W = torch.linalg.lstsq(activations, labels.float()).solution
        return W / (W.norm() + 1e-8)

def measure_feature_selectivity(activations, direction):
    scores = activations @ direction
    return (scores.max() / (scores.abs().mean() + 1e-8)).item()

def steer_feature(model, inputs, layer_idx, direction, scale):
    def hook_fn(module, input, output):
        return output + scale * direction.view(1, 1, -1)
    handle = model.blocks[layer_idx].register_forward_hook(hook_fn)
    result = model(inputs)
    handle.remove()
    return result
```

## Measurement
- Maximum activating examples
- Linear probe accuracy
- Selectivity
- Cosine similarity between feature direction and neuron direction

## Falsification
A claimed feature direction is falsified if:
- Linear probe accuracy is not significantly above random
- Steering the direction does not change the predicted behavior
- The direction is not statistically distinguishable from random directions

## Exercises
- **Mathematical**: Show orthogonal transformation preserves computations but changes neuron interpretations.
- **Implementation**: Find principal directions using PCA.
- **Experimental**: Assess polysemanticity of top variance neurons.
- **Research**: Do SAE features align with linear probe directions?

## References

- Hinton, G. E. (1986). "Learning Distributed Representations."
- Bengio, Y., et al. (2013). "Representation Learning: A Review."

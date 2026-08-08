# Chapter 12 — Neurons, Features, and Directions

## Motivation
We must distinguish between a neuron (coordinate) and a feature (functional pattern). Concepts may be distributed across many coordinates.

## Formalization
- **Neuron**: Coordinate $i$, value $x_i$
- **Feature direction**: Unit vector $v$ where $f(x) = v^\top x$
- **Nonlinear feature**: $f(x) = g(v^\top x + b)$
- **Feature subspace**: $f(x) = \|P_V x\|$

## Neuron vs. Feature

| Aspect | Neuron | Feature |
|--------|--------|---------|
| Definition | Coordinate axis | Functional pattern |
| Interpretability | May be polysemantic | Should be monosemantic |
| Intervention | Easy (ablate index) | Harder (subspace) |
| Discovery | Visual inspection | Requires analysis |

## Implementation

```python
def extract_neuron_activations(model, dataset, layer_idx, neuron_idx):
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
    if method == 'mean_diff':
        pos = activations[labels == 1].mean(dim=0)
        neg = activations[labels == 0].mean(dim=0)
        direction = pos - neg
        return direction / (direction.norm() + 1e-8)

def measure_feature_selectivity(activations, direction):
    scores = activations @ direction
    return (scores.max() / (scores.abs().mean() + 1e-8)).item()
```

## Intervention
```python
def steer_feature(model, inputs, layer_idx, direction, scale):
    def hook_fn(module, input, output):
        return output + scale * direction.view(1, 1, -1).to(output.device)
    handle = model.blocks[layer_idx].register_forward_hook(hook_fn)
    result = model(inputs)
    handle.remove()
    return result
```

## Falsification
Falsified if linear probe accuracy not above random, or steering does not change behavior.

## Exercises
- **Mathematical**: Show orthogonal transformation preserves computations but changes neuron interpretations.
- **Implementation**: Find principal directions using PCA.
- **Experimental**: Assess polysemanticity of top variance neurons.
- **Research**: Do SAE features align with linear probe directions?

## References
- Olah, C., et al. (2020). "Zoom In: An Introduction to Circuits." *Distill*.

# Chapter 23 — Attribution Graphs

## Motivation
Attribution graphs identify candidate components for causal testing. They are starting points, not proven circuits.

## Formalization
Graph $G = (V, E, S)$ where $S$ are attribution scores.

**Warning**: High attribution $\neq$ causation.

## Implementation

```python
def compute_attribution_graph(model, inputs, target, method='gradient'):
    if method == 'gradient':
        model.zero_grad()
        output = model(inputs)
        loss = compute_target_loss(output, target)
        loss.backward()
        scores = {}
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight.grad is not None:
                scores[name] = module.weight.grad.norm().item()
        return scores
```

## Falsification
Falsified if high-attribution components fail causal tests.

## Exercises
- **Mathematical**: Show gradient attribution equals zero for negative ReLU pre-activations.
- **Implementation**: Compare gradient vs. integrated gradients.
- **Experimental**: Compute attribution for IOI task. Validate with patching.
- **Research**: Do attribution methods systematically miss certain components?

## References
- Sundararajan, M., et al. (2017). "Axiomatic Attribution for Deep Networks."

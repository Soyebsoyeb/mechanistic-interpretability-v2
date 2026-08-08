# Chapter 23 — Attribution Graphs

## Motivation

Attribution graphs identify candidate components for causal testing. They are starting points, not proven circuits. A component with high attribution is not necessarily causal.

## Formalization

Graph $G = (V, E, S)$ where:
- $V$: components (heads, neurons, layers)
- $E$: information flow edges
- $S$: attribution scores

**Warning**: High attribution $\neq$ causation.

## Implementation

```python
def compute_attribution_graph(model, inputs, target, method='gradient'):
    if method == 'gradient':
        model.zero_grad()
        output = model(inputs)
        loss = compute_target_loss(output, target)
        loss.backward()

        attributions = {}
        for name, module in model.named_modules():
            if hasattr(module, 'weight') and module.weight.grad is not None:
                attributions[name] = module.weight.grad.norm().item()

    return attributions
```

## Falsification
Falsified if high-attribution components fail causal tests.

## Exercises
- **Mathematical**: Show gradient attribution equals zero for ReLU with negative pre-activation.
- **Implementation**: Compare gradient vs. integrated gradients.
- **Experimental**: Compute attribution graph for IOI task; validate top components.
- **Research**: Do attribution methods systematically miss certain component types?

## References

- Sundararajan, M., et al. (2017). "Axiomatic Attribution for Deep Networks."

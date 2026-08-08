# Chapter 19 — Activation Patching

## Motivation
Activation patching replaces a component's activation from corrupted run with clean run. It is the workhorse intervention.

## Formalization
Clean: $h_c = f(x_c)$  
Corrupted: $h_b = f(x_b)$  
Patched: $h_b^{(i)} \leftarrow h_c^{(i)}$

Normalized score:
$$S_i = \frac{M_{\text{patched}} - M_{\text{corrupted}}}{M_{\text{clean}} - M_{\text{corrupted}}}$$

- $S_i \approx 0$: little recovery
- $S_i \approx 1$: strong recovery
- $S_i < 0$: component harms behavior
- $S_i > 1$: over-recovery

## Implementation

```python
def activation_patch(model, clean_inputs, corrupted_inputs, component_name, metric_fn):
    clean_cache = {}
    def clean_hook(module, inputs, output):
        clean_cache['out'] = output.detach()
    target = dict(model.named_modules())[component_name]
    handle = target.register_forward_hook(clean_hook)
    with torch.no_grad():
        clean_metric = metric_fn(model(clean_inputs))
    handle.remove()

    with torch.no_grad():
        corrupted_metric = metric_fn(model(corrupted_inputs))

    def patch_hook(module, inputs, output):
        return clean_cache['out']
    handle = target.register_forward_hook(patch_hook)
    with torch.no_grad():
        patched_metric = metric_fn(model(corrupted_inputs))
    handle.remove()

    score = (patched_metric - corrupted_metric) / (clean_metric - corrupted_metric + 1e-10)
    return {
        "clean": clean_metric.item(),
        "corrupted": corrupted_metric.item(),
        "patched": patched_metric.item(),
        "score": score.item()
    }
```

## Controls
- Negative: Patch irrelevant component, expect $S \approx 0$
- Positive: Patch full clean state, expect $S \approx 1$

## Falsification
Falsified if patching does not recover behavior when it should, or if negative controls show high scores.

## Exercises
- **Mathematical**: Derive conditions for $S_i > 1$.
- **Implementation**: Build batch activation patching for all components.
- **Experimental**: Patch every head in 4-layer transformer. Plot heatmap.
- **Research**: Does patching overestimate importance due to redundancy?

## References
- Vig, J., et al. (2020). "Investigating Gender Bias in Language Models Using Causal Mediation Analysis."

# Chapter 19 — Activation Patching

## Motivation

Activation patching is the workhorse intervention of mechanistic interpretability. It replaces a component's activation from a corrupted run with its activation from a clean run, measuring whether that component is necessary for the behavior.

## Formalization

Clean: $h_c = f(x_c)$  
Corrupted: $h_b = f(x_b)$  
Patched: $h_b^{(i)} \leftarrow h_c^{(i)}$

Normalized score:
$$S_i = \frac{M_{\text{patched}} - M_{\text{corrupted}}}{M_{\text{clean}} - M_{\text{corrupted}}}$$

Interpretation:
- $S_i \approx 0$: little recovery
- $S_i \approx 1$: strong recovery
- $S_i < 0$: component harms behavior
- $S_i > 1$: over-recovery (investigate)

## Implementation

```python
def activation_patch(model, clean_inputs, corrupted_inputs,
                     component_name, metric_fn):
    clean_cache = run_and_cache(model, clean_inputs)
    clean_metric = metric_fn(model, clean_inputs)
    corrupted_metric = metric_fn(model, corrupted_inputs)

    def patch_hook(module, input, output):
        return clean_cache[component_name]

    target = navigate_to_component(model, component_name)
    handle = target.register_forward_hook(patch_hook)
    patched_metric = metric_fn(model, corrupted_inputs)
    handle.remove()

    score = (patched_metric - corrupted_metric) / (clean_metric - corrupted_metric + 1e-10)
    return {
        "clean": clean_metric,
        "corrupted": corrupted_metric,
        "patched": patched_metric,
        "score": score.item()
    }
```

## Controls
- Negative control: Patch irrelevant component, expect $S \approx 0$
- Positive control: Patch full clean state, expect $S \approx 1$

## Falsification
Falsified if patching does not recover behavior when it should, or if negative controls show high scores.

## Exercises
- **Mathematical**: Derive conditions under which $S_i > 1$.
- **Implementation**: Build batch activation patching for all components.
- **Experimental**: Patch every attention head in 4-layer transformer; plot heatmap.
- **Research**: Does activation patching overestimate component importance?

## References

- Vig, J. (2019). "Visualizing Attention in Transformer-Based Language Representation Models."

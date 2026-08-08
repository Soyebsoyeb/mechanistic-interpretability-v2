# Chapter 20 — Ablations

## Motivation

Ablations demonstrate necessity by removing a component and measuring behavior change. They are the simplest and most direct causal intervention.

## Types

| Type | Operation | Use Case |
|------|-----------|----------|
| Zero ablation | $h_i \leftarrow 0$ | Testing necessity |
| Mean ablation | $h_i \leftarrow \mathbb{E}[h_i]$ | Less distributional shift |
| Resampling | $h_i \leftarrow h_i^{(\text{random})}$ | Testing specific information |
| Knockout | Remove from architecture | Testing architectural necessity |

## Implementation

```python
def zero_ablate(model, inputs, component_name):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            return (output[0] * 0, *output[1:])
        return output * 0
    return run_with_hook(model, inputs, component_name, hook_fn)

def mean_ablate(model, inputs, component_name, mean_activation):
    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            return (mean_activation, *output[1:])
        return mean_activation
    return run_with_hook(model, inputs, component_name, hook_fn)
```

## Caution
Ablations may introduce out-of-distribution states. Always compare against controls.

## Falsification
Falsified if ablation has no effect (component is redundant) or if controls show similar effects.

## Exercises
- **Mathematical**: Show mean ablation minimizes expected squared error.
- **Implementation**: Implement resampling ablation.
- **Experimental**: Compare zero vs. mean ablation for all heads.
- **Research**: Does ablation order matter with redundancy?

## References

- Bau, D., et al. (2020). "Understanding the Role of Individual Units in a Deep Neural Network."

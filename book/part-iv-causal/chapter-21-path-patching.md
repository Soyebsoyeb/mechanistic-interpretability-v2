# Chapter 21 — Path Patching

## Motivation
Patching an entire component can hide interactions. Path patching isolates specific information pathways.

## Formalization
For path $C_i \rightarrow C_j$, patch only information flowing along that edge while preserving other functions.

## Implementation

```python
def path_patch(model, clean_inputs, corrupted_inputs, source, target, metric_fn):
    clean_cache = run_and_cache(model, clean_inputs)
    def path_hook(module, input, output):
        # Intercept only specific sub-tensor
        # Implementation depends on architecture
        pass
    # Complex: requires careful sub-tensor manipulation
    pass
```

## Measurement
Compare path-patching score against full-component patching.

## Falsification
Falsified if path patching does not recover behavior when full patching does.

## Exercises
- **Mathematical**: Formalize path patching in graph terms.
- **Implementation**: Implement path patching for attention heads.
- **Experimental**: Compare component vs. path patching for induction circuit.
- **Research**: Can path patching detect multi-path redundancy?

## References
- Wang, K., et al. (2022). "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small."

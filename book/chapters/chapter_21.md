# Chapter 21 — Path Patching

## Motivation

Patching an entire component can hide interactions. Path patching isolates specific information pathways, asking whether a particular edge in the computational graph is necessary.

## Formalization

For path $C_i \rightarrow C_j$, patch only the information flowing along that edge while preserving other functions of $C_i$ and $C_j$.

## Implementation

```python
def path_patch(model, clean_inputs, corrupted_inputs,
               source_component, target_component, metric_fn):
    clean_cache = run_and_cache(model, clean_inputs)

    def path_hook(module, input, output):
        # Complex: intercept specific sub-tensor
        # Implementation depends on architecture
        pass

    # Requires careful design per model
    pass
```

## Measurement
Compare path-patching score against full-component patching. If path score $\approx$ full score, the path carries the critical information.

## Falsification
Falsified if path patching does not recover behavior when full patching does.

## Exercises
- **Mathematical**: Formalize path patching in terms of graph edges.
- **Implementation**: Implement path patching for attention heads.
- **Experimental**: Compare component vs. path patching for induction circuit.
- **Research**: Can path patching detect multi-path redundancy?

## References

- Wang, K., et al. (2022). "Interpretability in the Wild: A Circuit for Indirect Object Identification."

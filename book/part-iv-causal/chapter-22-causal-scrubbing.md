# Chapter 22 — Causal Scrubbing

## Motivation
Causal scrubbing tests whether an abstraction is correct by intervening within equivalence classes.

## Formalization
Define abstraction $\alpha$ grouping states believed equivalent. Intervene within equivalence classes.

If behavior is consistent with $\alpha$, hypothesis gains evidence. Failures identify missing structure.

## Implementation

```python
def causal_scrubbing(model, inputs, abstraction_fn, metric_fn):
    clean_metric = metric_fn(model(inputs))
    def scrub_hook(module, input, output):
        eq_class = abstraction_fn(output)
        return random.choice(eq_class)
    # Run with scrubbing
    scrubbed_metric = metric_fn(model(inputs))
    return {
        "clean": clean_metric,
        "scrubbed": scrubbed_metric,
        "faithfulness": 1 - abs(clean_metric - scrubbed_metric) / abs(clean_metric)
    }
```

## Falsification
Falsified if scrubbed behavior differs significantly from clean.

## Exercises
- **Mathematical**: Define equivalence classes for induction head.
- **Implementation**: Implement causal scrubbing for simple circuit.
- **Experimental**: Test IOI circuit hypothesis with causal scrubbing.
- **Research**: Sensitivity to equivalence class definition?

## References
- Chan, L., et al. (2022). "Causal Scrubbing: A Method for Rigorously Testing Interpretability Hypotheses."

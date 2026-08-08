# Chapter 22 — Causal Scrubbing

## Motivation

Causal scrubbing tests whether an abstraction of model behavior is correct by intervening within equivalence classes. It provides a principled way to validate high-level algorithmic hypotheses.

## Formalization

Define abstraction $\alpha$ grouping states believed equivalent under the hypothesized algorithm. Intervene within equivalence classes.

If behavior is consistent with $\alpha$, the hypothesis gains evidence. Failures identify missing structure.

## Implementation

```python
def causal_scrubbing(model, inputs, abstraction_fn, metric_fn):
    clean_cache = run_and_cache(model, inputs)
    clean_metric = metric_fn(model, inputs)

    def scrub_hook(module, input, output):
        eq_class = abstraction_fn(output)
        return random.choice(eq_class)

    scrubbed_metric = metric_fn(model, inputs)

    return {
        "clean": clean_metric,
        "scrubbed": scrubbed_metric,
        "faithfulness": 1 - abs(clean_metric - scrubbed_metric) / abs(clean_metric)
    }
```

## Falsification
Falsified if scrubbed behavior differs significantly from clean behavior.

## Exercises
- **Mathematical**: Define equivalence classes for induction head algorithm.
- **Implementation**: Implement causal scrubbing for simple circuit.
- **Experimental**: Test IOI circuit hypothesis with causal scrubbing.
- **Research**: How sensitive is causal scrubbing to equivalence class definition?

## References

- Chan, L., et al. (2022). "Causal Scrubbing: A Method for Rigorously Testing Interpretability Hypotheses."

# Chapter 35 — Evaluation and Completeness

## Dimensions

| Dimension | Question | Metric |
|-----------|----------|--------|
| Faithfulness | Does explanation match model? | Behavioral similarity |
| Completeness | Does it explain all behavior? | Variance explained |
| Minimality | Is it as simple as possible? | Circuit size |
| Robustness | Stable under perturbation? | Cross-dataset performance |
| Generalization | Holds on unseen data? | Held-out accuracy |

**No single metric is sufficient.**

## Implementation

```python
def evaluate_explanation(model, circuit, test_inputs, metric_fn):
    faithfulness = circuit.evaluate_faithfulness(model, test_inputs, metric_fn)
    completeness = 1 - (metric_fn(model, test_inputs) - metric_fn(circuit, test_inputs))
    minimality = len(circuit.nodes) + len(circuit.edges)
    robustness = evaluate_on_perturbations(model, circuit, test_inputs)
    generalization = evaluate_on_distribution_shift(model, circuit)
    return {
        "faithfulness": faithfulness,
        "completeness": completeness,
        "minimality": minimality,
        "robustness": robustness,
        "generalization": generalization
    }
```

## Exercises
- **Mathematical**: Formalize trade-off between faithfulness and minimality.
- **Implementation**: Build multi-metric evaluation dashboard.
- **Research**: Can we learn a meta-evaluation function?

## References
- Huang, S., et al. (2023). "The Limitations of Interpreting Neural Networks via Attribution."

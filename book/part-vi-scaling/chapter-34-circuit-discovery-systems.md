# Chapter 34 — Circuit Discovery Systems

## Pipeline
```
Model -> Activation Collection -> Candidate Components -> Attribution 
-> Candidate Graph -> Intervention -> Circuit -> Validation
```

## Objectives
- Faithfulness
- Completeness
- Sparsity
- Robustness
- Scalability

## Implementation

```python
class CircuitDiscoverySystem:
    def __init__(self, model, attribution_method='gradient'):
        self.model = model
        self.attribution = attribution_method

    def discover(self, task_inputs, metric_fn, threshold=0.01):
        cache = self.collect_activations(task_inputs)
        scores = self.compute_attribution(task_inputs, metric_fn)
        candidates = self.select_candidates(scores, threshold)
        circuit = self.validate_with_intervention(candidates, task_inputs, metric_fn)
        return circuit
```

## Exercises
- **Implementation**: Build end-to-end circuit discovery pipeline.
- **Experimental**: Discover circuit for simple task automatically.
- **Research**: How does automation compare to human discovery?

## References
- Conmy, A., et al. (2023). "Towards Automated Circuit Discovery for Mechanistic Interpretability."

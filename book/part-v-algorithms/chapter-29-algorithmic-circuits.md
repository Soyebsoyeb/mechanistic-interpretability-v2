# Chapter 29 — Algorithmic Circuits

## Motivation
The goal is algorithm recovery. A valid explanation must predict behavior on unseen inputs.

## Candidate Algorithms
- Copying
- Retrieval
- Comparison
- Counting
- Modular arithmetic
- State tracking
- Pattern completion

## Validation Criteria
1. Predict behavior on held-out inputs
2. Be implementable as compact program
3. Match internal component functions
4. Be causally validated

## Implementation

```python
class AlgorithmicCircuit:
    def __init__(self, components, algorithm_fn):
        self.components = components
        self.algorithm = algorithm_fn

    def predict(self, inputs):
        return self.algorithm(inputs)

    def validate(self, model, test_inputs):
        model_outputs = model(test_inputs)
        circuit_outputs = self.predict(test_inputs)
        return (model_outputs == circuit_outputs).float().mean()
```

## Falsification
Falsified if circuit predictions diverge from model on held-out data.

## Exercises
- **Mathematical**: Formalize algorithm recovery as program synthesis.
- **Implementation**: Implement algorithmic circuit validator.
- **Experimental**: Recover algorithm from trained model on synthetic task.
- **Research**: Can we automatically synthesize algorithms from circuits?

## References
- Weiss, G., et al. (2021). "Thinking Like Transformers."

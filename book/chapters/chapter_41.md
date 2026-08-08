# Chapter 41 — Mechanistic Safety

## Applications
- Anomaly detection
- Representation monitoring
- Circuit-level auditing
- Feature steering
- Behavioral prediction
- Safety evaluation

## Caveat
Interpretability itself can fail. Safety arguments must include uncertainty and false-negative analysis.

## Implementation
```python
class MechanisticSafetyMonitor:
    def __init__(self, model, safety_features):
        self.model = model
        self.safety_features = safety_features

    def monitor(self, inputs):
        acts = extract_activations(self.model, inputs)
        for feature_name, direction in self.safety_features.items():
            activation = (acts @ direction).mean()
            if activation > threshold:
                return {"alert": True, "feature": feature_name}
        return {"alert": False}
```

## Exercises
- **Research**: Design mechanistic safety evaluation protocol.

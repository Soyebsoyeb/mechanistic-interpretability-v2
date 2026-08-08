# Chapter 37 — Latent Computation

## Motivation
Models may compute information internally without exposing it in outputs.

## Study Targets
- Intermediate representations
- Hidden state trajectories
- Recurrent computation
- Test-time computation
- State persistence
- Latent variables

## Implementation

```python
def analyze_latent_trajectory(model, inputs, n_steps=100):
    trajectories = []
    for step in range(n_steps):
        states = extract_all_layer_states(model, inputs)
        trajectories.append(states)
    # Analyze trajectory geometry
    pass
```

## Exercises
- **Research**: Where does latent computation occur in reasoning models?

## References
- Lieberum, T., et al. (2023). "Does Circuit Analysis Interpretability Scale?"

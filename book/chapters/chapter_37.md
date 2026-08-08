# Chapter 37 — Latent Computation

## Motivation
Models may compute information internally without exposing it in outputs. Studying latent computation is essential for understanding what models "know" but don't say.

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

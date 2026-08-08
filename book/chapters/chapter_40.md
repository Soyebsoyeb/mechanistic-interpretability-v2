# Chapter 40 — Deception and Conditional Behavior

## Motivation
Construct controlled models with conditional behavior to study deception mechanistically.

## Questions
- Where is the conditional policy represented?
- Which features trigger it?
- Can the trigger be causally modified?
- Does the behavior generalize?

**Avoid treating behavioral anomalies as proof of internal motives.**

## Implementation
```python
def analyze_conditional_behavior(model, trigger_inputs, normal_inputs):
    trigger_acts = extract_activations(model, trigger_inputs)
    normal_acts = extract_activations(model, normal_inputs)
    diff = trigger_acts.mean(dim=0) - normal_acts.mean(dim=0)
    discriminating = diff / diff.norm()
    steered = steer_feature(model, normal_inputs, layer_idx, discriminating, scale=1.0)
    pass
```

## Exercises
- **Experimental**: Build and analyze model with conditional deception.
- **Research**: Can we detect deceptive circuits before deployment?

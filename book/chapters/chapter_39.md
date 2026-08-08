# Chapter 39 — Goal Representations and Situational Awareness

## Motivation
Study whether models represent task objectives, environment state, evaluation context, and future consequences.

## Requirements
- Controlled interventions
- Careful distinction between correlation and causation

## Implementation
```python
def test_goal_representation(model, task_variants):
    for variant in task_variants:
        acts = extract_activations(model, variant.inputs)
        # Check if goal representation is invariant
        pass
```

## Exercises
- **Research**: Can we detect situational awareness mechanistically?

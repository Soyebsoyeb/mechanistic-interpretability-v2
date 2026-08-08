# Chapter 18 — Correlation Is Not Causation

## Motivation

If $C(x)$ correlates with behavior $B(x)$, this does not prove $C \rightarrow B$. A component may be causal, downstream, upstream, redundant, correlated, or epiphenomenal. Intervention is therefore central to mechanistic interpretability. This chapter establishes why correlation-based methods are insufficient and how to design proper causal tests.

## Possible Relationships

| Relationship | Description | Test |
|-------------|-------------|------|
| Causal | $C$ directly causes $B$ | Ablate $C$, $B$ changes |
| Downstream | $B$ causes $C$ | Ablate $B$, $C$ changes |
| Upstream | $C$ is necessary but not sufficient | Ablate $C$, $B$ changes; but $C$ alone insufficient |
| Redundant | Multiple components implement $B$ | Ablate $C$, $B$ unchanged (compensation) |
| Correlated | Confounder $Z$ drives both | Control for $Z$, correlation vanishes |
| Epiphenomenal | Coincidental correlation | No causal effect in either direction |

## The Fundamental Problem

In general: $P(Y|X=x) \neq P(Y|do(X=x))$

Observational data can only give us $P(Y|X=x)$. Causal inference requires interventions, which give us $P(Y|do(X=x))$.

## Implementation

```python
def test_causal_relationship(model, component, behavior_fn, inputs):
    baseline = behavior_fn(model, inputs)

    # Test 1: Ablate component
    ablated = ablate_component(model, inputs, component)
    ablated_metric = behavior_fn(model, ablated)

    # Test 2: Patch component from clean to corrupted
    clean_inputs, corrupted_inputs = generate_minimal_pair(inputs)
    patched = activation_patch(model, clean_inputs, corrupted_inputs, component)
    patched_metric = behavior_fn(model, patched)

    return {
        "is_causal": abs(baseline - ablated_metric) > 0.05,
        "is_necessary": ablated_metric < baseline * 0.9,
        "is_sufficient": patched_metric > baseline * 0.9
    }
```

## Falsification
Any causal claim is falsified if the intervention does not produce the predicted effect, or if a confounder explains the correlation.

## Exercises
- **Mathematical**: Formalize difference between $P(Y|X=x)$ and $P(Y|do(X=x))$.
- **Implementation**: Build causal testing framework.
- **Experimental**: Find correlated but non-causal component.
- **Research**: How often do high-attribution components fail causal tests?

## References

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*.

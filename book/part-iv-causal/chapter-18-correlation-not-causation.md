# Chapter 18 — Correlation Is Not Causation

## Motivation
If $C(x)$ correlates with behavior $B(x)$, this does not prove $C \rightarrow B$. Intervention is the only path to causal knowledge.

## Possible Relationships

| Relationship | Description | Test |
|-------------|-------------|------|
| Causal | $C$ directly causes $B$ | Ablate $C$, $B$ changes |
| Downstream | $B$ causes $C$ | Ablate $B$, $C$ changes |
| Upstream | $C$ necessary but not sufficient | Ablate $C$, $B$ changes; $C$ alone insufficient |
| Redundant | Multiple components implement $B$ | Ablate $C$, $B$ unchanged |
| Correlated | Confounder $Z$ drives both | Control for $Z$, correlation vanishes |
| Epiphenomenal | Coincidental correlation | No causal effect either direction |

## Implementation

```python
def test_causal_relationship(model, component, behavior_fn, inputs):
    baseline = behavior_fn(model, inputs)
    ablated = ablate_component(model, inputs, component)
    ablated_metric = behavior_fn(model, ablated)
    return {
        "is_causal": abs(baseline - ablated_metric) > 0.05,
        "is_necessary": ablated_metric < baseline * 0.9
    }
```

## Falsification
Falsified if intervention does not produce predicted effect, or if confounder explains correlation.

## Exercises
- **Mathematical**: Formalize difference between $P(Y|X=x)$ and $P(Y|do(X=x))$.
- **Implementation**: Build causal testing framework.
- **Experimental**: Find correlated but non-causal component.
- **Research**: How often do high-attribution components fail causal tests?

## References
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*.

# Chapter 17 — Feature Interpretation and Validation

## Motivation
A feature description is a hypothesis requiring rigorous validation.

## Validation Protocol
1. **Positive examples**: High activation, semantically coherent
2. **Negative examples**: Near-zero where feature absent
3. **Contrastive examples**: Minimal pairs differing in one property
4. **Causal tests**: Suppress/amplify feature, measure behavior

## Implementation

```python
def causal_feature_test(model, inputs, feature_direction, behavior_fn):
    baseline = behavior_fn(model, inputs)
    suppressed = intervene_on_feature(model, inputs, feature_direction, scale=-2.0)
    amplified = intervene_on_feature(model, inputs, feature_direction, scale=+2.0)
    return {
        "baseline": baseline,
        "suppress_effect": baseline - behavior_fn(model, suppressed),
        "amplify_effect": behavior_fn(model, amplified) - baseline,
        "is_causal": abs(baseline - behavior_fn(model, suppressed)) > 0.1
    }
```

## Scoring
| Criterion | Weight |
|-----------|--------|
| Activation coherence | 0.25 |
| Contrastive discrimination | 0.25 |
| Causal effect | 0.30 |
| Generalization | 0.20 |

## Falsification
Falsified if negative examples activate strongly, or intervention has no effect.

## Exercises
- **Mathematical**: Formalize contrastive test as statistical hypothesis test.
- **Implementation**: Build automated validation pipeline.
- **Experimental**: Validate 10 SAE features; report quality scores.
- **Research**: Do validated features generalize across tasks?

## References
- Bills, S., et al. (2023). "Language Models Can Explain Neurons in Language Models."

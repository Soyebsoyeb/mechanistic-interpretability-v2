# Chapter 17 — Feature Interpretation and Validation

## Motivation

A feature description is a hypothesis, not a fact. Rigorous validation requires multiple independent lines of evidence. This chapter establishes the gold standard for feature validation in mechanistic interpretability.

## Validation Protocol

### 1. Positive Examples
Find contexts where the feature activates strongly. These should be semantically coherent and diverse.

### 2. Negative Examples
Find contexts where the feature should not activate. These should have near-zero activation and be structurally similar to positive examples.

### 3. Contrastive Examples
Construct minimal pairs differing in exactly one semantic or syntactic property:
- "The cat sat" vs. "The cats sat" (number)
- "He went to the store" vs. "She went to the store" (gender)
- "The big red ball" vs. "The small red ball" (size)

### 4. Causal Tests
Suppress or amplify the feature and measure behavior change:

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

## Scoring Feature Quality

| Criterion | Weight | Test |
|-----------|--------|------|
| Activation coherence | 0.25 | Top examples share property |
| Contrastive discrimination | 0.25 | Minimal pairs show activation difference |
| Causal effect | 0.30 | Intervention changes behavior |
| Generalization | 0.20 | Holds on held-out data |

## Falsification
A feature interpretation is falsified if:
- Negative examples activate as strongly as positive examples
- Contrastive pairs show no activation difference
- Causal intervention does not change behavior
- The feature fails on a held-out test set

## Exercises
- **Mathematical**: Formalize contrastive test as statistical hypothesis test.
- **Implementation**: Build automated validation pipeline.
- **Experimental**: Validate 10 SAE features; report quality scores.
- **Research**: Do validated features generalize across tasks?

## References

- Bills, S., et al. (2023). "Language Models Can Explain Neurons in Language Models."

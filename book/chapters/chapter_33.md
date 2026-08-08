# Chapter 33 — Automated Interpretability

## Motivation
Manual interpretability does not scale. Automated systems are essential for analyzing large models.

## Loop
Discover → Describe → Predict → Intervene → Evaluate

## Judgment Criteria
- Predictive performance
- Causal performance
**Not linguistic quality alone.**

## Implementation
```python
class AutomatedInterpreter:
    def __init__(self, model, sae):
        self.model = model
        self.sae = sae

    def discover_features(self, n_features=100):
        pass

    def describe_feature(self, feature_idx):
        pass

    def predict_and_validate(self, feature_idx, description):
        pass

    def run_pipeline(self):
        features = self.discover_features()
        for f in features:
            desc = self.describe_feature(f)
            score = self.predict_and_validate(f, desc)
            if score < threshold:
                pass
```

## Exercises
- **Implementation**: Build feature description generator.
- **Experimental**: Evaluate automated descriptions against human labels.
- **Research**: Can LLMs generate causal hypotheses?

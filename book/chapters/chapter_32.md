# Chapter 32 — Cross-Model Feature Alignment

## Motivation
Compare features across different models to identify universal or model-specific computations.

## Metrics
- Cosine similarity: $\cos(v_A, v_B)$
- Activating context overlap
- Intervention effect similarity
- Downstream consumer alignment
- Feature selectivity comparison

**Geometry alone is insufficient.**

## Implementation
```python
def align_features(model_a, model_b, dataset, layer_a, layer_b):
    acts_a = extract_activations(model_a, dataset, layer_a)
    acts_b = extract_activations(model_b, dataset, layer_b)
    sae_a = train_sae(acts_a, acts_a.shape[-1], acts_a.shape[-1] * 8)
    sae_b = train_sae(acts_b, acts_b.shape[-1], acts_b.shape[-1] * 8)
    alignment_matrix = sae_a.W_d @ sae_b.W_d.T
    best_matches = alignment_matrix.argmax(dim=1)
    match_scores = alignment_matrix.max(dim=1).values
    return best_matches, match_scores
```

## Exercises
- **Mathematical**: Formalize feature alignment as assignment problem.
- **Experimental**: Align features between GPT-2 and Pythia.
- **Research**: Do aligned features have similar causal effects?

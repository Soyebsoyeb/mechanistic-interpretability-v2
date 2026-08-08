# Chapter 13 — Polysemanticity

## Motivation

A neuron that activates for apparently different contexts is polysemantic. This is one of the central puzzles of interpretability: why would a single neuron represent multiple unrelated concepts? Understanding polysemanticity is essential for correctly interpreting neural network internals.

## Formalization

Neuron $i$ is polysemantic if there exist contexts $c_1, c_2$ such that:
- $\mathbb{E}[h_i | c_1] \gg 0$
- $\mathbb{E}[h_i | c_2] \gg 0$
- $c_1$ and $c_2$ are semantically unrelated by human judgment

### Potential Explanations

1. **Superposition**: $n > d$ features in $d$ dimensions force overlap (Chapter 14)
2. **Feature overlap**: The "unrelated" features share a statistical regularity
3. **Nonlinear interactions**: The neuron computes a nonlinear function of multiple features
4. **Correlated training data**: Spurious correlations in training create apparent polysemanticity
5. **Insufficient experimental context**: The neuron may respond to an abstract feature we have not identified
6. **Hierarchical features**: The neuron may represent a superordinate category

## Response Protocol

The correct response to apparent polysemanticity is **not** to assign a label immediately. Instead:

1. Collect activations across diverse contexts
2. Find positive examples (high activation)
3. Find negative examples (low activation on similar contexts)
4. Test held-out contexts
5. Intervene on the neuron
6. Measure causal effects
7. Attempt SAE decomposition

## Implementation

```python
def analyze_polysemanticity(model, dataset, layer_idx, neuron_idx, n_examples=100):
    acts = extract_neuron_activations(model, dataset, layer_idx, neuron_idx)
    top_values, top_indices = torch.topk(acts, n_examples)

    # Cluster top examples by embedding similarity
    embeddings = [model.embed(dataset[i]).mean(dim=1) for i in top_indices]
    embeddings = torch.stack(embeddings)
    sim_matrix = embeddings @ embeddings.T

    threshold = 0.5
    clusters = 0
    visited = set()
    for i in range(n_examples):
        if i in visited: continue
        clusters += 1
        for j in range(i+1, n_examples):
            if sim_matrix[i,j] > threshold: visited.add(j)
        visited.add(i)

    baseline = evaluate_model(model, dataset)
    ablated = evaluate_with_ablation(model, dataset, layer_idx, neuron_idx)

    return {"n_clusters": clusters, "causal_effect": baseline - ablated}
```

## Falsification
A polysemanticity claim is falsified if:
- SAE decomposes neuron into single abstract feature
- The "different" contexts share a hidden statistical property
- The neuron is actually a superposition of features separable by change of basis

## Exercises
- **Mathematical**: Construct toy example where neuron must be polysemantic ($n > d$).
- **Implementation**: Build clustering-based polysemanticity detector.
- **Experimental**: Decompose polysemantic neuron with SAE; report monosemanticity.
- **Research**: Relationship between data diversity and polysemanticity.

## References

- Elhage, N., et al. (2022). "Superposition, Memorization, and Double Descent."

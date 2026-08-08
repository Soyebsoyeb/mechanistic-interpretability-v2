# Chapter 13 — Polysemanticity

## Motivation
A neuron activating for apparently different contexts is polysemantic — a central puzzle.

## Formalization
Neuron $i$ is polysemantic if $\mathbb{E}[h_i | c_1] \gg 0$ and $\mathbb{E}[h_i | c_2] \gg 0$ for semantically unrelated $c_1, c_2$.

## Explanations
1. Superposition ($n > d$)
2. Feature overlap (hidden statistical regularity)
3. Nonlinear interactions
4. Correlated training data
5. Insufficient experimental context

## Response Protocol
1. Collect activations
2. Find positive examples
3. Find negative examples
4. Test held-out contexts
5. Intervene
6. Measure causal effects

## Implementation

```python
def analyze_polysemanticity(model, dataset, layer_idx, neuron_idx, n_examples=100):
    acts = extract_neuron_activations(model, dataset, layer_idx, neuron_idx)
    top_values, top_indices = torch.topk(acts, n_examples)
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
    return {"n_clusters": clusters}
```

## Falsification
Falsified if SAE decomposes neuron into single abstract feature.

## Exercises
- **Mathematical**: Construct toy example where neuron must be polysemantic.
- **Implementation**: Build clustering-based polysemanticity detector.
- **Experimental**: Decompose polysemantic neuron with SAE.
- **Research**: Relationship between data diversity and polysemanticity.

## References
- Elhage, N., et al. (2022). "Superposition, Memorization, and Double Descent."

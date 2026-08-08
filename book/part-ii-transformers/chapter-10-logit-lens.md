# Chapter 10 — Logit Lens and Intermediate Predictions

## Motivation
The logit lens projects intermediate states onto vocabulary space, revealing what information is present at each layer.

## Formalization
Let $W_U \in \mathbb{R}^{d \times |\mathcal{V}|}$ be unembedding matrix.
$$z_\ell = x_\ell W_U, \quad p_\ell = \text{softmax}(z_\ell)$$

**Caution**: Intermediate prediction is a measurement tool, not proof of reasoning.

## Implementation

```python
def logit_lens(model, inputs, layer_idx):
    cache = {}
    def hook_fn(module, input, output):
        cache["residual"] = output.detach()
    target = model.blocks[layer_idx]
    handle = target.register_forward_hook(hook_fn)
    with torch.no_grad(): _ = model(inputs)
    handle.remove()
    x = cache["residual"]
    logits = x @ model.W_U
    return logits, torch.softmax(logits, dim=-1)

def track_prediction_evolution(model, inputs, target_token_id):
    probs = []
    for layer_idx in range(model.n_layers):
        _, p = logit_lens(model, inputs, layer_idx)
        probs.append(p[:, -1, target_token_id].mean().item())
    return probs
```

## Measurement
- Logit trajectory
- Rank evolution
- Probability mass
- Entropy evolution

## Intervention
Patch intermediate states, observe logit changes to reveal when information becomes available.

## Falsification
Falsified if intermediate prediction does not correlate with final, or if random directions produce comparable signals.

## Exercises
- **Mathematical**: Show logit lens is orthogonal projection if $W_U$ has orthogonal columns.
- **Implementation**: Implement steering intervention.
- **Experimental**: Plot correct-token probability across layers.
- **Research**: Can logit lens detect hallucinations?

## References
- nostalgebraist (2020). "Interpreting GPT: The Logit Lens."

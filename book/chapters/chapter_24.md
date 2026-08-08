# Chapter 24 — Induction Heads

## Motivation

Induction heads copy information from earlier in a sequence. They are a canonical example of a transformer circuit and were among the first circuits to be fully reverse-engineered. Understanding induction heads provides a template for reverse engineering more complex circuits.

## Formalization

Given sequence $[A][B]...[A]$, predict $[B]$. The induction head attends from the second $[A]$ to $[B]$ and copies its information.

## Procedure
1. Construct controlled prompts: $[A][B]...[A]$
2. Identify heads with suitable attention patterns
3. Measure induction scores
4. Ablate candidates
5. Patch candidates
6. Inspect QK and OV structure
7. Test new token identities

## Implementation

```python
def measure_induction_score(model, prompts):
    scores = []
    for prompt in prompts:
        with torch.no_grad():
            logits = model(prompt)
        last_a_pos = (prompt == prompt[0]).nonzero()[-1].item()
        b_token = prompt[1].item()
        prob = torch.softmax(logits[0, last_a_pos], dim=-1)[b_token]
        scores.append(prob.item())
    return sum(scores) / len(scores)

def find_induction_heads(model, n_heads, prompts):
    scores = torch.zeros(model.n_layers, n_heads)
    for layer in range(model.n_layers):
        for head in range(n_heads):
            ablated = ablate_head(model, layer, head, prompts)
            scores[layer, head] = measure_induction_score(model, ablated)
    return scores
```

## Falsification
Falsified if ablating the head does not reduce induction score, or if QK does not show [A]→[B] pattern.

## Exercises
- **Mathematical**: Write QK matrix structure for ideal induction head.
- **Implementation**: Implement full induction circuit discovery pipeline.
- **Experimental**: Discover and validate induction heads in GPT-2 small.
- **Research**: Do induction heads generalize to unseen token pairs?

## References

- Olsson, C., et al. (2022). "In-context Learning and Induction Heads."

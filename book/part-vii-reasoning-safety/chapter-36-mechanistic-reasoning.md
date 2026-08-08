# Chapter 36 — Mechanistic Interpretability of Reasoning

## Motivation
Reasoning models introduce new questions about latent computation.

## Key Question
A visible reasoning trace may be:
- Causally relevant
- Partially relevant
- Compressed
- Post hoc
- One component of larger latent computation

**Reasoning trace $\neq$ guaranteed complete mechanism.**

## Implementation

```python
def analyze_reasoning_trace(model, reasoning_prompts):
    for prompt in reasoning_prompts:
        hidden_states = extract_hidden_states(model, prompt)
        text_trace = model.generate(prompt)
        # Check if hidden states predict reasoning steps before text
        pass
```

## Exercises
- **Experimental**: Intervene on hidden states during reasoning.
- **Research**: Do reasoning traces expose full computation?

## References
- Nye, M., et al. (2021). "Show Your Work: Scratchpads for Intermediate Computation with Language Models."

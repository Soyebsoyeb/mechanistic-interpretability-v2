# Chapter 36 — Mechanistic Interpretability of Reasoning

## Motivation

Reasoning models introduce new questions about latent computation. A visible reasoning trace may be causally relevant, partially relevant, compressed, post hoc, or one component of a larger latent computation.

**Reasoning trace ≠ guaranteed complete mechanism.**

## Key Questions
- Does the model compute reasoning steps internally before generating text?
- Are reasoning traces faithful to internal computation?
- Can we intervene on reasoning without changing the text trace?

## Implementation
```python
def analyze_reasoning_trace(model, reasoning_prompts):
    for prompt in reasoning_prompts:
        hidden_states = extract_hidden_states(model, prompt)
        text_trace = model.generate(prompt)
        # Compare hidden states with textual reasoning
        pass
```

## Exercises
- **Experimental**: Intervene on hidden states during reasoning.
- **Research**: Do reasoning traces expose full computation?

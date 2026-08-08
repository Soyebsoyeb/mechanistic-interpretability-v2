# Chapter 52 — Analyze a Reasoning Model

## Approach
Choose controlled reasoning task. Measure:
- Hidden states
- Attention patterns
- Intermediate logits
- Feature activations
- Causal effects

**Do not assume textual reasoning traces expose the full mechanism.**

## Template
```python
def project_reasoning_model():
    model = load_reasoning_model()
    tasks = load_reasoning_tasks()
    for task in tasks:
        traces = extract_reasoning_traces(model, task)
        hidden_analysis = analyze_hidden_during_reasoning(model, task)
        causal = test_reasoning_causality(model, task)
        generate_reasoning_report(traces, hidden_analysis, causal)
```

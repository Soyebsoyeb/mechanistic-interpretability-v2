# Chapter 49 — Discover an Induction Circuit

## Deliverable
Validated circuit diagram and reproducible code.

## Steps
1. Train or obtain model with copying behavior
2. Measure induction score
3. Analyze attention patterns
4. Inspect QK and OV matrices
5. Ablate candidates
6. Patch candidates

## Template
```python
def project_induction_circuit():
    model = load_model("gpt2")
    prompts = generate_induction_prompts(n=1000)
    scores = torch.zeros(model.n_layers, model.n_heads)
    for l in range(model.n_layers):
        for h in range(model.n_heads):
            scores[l, h] = measure_induction_score(model, prompts, l, h)
    top_heads = scores.argmax(dim=1)
    for l, h in enumerate(top_heads):
        validate_induction_head(model, l, h, prompts)
    generate_circuit_diagram(scores, validations)
```

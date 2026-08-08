# Chapter 50 — Train and Interpret an SAE

## Deliverable
Feature catalogue with evidence.

## Steps
1. Collect transformer activations
2. Train SAE
3. Measure reconstruction and sparsity
4. For selected features: display top examples
5. Generate hypotheses
6. Test held-out examples
7. Perform feature ablations

## Template
```python
def project_train_sae():
    acts = extract_activations(model, dataset, layer=5)
    sae = train_sae(acts, d_hidden=acts.shape[-1] * 8)
    catalogue = []
    for i in range(sae.d_hidden):
        top_examples = find_top_activating_examples(sae, acts, i)
        hypothesis = generate_hypothesis(top_examples)
        validation = validate_feature(sae, acts, i, hypothesis)
        catalogue.append({
            "feature_id": i,
            "hypothesis": hypothesis,
            "validation": validation,
            "top_examples": top_examples
        })
    generate_feature_catalogue(catalogue)
```

## Exercises
- **Research**: What is the optimal expansion factor for SAEs?

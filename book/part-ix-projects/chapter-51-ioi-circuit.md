# Chapter 51 — Reconstruct an IOI Circuit

## Steps
1. Behavioral baseline
2. Candidate discovery
3. Attribution
4. Component patching
5. Edge/path analysis
6. Circuit pruning
7. Sufficiency testing
8. Generalization testing

## Deliverable
Compact computational graph explaining the task.

## Template
```python
def project_ioi_circuit():
    model = load_model("pythia-160m")
    templates = load_ioi_templates()
    names = load_name_pairs()
    baseline = measure_ioi_accuracy(model, templates, names)
    candidates = discover_ioi_candidates(model, templates, names)
    circuit = validate_candidates(model, candidates, templates, names)
    minimal_circuit = prune_circuit(circuit, model, templates, names)
    generalization = test_on_new_templates(model, minimal_circuit)
    return minimal_circuit, generalization
```

## Exercises
- **Research**: How does IOI circuit vary across model scales?

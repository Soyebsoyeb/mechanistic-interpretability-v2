# Chapter 48 — Reverse Engineer a Tiny Transformer

## Requirements
1. Known ground-truth algorithm
2. Small number of layers
3. Inspect every parameter
4. Identify candidate features
5. Identify candidate circuits
6. Intervene
7. Reconstruct algorithm

## Deliverable
Report containing proposed mechanism and all experiments.

## Suggested Task
Train 2-layer transformer on $(a + b) \mod 97$ or token copying.

## Template
```python
def project_tiny_transformer():
    model = train_tiny_transformer(task="copying", n_layers=2)
    for name, param in model.named_parameters():
        print(f"{name}: {param.shape}")
    features = analyze_all_neurons(model)
    circuit = discover_circuit(model, task="copying")
    faithfulness = circuit.evaluate_faithfulness(model)
    generate_report(model, features, circuit, faithfulness)
```

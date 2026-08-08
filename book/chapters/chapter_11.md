# Chapter 11 — Transformer Circuits

## Motivation

A circuit is a computational subgraph responsible for a behavior. Identifying circuits is the central goal of mechanistic interpretability. The challenge is not merely finding correlations, but isolating the minimal set of components that causally implement a target computation.

## Formalization

A candidate circuit is $G = (V, E)$ where:
- $V$ are components (attention heads, MLP neurons, residual stream positions)
- $E$ represent information flow (attention edges, MLP feedforward edges, residual connections)

The goal is to find a minimal $G$ that preserves behavior $B(x)$ for relevant inputs $x$.

### Circuit Properties

| Property | Definition | Measurement |
|----------|-----------|-------------|
| Faithfulness | $B_{\text{circuit}}(x) \approx B_{\text{full}}(x)$ | Relative error on test set |
| Completeness | No missing components | Behavior degradation when adding pruned components |
| Minimality | No redundant components | Behavior unchanged when removing any component |
| Robustness | Stable under perturbation | Variance across input subsets |
| Generalization | Holds on unseen data | Test on held-out distribution |

## Implementation

```python
@dataclass
class CircuitNode:
    layer: int
    component_type: str
    index: int
    position: int

class Circuit:
    def __init__(self, nodes, edges, target):
        self.nodes = nodes
        self.edges = edges
        self.target = target

    def evaluate_faithfulness(self, model, test_inputs, metric_fn):
        # Run with only circuit nodes active
        pass

def prune_circuit(circuit, model, test_inputs, metric_fn, threshold=0.01):
    essential = []
    for edge in circuit.edges:
        temp = Circuit(circuit.nodes, [e for e in circuit.edges if e != edge], circuit.target)
        if temp.evaluate_faithfulness(model, test_inputs, metric_fn) < threshold:
            essential.append(edge)
    return Circuit(circuit.nodes, essential, circuit.target)
```

## Measurement
1. Faithfulness score: $1 - \frac{|B_{\text{circuit}} - B_{\text{full}}|}{|B_{\text{full}}|}$
2. Circuit size: $|V| + |E|$ normalized by full model size
3. Edge attribution: Contribution of each edge to target behavior

## Intervention: Path Patching

Path patching (detailed in Chapter 21) is the primary tool for validating circuit edges. It isolates whether information flowing along a specific path is necessary.

## Falsification

A circuit is falsified if:
- Pruning any edge or node does not change behavior (circuit is not minimal)
- The circuit fails on held-out examples (poor generalization)
- A smaller circuit achieves equal faithfulness (circuit is not minimal)
- The circuit does not reproduce behavior when composed in isolation

## Exercises

### Mathematical
1. Prove that for any behavior $B$, there exists a trivially faithful circuit: the full model itself. What makes a circuit interesting?

### Implementation
2. Implement a circuit visualization function that plots the circuit graph using networkx.

### Experimental
3. For a 2-layer transformer on a synthetic task, manually identify a candidate circuit and measure its faithfulness.

### Research
4. Investigate whether circuit faithfulness on a task correlates with the circuit's ability to generalize to structurally similar tasks.

## References

- Elhage, N., et al. (2021). "A Mathematical Framework for Transformer Circuits."
- Wang, K., et al. (2022). "Interpretability in the Wild: A Circuit for Indirect Object Identification."

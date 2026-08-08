# Chapter 11 — Transformer Circuits

## Motivation
A circuit is a computational subgraph responsible for a behavior. Identifying circuits is the central goal.

## Formalization
Circuit $G = (V, E)$ where $V$ are components, $E$ are information flows.

| Property | Definition |
|----------|-----------|
| Faithfulness | $B_{\text{circuit}} \approx B_{\text{full}}$ |
| Completeness | No missing components |
| Minimality | No redundant components |
| Robustness | Stable under perturbation |
| Generalization | Holds on unseen data |

## Implementation

```python
from dataclasses import dataclass
from typing import List

@dataclass
class CircuitNode:
    layer: int
    component_type: str
    index: int
    position: int

@dataclass
class CircuitEdge:
    source: CircuitNode
    target: CircuitNode
    weight: float

class Circuit:
    def __init__(self, nodes, edges, target):
        self.nodes = nodes
        self.edges = edges
        self.target = target

    def to_dict(self):
        return {
            "nodes": [{"layer": n.layer, "type": n.component_type, "index": n.index} for n in self.nodes],
            "edges": [{"src": e.source.layer, "dst": e.target.layer, "weight": e.weight} for e in self.edges],
            "target": self.target
        }

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

## Falsification
Falsified if pruning any edge does not change behavior, or if circuit fails on held-out data.

## Exercises
- **Mathematical**: Prove full model is trivially faithful circuit.
- **Implementation**: Visualize circuit graph with networkx.
- **Experimental**: Identify candidate circuit in 2-layer transformer.
- **Research**: Does faithfulness correlate with generalization?

## References
- Elhage, N., et al. (2021). "A Mathematical Framework for Transformer Circuits."
- Wang, K., et al. (2022). "Interpretability in the Wild."

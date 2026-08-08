import torch
from dataclasses import dataclass
from typing import List

@dataclass
class CircuitNode:
    layer: int
    component_type: str
    index: int

@dataclass
class CircuitEdge:
    source: CircuitNode
    target: CircuitNode
    weight: float

class Circuit:
    def __init__(self, nodes: List[CircuitNode], edges: List[CircuitEdge], target: str):
        self.nodes = nodes
        self.edges = edges
        self.target = target

    def to_dict(self):
        return {
            "nodes": [{"layer": n.layer, "type": n.component_type, "index": n.index} for n in self.nodes],
            "edges": [{"src": e.source.layer, "dst": e.target.layer, "weight": e.weight} for e in self.edges],
            "target": self.target
        }

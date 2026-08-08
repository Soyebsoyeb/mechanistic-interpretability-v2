# Chapter 46 — Circuit Discovery Pipeline

## Stages
```
extract.py -> cache.py -> attribute.py -> candidate_graph.py
-> patch.py -> validate.py -> visualize.py
```

## Export Format
```json
{
  "nodes": [],
  "edges": [],
  "target": "",
  "metric": "",
  "evidence": []
}
```

## Implementation
```python
class CircuitDiscoveryPipeline:
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.stages = ['extract', 'cache', 'attribute', 'graph', 'patch', 'validate']

    def run(self, task_inputs, metric_fn):
        results = {}
        for stage in self.stages:
            results[stage] = getattr(self, f"run_{stage}")(task_inputs, metric_fn)
        return results
```

## Exercises
- **Implementation**: Build full pipeline with tests for each stage.

## References
- Conmy et al. (2023). "Towards Automated Circuit Discovery."

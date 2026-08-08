# Chapter 47 — Scaling Experiments

## Best Practices
- Profile GPU memory
- Stream activations
- Batch carefully
- Save intermediate artifacts
- Checkpoint analyses
- Deterministic configurations
- Avoid storing unnecessary tensors
- Record total computational cost

## Implementation
```python
class ScalingExperiment:
    def __init__(self, model_name, config):
        self.model_name = model_name
        self.config = config
        self.cost_log = []

    def profile_memory(self, fn):
        torch.cuda.reset_peak_memory_stats()
        result = fn()
        peak = torch.cuda.max_memory_allocated() / 1e9
        self.cost_log.append({"peak_memory_gb": peak})
        return result

    def stream_activations(self, dataset, batch_size):
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i+batch_size]
            acts = self.extract(batch)
            yield acts
            del acts
            torch.cuda.empty_cache()
```

## Exercises
- **Implementation**: Build profiling wrapper for all experiments.

## References
- Lieberum et al. (2023). "Does Circuit Analysis Interpretability Scale?"

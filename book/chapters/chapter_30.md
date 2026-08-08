# Chapter 30 — Scaling Feature Discovery

## Motivation

Larger models produce more layers, activations, features, and interactions. Interpretability becomes a systems problem requiring careful engineering, not just scientific insight.

## Challenges
- Computational cost: activation extraction, SAE training, analysis
- Memory: storing activations for billion-parameter models
- Human analysis bottleneck: cannot manually inspect millions of features
- Interaction complexity: combinatorial feature interactions

## Cost Model
$$\text{Cost} = \text{Compute}_{\text{extract}} + \text{Compute}_{\text{train}} + \text{Compute}_{\text{analyze}} + \text{Human}_{\text{review}}$$

## Implementation
```python
def estimate_interpretability_cost(model_size, n_layers, d_model, 
                                   dataset_size, sae_expansion=8):
    activation_memory = dataset_size * n_layers * d_model * 4 / 1e9
    sae_params = d_model * d_model * sae_expansion * 2
    sae_training_flops = sae_params * dataset_size * 10
    return {
        "activation_memory_gb": activation_memory,
        "sae_training_flops": sae_training_flops,
        "estimated_hours": sae_training_flops / (1e15 * 8)
    }
```

## Exercises
- **Mathematical**: Derive scaling laws for interpretability cost vs. model size.
- **Implementation**: Build streaming activation extractor.
- **Experimental**: Profile memory usage for GPT-2 activation extraction.
- **Research**: Can we prioritize which features to analyze?

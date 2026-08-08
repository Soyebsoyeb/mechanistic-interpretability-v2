# Chapter 1 — What Is Mechanistic Interpretability?

## Motivation

A neural network implements a function $f_{\theta}(x) = y$. Knowing the network's input-output behavior does not uniquely tell us how it computes $y$. Many different internal mechanisms can produce similar outputs. This is the fundamental problem of the black box: observational equivalence.

Mechanistic interpretability asks: **What internal computation gives rise to the observed behavior?**

The goal is not simply to describe what the model does. The goal is to recover a computational description of *how* it does it.

## The Black Box Problem

Consider three different programs that all implement $f(x) = x^2$:

1. `return x * x`
2. `return exp(2 * log(x))`
3. `return sum([x for _ in range(x)])` (for integer $x$)

All produce identical outputs, yet internal computations differ radically. Neural networks face the same problem at massive scale.

## Learning Objectives

By the end of this chapter, you will:
- Distinguish behavioral, statistical, and mechanistic analysis
- Identify six levels of explanation
- Apply seven scientific criteria to evaluate hypotheses
- Design interventions to test mechanistic claims

## Levels of Explanation

| Level | Description | Example | Strength |
|-------|-------------|---------|----------|
| **Behavioral** | Input-output mapping | "The model predicts positive sentiment" | Weak |
| **Statistical** | Correlation patterns | "Output correlates with sentiment words" | Weak |
| **Feature** | Identified representation | "Direction in layer 5 encodes sentiment" | Moderate |
| **Component** | Specific unit | "Neuron 847 activates on negatives" | Moderate |
| **Circuit** | Subgraph of components | "Heads 3.1 and 4.2 form sentiment circuit" | Strong |
| **Algorithmic** | Full procedure | "Negation detection -> polarity accumulation -> threshold" | Strongest |

## Scientific Criteria for Mechanistic Hypotheses

For hypothesis $H$ about mechanism $M$ producing behavior $B$:

1. **Localization**: Where is $M$? Specify layer, component, position, tensor index.
2. **Specification**: What exactly does $M$ compute? Provide mathematical function.
3. **Causal Relevance**: Does modifying $M$ change $B$? Require intervention evidence.
4. **Sufficiency**: Does $M$ reproduce $B$ without unexplained magic?
5. **Generalization**: Does $M$ work on held-out examples?
6. **Falsifiability**: What result $R$ would prove $H$ wrong?
7. **Independence**: Can the mechanism be validated by multiple methods?

## The Eight Questions Framework

| Question | Example |
|----------|---------|
| What is being claimed? | "Head 2 in layer 3 copies from position $i$ to $j$" |
| What mathematical object? | OV matrix $W_V^{(3,2)} W_O^{(3,2)} \in \mathbb{R}^{d \times d}$ |
| Where in the model? | Layer 3, head 2, post-attention residual stream |
| How measured? | Attention weight $A_{j,i}$ and output projection |
| How intervened? | Zero-ablate head output; patch OV circuit |
| What falsifies? | Ablating head does not change copying behavior |
| Reproducible? | Fixed seed 42, checkpoint gpt2-v1.2.3 |
| Alternatives? | Redundancy, epiphenomenon, confounding |

## Implementation: Testing a Hypothesis

```python
import torch
import torch.nn as nn
from typing import Callable, Dict, Any

def evaluate_hypothesis(
    model: nn.Module,
    mechanism_location: str,
    intervention_fn: Callable,
    test_inputs: torch.Tensor,
    behavior_metric: Callable[[torch.Tensor], torch.Tensor]
) -> Dict[str, Any]:
    # Evaluate a mechanistic hypothesis via controlled intervention.
    # Compares clean vs intervened behavior to measure causal effect.

    model.eval()
    device = next(model.parameters()).device
    test_inputs = test_inputs.to(device)

    # Clean forward pass
    with torch.no_grad():
        clean_output = model(test_inputs)
        clean_score = behavior_metric(clean_output)

    # Intervened forward pass using hooks
    parts = mechanism_location.split(".")
    target = model
    for part in parts:
        target = getattr(target, part)

    def hook_fn(module, inputs, output):
        return intervention_fn(output)

    handle = target.register_forward_hook(hook_fn)

    with torch.no_grad():
        intervened_output = model(test_inputs)
        intervened_score = behavior_metric(intervened_output)

    handle.remove()

    effect_size = (intervened_score - clean_score).abs().mean().item()

    return {
        "clean_score": clean_score.mean().item(),
        "intervened_score": intervened_score.mean().item(),
        "effect_size": effect_size,
        "hypothesis_supported": effect_size > 0.05,
        "mechanism_location": mechanism_location,
        "n_samples": test_inputs.shape[0]
    }
```

## Alternative Explanations

- **Downstream effect**: Component activates because behavior occurs, not reverse.
- **Redundancy**: Multiple components implement same function; ablation has no effect.
- **Epiphenomenon**: Component activates coincidentally with no causal role.
- **Correlated confounder**: Third variable drives both component and behavior.
- **Partial cause**: Component necessary but not sufficient.

## Limitations

Mechanistic interpretability does not guarantee:
- Complete understanding of all behaviors
- Human-interpretable descriptions of every computation
- Safety guarantees without additional verification
- Computational tractability for all models

## Exercises

### Mathematical
1. Prove that knowing $f_{\theta}(x)$ for all $x$ in a finite domain does not uniquely determine the internal computation graph.
2. Formalize the seven scientific criteria as logical predicates.

### Implementation
3. Implement `extract_computational_graph(model)` returning DAG as adjacency list.
4. Write test suite verifying `evaluate_hypothesis` on simple MLP.

### Experimental
5. Train 3-layer MLP on MNIST. Identify neuron correlating with digit "7". Ablate it. Measure selective accuracy change.

### Research
6. Formulate mechanistic hypothesis for 2-layer transformer on arithmetic sequence. Design three interventions with null hypotheses and predicted effect sizes.

## References

- Olah, C., et al. (2020). "An Overview of Early Vision in InceptionV1." *Distill*.
- Elhage, N., et al. (2021). "A Mathematical Framework for Transformer Circuits." *Anthropic*.
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge.

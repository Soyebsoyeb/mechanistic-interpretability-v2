# Chapter 1 — What Is Mechanistic Interpretability?

## Motivation

A neural network implements a function $f_{\theta}(x) = y$. Knowing the network's input-output behavior does not uniquely tell us how it computes $y$. Many different internal mechanisms — different computational graphs, different feature representations, different circuit structures — can produce similar outputs. Mechanistic interpretability asks: **What internal computation gives rise to the observed behavior?**

The goal is not simply to describe what the model does. The goal is to recover a computational description of *how* it does it. This is reverse engineering in the classical sense: given a functioning artifact, determine the algorithm it implements.

## Learning Objectives

By the end of this chapter, you will:
- Distinguish black-box analysis from mechanistic analysis
- Identify six levels of explanation in neural network interpretation
- Apply seven scientific criteria to evaluate mechanistic hypotheses
- Design an intervention to test a mechanistic claim
- Recognize the epistemic limits of correlation-based interpretation

## Intuition

Consider a pocket calculator. Behavioral analysis asks: "What is $2+2$?" and receives the answer $4$. This tells us nothing about the internal logic gates, adder circuits, or binary arithmetic that produced the answer. Mechanistic analysis opens the calculator, traces electrical signals through NAND gates, half-adders, and full-adders, and explains *why* the output is $4$ based on the internal structure.

Neural networks are far more complex than calculators, but the ambition is the same: reverse engineer the internal algorithm from the trained parameters. The difference is that neural networks have millions or billions of parameters arranged in high-dimensional vector spaces, making the reverse engineering problem vastly more difficult — but not fundamentally different in kind.

## Formal Definition

**Mechanistic interpretability** is the scientific discipline that seeks to reverse engineer the computations implemented by trained neural networks by identifying and validating causal relationships between internal components and observable behaviors.

### Levels of Explanation

| Level | Description | Example | Strength |
|-------|-------------|---------|----------|
| Behavioral | Input-output mapping | "The model predicts positive sentiment" | Weak |
| Statistical | Correlation patterns | "The output correlates with sentiment words" | Weak |
| Feature | Identified representation | "A direction in layer 5 encodes sentiment polarity" | Moderate |
| Component | Specific unit | "Neuron 847 activates on negative phrases" | Moderate |
| Circuit | Subgraph of components | "Heads 3.1 and 4.2 form a sentiment circuit" | Strong |
| Algorithmic | Full procedure | "The model computes sentiment via negation detection → polarity accumulation → threshold comparison" | Strongest |

A statement such as "the model predicts positive sentiment because its output correlates with sentiment" is **not** a mechanistic explanation. It is a statistical observation. A mechanistic explanation must identify the feature, its representation, its downstream consumers, and demonstrate causal relevance through intervention.

### The Causal Chain

A complete mechanistic explanation traces a causal chain:

```
Input → Feature Detection → Feature Representation → Information Routing → Decision Computation → Output
```

Each arrow in this chain must be validated. Missing any link leaves the explanation incomplete.

## Scientific Criteria for Mechanistic Hypotheses

For a hypothesis $H$ about mechanism $M$ producing behavior $B$:

1. **Localization**: Where is $M$ in the model? Specify layer index, component type, head index, neuron index, and tensor coordinates. Vague claims like "somewhere in the middle layers" are insufficient.

2. **Specification**: What exactly does $M$ compute? Provide a mathematical function $M: \mathbb{R}^d \rightarrow \mathbb{R}^k$ with explicit formula.

3. **Causal relevance**: Does modifying $M$ change $B$? Require intervention evidence. Correlation alone is never sufficient.

4. **Sufficiency**: Does $M$ reproduce $B$? The proposed mechanism should explain the behavior without invoking unexplained components. If the explanation requires "and then some other stuff happens," it is incomplete.

5. **Generalization**: Does $M$ work on examples not used to discover it? Test on held-out data, different token identities, different syntactic structures, and different semantic domains.

6. **Falsifiability**: What result $R$ would prove $H$ wrong? State $R$ before running experiments. If no possible observation could falsify $H$, it is not a scientific hypothesis.

7. **Independence**: Is $M$ independently identifiable, or does it depend on other components in ways that make isolation impossible?

## The Eight Questions Framework

For every claim in this book, we demand answers:

| Question | Chapter 1 Example |
|----------|-------------------|
| What is being claimed? | "Attention head 2 in layer 3 copies information from position $i$ to position $j$" |
| What mathematical object? | The OV matrix $W_V^{(3,2)} W_O^{(3,2)} \in \mathbb{R}^{d \times d}$ |
| Where in the model? | Layer 3, head 2, post-attention residual stream |
| How measured? | Attention weight $A_{j,i}$ and output projection onto target direction |
| How intervened? | Zero-ablate head output; patch OV circuit from clean to corrupted |
| What falsifies? | Ablating the head does not change copying behavior on held-out data |
| Reproducible? | Yes, with fixed seed 42, checkpoint `gpt2`, prompt template `[A][B]...[A]` |
| Alternatives? | The head may be redundant; another head may compensate; copying may be epiphenomenal |

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
    behavior_metric: Callable,
    n_repetitions: int = 5
) -> Dict[str, Any]:
    """Evaluate a mechanistic hypothesis via intervention.

    Args:
        model: The neural network under investigation
        mechanism_location: Dot-path to component, e.g. "blocks.5.attn.heads.2"
        intervention_fn: Callable that modifies the mechanism activation
        test_inputs: Input tensor [batch, ...]
        behavior_metric: Callable(output) -> float measuring target behavior
        n_repetitions: Number of independent runs for uncertainty estimation

    Returns:
        Dict with clean_score, intervened_score, difference, confidence interval,
        and hypothesis_support boolean.
    """
    model.eval()
    clean_scores = []
    intervened_scores = []

    for seed in range(n_repetitions):
        torch.manual_seed(seed)

        with torch.no_grad():
            clean_output = model(test_inputs)
            clean_scores.append(behavior_metric(clean_output).item())

        # Intervened forward pass with hook
        intervened_score = _run_with_intervention(
            model, test_inputs, mechanism_location, 
            intervention_fn, behavior_metric
        )
        intervened_scores.append(intervened_score.item())

    clean_mean = sum(clean_scores) / len(clean_scores)
    inter_mean = sum(intervened_scores) / len(intervened_scores)
    clean_std = (sum((s - clean_mean)**2 for s in clean_scores) / len(clean_scores)) ** 0.5

    return {
        "clean_mean": clean_mean,
        "clean_std": clean_std,
        "intervened_mean": inter_mean,
        "difference": clean_mean - inter_mean,
        "effect_size": abs(clean_mean - inter_mean) / (clean_std + 1e-8),
        "hypothesis_supported": abs(inter_mean - clean_mean) > 2 * clean_std,
        "n_repetitions": n_repetitions
    }


def _run_with_intervention(model, inputs, location, intervention_fn, metric_fn):
    """Internal: run model with intervention hook applied."""
    # Navigate to target module
    parts = location.split(".")
    target = model
    for part in parts:
        target = getattr(target, part)

    handle = target.register_forward_hook(
        lambda m, inp, out: intervention_fn(out)
    )

    with torch.no_grad():
        output = model(inputs)
        score = metric_fn(output)

    handle.remove()
    return score
```

## Alternative Explanations

Even when a component correlates with a behavior, consider:

- **Downstream effect**: The component may read information computed elsewhere. It activates when the behavior occurs because it is a consumer, not a cause.
- **Redundancy**: Multiple components may implement the same function. Ablating one has no effect because others compensate.
- **Epiphenomenon**: The component may activate coincidentally with no causal role. The correlation is spurious.
- **Correlated confounder**: A third variable drives both the component and the behavior. The component is a proxy, not a mechanism.
- **Common cause**: Both the component and the behavior are effects of an upstream cause. Intervening on the component does not change the behavior because the upstream cause remains.

## Limitations

Mechanistic interpretability does not guarantee:
- Complete understanding of all model behaviors (the problem may be computationally intractable)
- Human-interpretable descriptions of every computation (some computations may not have compact descriptions)
- Safety guarantees without additional verification (understanding a mechanism does not prove it is safe)
- Transfer across model scales (mechanisms in small models may differ from large models)

## Exercises

### Mathematical
1. Prove that knowing $f_{\theta}(x)$ for all $x$ in a finite domain does not uniquely determine the internal computation graph. Construct two distinct networks that implement the same function on a given domain.
2. Formalize the six levels of explanation as a hierarchy of equivalence relations on the space of neural networks.

### Implementation
3. Implement a function `extract_computational_graph(model)` that returns a DAG representing a neural network's forward pass as an adjacency list. Test it on a simple MLP and a transformer block.
4. Write a test suite that verifies the `evaluate_hypothesis` function returns consistent results across repeated runs with the same seed.

### Experimental
5. Train a small MLP (2 hidden layers, 128 neurons each) on MNIST. Identify a neuron that correlates with the digit "7" using maximum activating examples. Ablate it. Measure accuracy change on "7" vs. other digits. Report whether the neuron is causal or merely correlated.

### Research
6. Formulate a mechanistic hypothesis for how a transformer predicts the next token in a simple arithmetic sequence (e.g., "2, 4, 6, 8, ..."). Design three distinct interventions to test it. For each intervention, state the null hypothesis, the expected result, and the falsifying result.
7. Investigate whether mechanistic explanations of small models (under 1M parameters) transfer to larger models trained on the same task. What breaks? What generalizes?

## References

- Olah, C., Mordvintsev, A., & Schubert, L. (2017). "Feature Visualization." *Distill*.
- Olah, C., et al. (2020). "An Overview of Early Vision in InceptionV1." *Distill*.
- Elhage, N., et al. (2021). "A Mathematical Framework for Transformer Circuits." *Anthropic*.
- Nanda, N. (2022). "A Comprehensive Mechanistic Interpretability Explainer."
- Meng, K., et al. (2022). "Locating and Editing Factual Associations in GPT." *NeurIPS*.

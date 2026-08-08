# Chapter 5 — Experimental Methodology

## Motivation
Mechanistic interpretability is an experimental science. A serious experiment begins with a behavioral phenomenon, a precise hypothesis, and a controlled intervention.

## The Experimental Workflow

```
Behavior -> Observation -> Hypothesis -> Mathematical Model -> Code -> Intervention -> Evidence -> Mechanistic Explanation
```

### Step 1: Define the Task
Specify: input distribution, target behavior, metric, baseline, control condition.

### Step 2: Form a Hypothesis
Example: "Attention head $h$ implements copying from position $i$ to $j$ when token $i$ matches token $j-1$."

### Step 3: Localize
Inspect: activations, attention patterns, attribution, gradients, feature activations.

### Step 4: Intervene
Modify suspected component. Measure effect.

### Step 5: Measure
Compare $M_{\text{clean}}$, $M_{\text{corrupted}}$, $M_{\text{patched}}$.

Normalized score:
$$S = \frac{M_{\text{patched}} - M_{\text{corrupted}}}{M_{\text{clean}} - M_{\text{corrupted}}}$$

- $S \approx 0$: little recovery
- $S \approx 1$: strong recovery

### Step 6: Falsify
Search for: counterexamples, inputs where component activates but behavior absent, alternative components.

### Step 7: Reproduce
Store: code version, model identifier, tokenizer, dataset, seeds, hardware, configuration.

## Implementation

```python
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Any
import json

@dataclass
class ExperimentConfig:
    experiment_id: str
    chapter: str
    research_question: str
    model_name: str
    model_revision: str
    tokenizer_name: str
    dataset_source: str
    task: str
    metric: str
    hypothesis: str
    null_hypothesis: str
    intervention: str
    expected_result: str
    random_seed: int
    hardware: str
    code_commit: str

class InterpretabilityExperiment:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {}
        torch.manual_seed(config.random_seed)

    def normalize_score(self, clean, corrupted, patched):
        denominator = clean - corrupted
        if abs(denominator) < 1e-8:
            return 0.0
        return (patched - corrupted) / denominator

    def save(self, path: str):
        output = {"config": asdict(self.config), "results": self.results}
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
```

## Controls
1. Negative control: Patch irrelevant component, expect $S \approx 0$
2. Positive control: Patch full clean state, expect $S \approx 1$
3. Random baseline: Compare against random directions
4. Distribution baseline: Measure on random inputs

## Common Pitfalls
- Cherry-picking: Report on held-out test sets
- Multiple comparisons: Apply Bonferroni correction
- OOD interventions: Verify model internals are not broken
- Confounding: Use causal interventions, not correlation
- Overfitting: Test on unseen templates

## Reproducibility Checklist
- [ ] Model identifier recorded
- [ ] Model revision recorded
- [ ] Tokenizer recorded
- [ ] Dataset source recorded
- [ ] Preprocessing documented
- [ ] Code version (git commit) recorded
- [ ] Dependencies pinned
- [ ] Random seeds recorded
- [ ] Hardware recorded
- [ ] Configuration saved
- [ ] Raw results saved
- [ ] Figures reproducible
- [ ] Statistical uncertainty reported
- [ ] Limitations documented

## Exercises
- **Mathematical**: Show that if $M_{\text{clean}} = M_{\text{corrupted}}$, normalized score $S$ is undefined.
- **Implementation**: Implement `InterpretabilityExperiment` for MLP ablation.
- **Experimental**: Design controlled experiment on 2-layer transformer. Document all seven steps.
- **Research**: Propose novel control for activation patching addressing OOD intervention problem.

## References
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*.
- Meng, K., et al. (2022). "Locating and Editing Factual Associations in GPT."

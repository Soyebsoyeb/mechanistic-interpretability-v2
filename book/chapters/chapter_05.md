# Chapter 5 — Experimental Methodology

## Motivation

Mechanistic interpretability is an experimental science. A serious interpretability experiment does not begin with visualization; it begins with a behavioral phenomenon, a precise hypothesis, and a controlled intervention. This chapter establishes the experimental protocol that governs all subsequent work in this book.

## Learning Objectives

- Design controlled interpretability experiments with proper baselines and controls
- Formulate falsifiable mechanistic hypotheses with explicit mathematical objects
- Apply appropriate statistical methods and correction for multiple comparisons
- Document experiments comprehensively for reproducibility
- Recognize and avoid common experimental pitfalls

## The Experimental Workflow

```
Behavior → Observation → Hypothesis → Mathematical Model → Code → Intervention → Evidence → Mechanistic Explanation
```

### Step 1: Define the Task

Specify with precision:
- **Input distribution**: What inputs will you study? Define the distribution $\mathcal{D}$ explicitly.
- **Target behavior**: What output pattern are you trying to explain? Define the metric $B(x)$.
- **Metric**: How will you quantify the behavior? Use a scalar-valued function.
- **Baseline**: What is the expected behavior under a null model or random chance?
- **Control condition**: What inputs should *not* trigger the behavior?

### Step 2: Form a Hypothesis

Example hypothesis: "Attention head $h$ at layer 3, head 2 implements a copying mechanism from position $i$ to position $j$ when token at $j$ matches token at $i-1$."

A good hypothesis specifies:
- **Component**: Which layer, head, neuron, or circuit? Use exact coordinates.
- **Function**: What computation does it perform? Provide a mathematical formula.
- **Condition**: Under what inputs does it activate? Define the triggering distribution.
- **Output**: How does it affect the final prediction? State the causal chain.

### Step 3: Localize

Inspect multiple sources of evidence:
- **Activations**: Which components activate on target inputs?
- **Attention patterns**: Which heads attend to relevant positions?
- **Attribution**: Which components contribute most to the target output?
- **Gradients**: Which parameters most influence the target?
- **Feature activations**: Which SAE features fire on target inputs?
- **Logit lens**: What do intermediate layers predict?

Localization is exploratory. It generates candidates, not conclusions.

### Step 4: Intervene

Modify the suspected component and measure the effect. This is the causal step.

### Step 5: Measure

Compare three conditions:
- $M_{\text{clean}}$: Metric on unmodified inputs
- $M_{\text{corrupted}}$: Metric on corrupted inputs (behavior disrupted)
- $M_{\text{patched}}$: Metric when component is patched from clean to corrupted

Normalized patching score:

$$S_i = \frac{M_{\text{patched}} - M_{\text{corrupted}}}{M_{\text{clean}} - M_{\text{corrupted}}}$$

Interpretation:
- $S_i \approx 0$: The component does not recover the behavior (not necessary)
- $S_i \approx 1$: The component fully recovers the behavior (necessary and sufficient)
- $S_i < 0$: The component actively harms the behavior (unexpected — investigate)
- $S_i > 1$: The patched component over-recovers (investigate — may indicate nonlinearity or compensation)

### Step 6: Falsify

Actively search for:
- Counterexamples where the hypothesis predicts behavior but the model does something else
- Inputs where the component activates but the behavior does not occur
- Alternative components that produce the same behavior when patched
- Edge cases that break the hypothesized mechanism

### Step 7: Reproduce

Store:
- Code version (git commit hash)
- Model identifier and revision
- Tokenizer version
- Dataset source and preprocessing
- Random seeds
- Hardware and software versions
- Configuration files
- Raw results and generated figures

## Implementation: Experiment Template

```python
import json
import torch
from dataclasses import dataclass, asdict
from typing import Callable, Dict, Any, Optional
from datetime import datetime

@dataclass
class ExperimentConfig:
    """Standard experiment configuration."""
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
    software_versions: Optional[Dict[str, str]] = None

class InterpretabilityExperiment:
    """Base class for mechanistic interpretability experiments."""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {}
        self.metadata = {
            "start_time": datetime.now().isoformat(),
            "config": asdict(config)
        }
        self.set_seed(config.random_seed)

    def set_seed(self, seed: int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def run_clean(self, model, inputs) -> Dict[str, Any]:
        """Run clean forward pass."""
        raise NotImplementedError

    def run_corrupted(self, model, inputs) -> Dict[str, Any]:
        """Run corrupted forward pass."""
        raise NotImplementedError

    def run_patched(self, model, inputs, component) -> Dict[str, Any]:
        """Run patched forward pass."""
        raise NotImplementedError

    def compute_metric(self, outputs, targets) -> float:
        """Compute target metric."""
        raise NotImplementedError

    def normalize_score(self, clean, corrupted, patched) -> float:
        """Compute normalized patching score."""
        denominator = clean - corrupted
        if abs(denominator) < 1e-8:
            return 0.0
        return (patched - corrupted) / denominator

    def run_full_pipeline(self, model, inputs, component):
        """Run complete experiment pipeline."""
        clean = self.run_clean(model, inputs)
        corrupted = self.run_corrupted(model, inputs)
        patched = self.run_patched(model, inputs, component)

        score = self.normalize_score(
            self.compute_metric(clean),
            self.compute_metric(corrupted),
            self.compute_metric(patched)
        )

        self.results = {
            "clean": clean,
            "corrupted": corrupted,
            "patched": patched,
            "score": score,
            "supports_hypothesis": score > 0.5
        }
        return self.results

    def save(self, path: str):
        """Save experiment results and metadata."""
        self.metadata["end_time"] = datetime.now().isoformat()
        output = {
            "metadata": self.metadata,
            "results": self.results
        }
        with open(path, 'w') as f:
            json.dump(output, f, indent=2)
```

## Controls and Baselines

Every experiment requires multiple controls:

1. **Negative control**: Patch a component known to be irrelevant. Expect $S \approx 0$.
2. **Positive control**: Patch the full clean state into corrupted. Expect $S \approx 1$.
3. **Random baseline**: Compare against random directions or random components.
4. **Distribution baseline**: Measure metric on random inputs from the data distribution.
5. **Shuffle control**: Randomly shuffle activations before patching. Tests whether temporal/positional structure matters.

## Statistical Considerations

### Multiple Comparisons

When testing many components (e.g., all attention heads), apply correction:
- **Bonferroni**: Divide $\alpha$ by number of tests. Conservative but simple.
- **False Discovery Rate (FDR)**: Control expected proportion of false discoveries. More powerful.

### Effect Sizes

Report effect sizes, not just p-values:
- **Cohen's d**: $(\mu_1 - \mu_2) / \sigma$
- **Normalized score**: $S_i$ as defined above
- **Percentage change**: $(M_{\text{patched}} - M_{\text{corrupted}}) / M_{\text{corrupted}} \times 100$

### Confidence Intervals

Report confidence intervals for all metrics. A point estimate without uncertainty is incomplete.

## Common Pitfalls

| Pitfall | Description | Solution |
|---------|-------------|----------|
| **Cherry-picking examples** | Selecting only examples that support the hypothesis | Report metrics on held-out test sets |
| **Multiple comparisons** | Testing many components without correction | Apply Bonferroni or FDR |
| **Out-of-distribution interventions** | Intervention breaks model internals | Verify intervention preserves model stability |
| **Confounding by correlation** | Correlation mistaken for causation | Use causal interventions, not just correlation |
| **Overfitting hypotheses** | Hypothesis tailored to specific examples | Test on unseen templates and token identities |
| **Publication bias** | Only reporting successful experiments | Report negative results and failure cases |
| **HARKing** | Hypothesizing after results are known | Pre-register hypotheses before experiments |

## Reproducibility Checklist

Before considering an experiment complete:

- [ ] Model identifier recorded
- [ ] Model revision (commit hash or version) recorded
- [ ] Tokenizer name and revision recorded
- [ ] Dataset source and license recorded
- [ ] Preprocessing pipeline documented
- [ ] Code version (git commit hash) recorded
- [ ] Dependencies pinned in `requirements.txt`
- [ ] Random seeds recorded (PyTorch, NumPy, Python)
- [ ] Hardware (GPU model, CUDA version) recorded
- [ ] Experiment configuration saved as JSON/YAML
- [ ] Raw results saved (not just figures)
- [ ] Figures generated programmatically from raw results
- [ ] Statistical uncertainty reported (confidence intervals, standard errors)
- [ ] Effect sizes reported
- [ ] Negative results and failure cases documented
- [ ] Limitations explicitly stated

## Exercises

### Mathematical
1. Show that if $M_{\text{clean}} = M_{\text{corrupted}}$, the normalized score $S$ is undefined. What does this imply experimentally? How should one handle this case?
2. Derive the Bonferroni correction: if $n$ independent tests are performed at level $\alpha$, the family-wise error rate is $\leq n\alpha$.
3. Prove that the normalized score $S_i$ is bounded: $S_i \in (-\infty, \infty)$. Under what conditions can $S_i > 1$ or $S_i < 0$?

### Implementation
4. Implement the `InterpretabilityExperiment` base class with concrete methods for a simple MLP ablation study. Include all controls and statistical reporting.
5. Write a function that applies False Discovery Rate correction to a list of p-values using the Benjamini-Hochberg procedure.

### Experimental
6. Design and run a controlled experiment on a 2-layer transformer trained on a synthetic copying task. Document all seven steps of the workflow explicitly.
7. Perform an experiment where you test all attention heads in a model for a specific behavior. Apply FDR correction. Report how many heads pass the corrected threshold vs. uncorrected.

### Research
8. Propose a novel control condition for activation patching that addresses the "out-of-distribution intervention" problem. Implement and validate it.
9. Investigate whether mechanistic hypotheses discovered on small datasets generalize to larger, more diverse datasets. Quantify the generalization gap.
10. Design a pre-registration protocol for mechanistic interpretability experiments. What should be specified before any data is collected?

## References

- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*, 2nd ed.
- Meng, K., et al. (2022). "Locating and Editing Factual Associations in GPT." *NeurIPS*.
- Benjamini, Y., & Hochberg, Y. (1995). "Controlling the False Discovery Rate." *JRSSB*.

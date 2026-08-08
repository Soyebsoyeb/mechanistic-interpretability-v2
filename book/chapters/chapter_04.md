# Chapter 4 — Probability, Information, and Optimization

## Motivation

Neural networks are trained by optimizing probability distributions over high-dimensional spaces. Language models estimate conditional distributions $P(x_t | x_{<t})$. To interpret these systems, we need the mathematical tools of probability theory, information theory, and optimization — not as abstract formalisms, but as diagnostic instruments for measuring, comparing, and understanding internal representations.

## Learning Objectives

- Compute cross-entropy and KL divergence for language model outputs
- Apply mutual information to quantify representation-content relationships
- Understand how optimization shapes internal representations
- Use information-theoretic measures as diagnostic tools (not causal proofs)
- Analyze training dynamics and their effect on feature formation

## Language Model Probability

An autoregressive language model estimates the joint probability of a sequence by factorization:

$$P(x_1, \ldots, x_T) = \prod_{t=1}^T P(x_t | x_{<t})$$

At position $t$, the model outputs a distribution over the vocabulary $\mathcal{V}$:

$$p_t = \text{softmax}(z_t) \in \Delta^{|\mathcal{V}|-1}$$

where $z_t = x_t^{(L)} W_U$ is the logit vector and $W_U \in \mathbb{R}^{d \times |\mathcal{V}|}$ is the unembedding matrix.

### Perplexity

Perplexity is the exponential of average cross-entropy:

$$\text{PPL} = \exp\left(-\frac{1}{T} \sum_{t=1}^T \log p_t(x_t)\right)$$

Lower perplexity indicates better prediction. Perplexity is the standard evaluation metric for language models.

## Cross-Entropy

For target distribution $p$ and model distribution $q$:

$$H(p, q) = -\sum_x p(x) \log q(x)$$

For a one-hot target $y$ (the true next token):

$$H(y, q) = -\log q(y)$$

The training objective minimizes the expected cross-entropy over the data distribution:

$$\mathcal{L}(\theta) = \mathbb{E}_{x \sim \mathcal{D}}\left[-\log p_\theta(x)\right]$$

## KL Divergence

$$D_{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)}$$

Properties:
- $D_{KL}(p \| q) \geq 0$ with equality iff $p = q$ (Gibbs' inequality)
- Not symmetric: $D_{KL}(p \| q) \neq D_{KL}(q \| p)$
- Not a metric (does not satisfy triangle inequality)

**Interpretability application**: After intervening on a component, compare $D_{KL}(p_{\text{clean}} \| p_{\text{intervened}})$ to measure the intervention's effect on the output distribution. Large KL indicates the component is important for the output.

## Mutual Information

For random variables $X, Y$:

$$I(X; Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$

Equivalently:

$$I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)$$

### Critical Caveat

High mutual information between a representation $h$ and a variable $Y$ indicates $h$ *carries information* about $Y$, but does **not** establish that $h$ *causes* $Y$ or that $Y$ causes $h$. Information alone does not establish causal relevance.

**Example**: A clock on the wall and the position of the sun have high mutual information with time of day, but neither causes the other.

## Optimization

Training minimizes a loss function $\mathcal{L}(\theta)$:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}(\theta_t)$$

For stochastic gradient descent:

$$\theta_{t+1} = \theta_t - \eta \nabla_\theta \mathcal{L}_i(\theta_t)$$

where $\mathcal{L}_i$ is the loss on a mini-batch.

### Representation Structure Emergence

Representation structure emerges from the interaction of:
1. **Architecture**: Inductive biases (attention, convolutions, recurrence)
2. **Optimization algorithm**: Gradient descent, Adam, learning rate schedule
3. **Initialization**: Weight distribution at $t=0$
4. **Data distribution**: What patterns exist in training data
5. **Objective function**: What the model is incentivized to learn

No single factor determines representation structure. This makes interpretability challenging: the same architecture trained on different data may develop different internal representations.

## Implementation

```python
import torch
import torch.nn.functional as F

def compute_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy loss.

    Args:
        logits: torch.Tensor [batch, seq, vocab_size]
        targets: torch.Tensor [batch, seq], integer indices

    Returns:
        torch.Tensor scalar
    """
    return F.cross_entropy(
        logits.view(-1, logits.size(-1)), 
        targets.view(-1),
        reduction='mean'
    )


def compute_perplexity(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute perplexity.

    Args:
        logits: [batch, seq, vocab]
        targets: [batch, seq]

    Returns:
        float, perplexity
    """
    ce = compute_cross_entropy(logits, targets)
    return torch.exp(ce).item()


def compute_kl_divergence(p_clean: torch.Tensor, p_intervened: torch.Tensor, 
                           eps: float = 1e-10) -> torch.Tensor:
    """Compute KL(p_clean || p_intervened).

    Args:
        p_clean: torch.Tensor [..., vocab], probability distribution
        p_intervened: torch.Tensor [..., vocab], probability distribution

    Returns:
        torch.Tensor [...]
    """
    ratio = p_clean / (p_intervened + eps)
    kl = (p_clean * torch.log(ratio + eps)).sum(dim=-1)
    return kl


def estimate_mutual_information_binning(h: torch.Tensor, y: torch.Tensor, 
                                       num_bins: int = 10) -> float:
    """Estimate mutual information I(h; y) via binning.

    Args:
        h: torch.Tensor [N, d], representations
        y: torch.Tensor [N], discrete labels

    Returns:
        float, estimated mutual information in nats
    """
    # Project to first principal component for binning
    if h.dim() > 1:
        h_proj = h @ torch.randn(h.size(1), 1, device=h.device)
        h_proj = h_proj.squeeze()
    else:
        h_proj = h

    # Create bins
    min_val, max_val = h_proj.min(), h_proj.max()
    bins = torch.linspace(min_val, max_val, num_bins + 1, device=h.device)
    h_discrete = torch.bucketize(h_proj, bins).clamp(1, num_bins) - 1

    # Joint and marginal distributions
    n_classes = int(y.max().item()) + 1
    joint = torch.zeros(num_bins, n_classes, device=h.device)
    for i in range(len(h_discrete)):
        joint[h_discrete[i], y[i]] += 1

    joint /= joint.sum()
    p_h = joint.sum(dim=1)
    p_y = joint.sum(dim=0)

    # Compute MI
    mi = 0.0
    for i in range(num_bins):
        for j in range(n_classes):
            if joint[i, j] > 0:
                mi += joint[i, j] * torch.log(joint[i, j] / (p_h[i] * p_y[j] + 1e-10))

    return mi.item()


def compute_js_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    """Jensen-Shannon divergence.

    JS(p||q) = 0.5 * KL(p||m) + 0.5 * KL(q||m) where m = 0.5(p+q)
    """
    m = 0.5 * (p + q)
    kl_pm = compute_kl_divergence(p, m, eps)
    kl_qm = compute_kl_divergence(q, m, eps)
    return 0.5 * (kl_pm + kl_qm)
```

## Measurement: Information Content of Representations

Given a dataset $\{(x_i, y_i)\}_{i=1}^N$ and representations $\{h_i\}$:

1. **Linear mutual information**: Train linear probe, compute $I(\hat{y}; y)$
2. **Nonlinear mutual information**: Use MINE (Mutual Information Neural Estimation) or InfoNCE
3. **Layer-wise progression**: Plot $I(h^{(\ell)}; y)$ vs. $\ell$ to track information flow
4. **Information bottleneck**: Plot $I(h^{(\ell)}; x)$ vs. $I(h^{(\ell)}; y)$ to identify compression

## Intervention: Distribution Shift

```python
def measure_distribution_shift(model, clean_inputs, corrupted_inputs, 
                               intervention_fn, layer_name):
    """Measure KL divergence between clean and intervened outputs."""
    with torch.no_grad():
        clean_logits = model(clean_inputs)
        clean_probs = F.softmax(clean_logits, dim=-1)

        intervened_logits = run_with_intervention(
            model, corrupted_inputs, layer_name, intervention_fn
        )
        intervened_probs = F.softmax(intervened_logits, dim=-1)

        kl = compute_kl_divergence(clean_probs, intervened_probs)
        js = compute_js_divergence(clean_probs, intervened_probs)

    return {
        "kl_mean": kl.mean().item(),
        "kl_max": kl.max().item(),
        "js_mean": js.mean().item()
    }
```

## Falsification

A claim that "representation $h$ encodes feature $Y$" based solely on mutual information is falsified if:
- Intervening on $h$ does not change $Y$ (no causal effect)
- $h$ correlates with $Y$ only on the training distribution (poor generalization)
- A random projection of $h$ has comparable mutual information with $Y$ (no specificity)
- The mutual information estimate is statistically indistinguishable from zero

## Alternative Explanations

- **Spurious correlation**: $h$ and $Y$ may both depend on a confounder $Z$
- **Information bottleneck**: $h$ may discard information about $Y$ that is irrelevant for the training objective
- **Redundant encoding**: Multiple representations may encode $Y$; $h$ is not the unique locus
- **Compression artifact**: High $I(h; Y)$ may reflect compression of $X$ rather than explicit encoding of $Y$

## Exercises

### Mathematical
1. Prove that $D_{KL}(p \| q) \geq 0$ using Jensen's inequality.
2. Show that $I(X; Y) = D_{KL}(p(x,y) \| p(x)p(y))$ and derive the equivalence with $H(X) - H(X|Y)$.
3. Prove that the cross-entropy loss is minimized when $q = p$ (the true distribution).
4. Show that for any representation $h$, $I(h; Y) \leq I(X; Y)$ where $X$ is the input. What does this imply about information loss in neural networks?

### Implementation
5. Implement a linear probe and compute $R^2$ for predicting $Y$ from $h$. Compare with a random baseline and report statistical significance.
6. Write a function that computes the Jensen-Shannon divergence between two distributions and verify its symmetry.
7. Implement a simple MINE estimator for mutual information.

### Experimental
8. For a trained transformer, compute $I(h^{(\ell)}; y)$ for each layer $\ell$ on a classification task. Plot the information curve. Identify where information about $y$ first appears and where it is most concentrated.

### Research
9. Investigate whether gradient-based attribution methods approximate causal effects or merely correlation structures. Design an experiment to test this.
10. Study how the information bottleneck curve ($I(h; X)$ vs. $I(h; Y)$) changes during training. Does the network first memorize then compress?

## References

- Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*, 2nd ed.
- Tishby, N., & Zaslavsky, N. (2015). "Deep Learning and the Information Bottleneck Principle."
- Belghazi, M. I., et al. (2018). "MINE: Mutual Information Neural Estimation." *ICML*.

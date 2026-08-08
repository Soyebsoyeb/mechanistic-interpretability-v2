# Chapter 4 — Probability, Information, and Optimization

## Motivation
Neural networks are trained by optimizing probability distributions. Language models estimate $P(x_t | x_{<t})$. We need probability theory, information theory, and optimization.

## Language Model Probability

Autoregressive model:
$$P(x_1, \ldots, x_T) = \prod_{t=1}^T P(x_t | x_{<t})$$

At position $t$: $p_t = \text{softmax}(z_t) \in \Delta^{|\mathcal{V}|-1}$ where $z_t = x_t^{(L)} W_U$.

## Cross-Entropy

For target $p$ and model $q$:
$$H(p, q) = -\sum_x p(x) \log q(x)$$

For one-hot target $y$: $H(y, q) = -\log q(y)$.

## KL Divergence

$$D_{KL}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} \geq 0$$

**Application**: After intervention, compare $D_{KL}(p_{\text{clean}} \| p_{\text{intervened}})$.

## Mutual Information

$$I(X; Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}$$

**Critical caveat**: High $I(h; Y)$ means $h$ carries information about $Y$, but does **not** establish causation.

## Optimization

Gradient descent: $\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)$

Representation structure emerges from: architecture, optimization, initialization, data, objective.

## Implementation

```python
import torch
import torch.nn.functional as F

def compute_cross_entropy(logits, targets):
    return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='mean')

def compute_kl_divergence(p_clean, p_intervened, eps=1e-10):
    ratio = p_clean / (p_intervened + eps)
    return (p_clean * torch.log(ratio + eps)).sum(dim=-1)

def estimate_mutual_information(h, y, num_bins=10):
    h_proj = h @ torch.randn(h.size(1), 1).to(h.device)
    h_proj = h_proj.squeeze()
    min_val, max_val = h_proj.min(), h_proj.max()
    bins = torch.linspace(min_val, max_val, num_bins + 1).to(h.device)
    h_discrete = torch.bucketize(h_proj, bins)
    joint = torch.zeros(num_bins, y.max().item() + 1).to(h.device)
    for i in range(len(h_discrete)):
        joint[h_discrete[i] - 1, y[i]] += 1
    joint /= joint.sum()
    p_h = joint.sum(dim=1)
    p_y = joint.sum(dim=0)
    mi = 0.0
    for i in range(num_bins):
        for j in range(y.max().item() + 1):
            if joint[i, j] > 0:
                mi += joint[i, j] * torch.log(joint[i, j] / (p_h[i] * p_y[j] + 1e-10))
    return mi.item()
```

## Measurement
1. Linear mutual information: $I(\hat{y}; y)$ via linear probe
2. Nonlinear mutual information: MINE or InfoNCE
3. Layer-wise progression: $I(h^{(\ell)}; y)$ vs $\ell$

## Falsification
Falsified if intervention on $h$ does not change $Y$, or if random projection has comparable MI.

## Exercises
- **Mathematical**: Prove $D_{KL} \geq 0$ via Jensen's inequality. Show $I(X;Y) = D_{KL}(p(x,y) \| p(x)p(y))$.
- **Implementation**: Implement linear probe. Compute Jensen-Shannon divergence.
- **Experimental**: Compute $I(h^{(\ell)}; y)$ for each layer. Plot information curve.
- **Research**: Do gradient attribution methods approximate causal effects or correlations?

## References
- Cover, T. M., & Thomas, J. A. (2006). *Elements of Information Theory*.
- Tishby, N., & Zaslavsky, N. (2015). "Deep Learning and the Information Bottleneck Principle."

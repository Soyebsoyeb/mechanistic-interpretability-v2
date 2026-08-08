# Chapter 7 — The Residual Stream

## Motivation

The residual stream is the central communication channel of the transformer. Every layer reads from it and writes to it. Understanding the residual stream is essential for understanding how information flows, accumulates, and transforms through the network. Without the residual stream, there is no mechanism for information to propagate across layers.

## Learning Objectives

- Decompose the residual stream into its constituent contributions from each layer
- Track information flow through layers using directional analysis
- Measure the contribution of specific components to specific directions
- Understand why the residual stream enables deep learning and gradient flow
- Apply residual stream analysis to localize features and circuits

## Formal Definition

For layer $\ell$ in a transformer with $L$ total layers:

$$x_{\ell+1} = x_\ell + a_\ell + m_\ell$$

where:
- $x_\ell \in \mathbb{R}^{n \times d}$ is the residual stream at layer $\ell$
- $a_\ell$ is the attention output at layer $\ell$
- $m_\ell$ is the MLP output at layer $\ell$

### Unrolling the Residual Stream

By recursive substitution:

$$x_L = x_0 + \sum_{\ell=0}^{L-1} a_\ell + \sum_{\ell=0}^{L-1} m_\ell$$

This decomposition is **exact** (not approximate) and follows directly from the definition. It is one of the most powerful tools in transformer interpretability because it allows us to attribute the final representation to individual layer contributions.

### Key Insight

The final representation $x_L$ is the sum of:
1. The initial embedding $x_0$ (token + position embeddings)
2. All attention contributions $\sum a_\ell$
3. All MLP contributions $\sum m_\ell$

This means we can ask: **Which layers contribute to feature $v$?** and answer precisely.

## Directional Analysis

For any direction $v \in \mathbb{R}^d$ (with $\|v\| = 1$):

$$v^\top x_L = v^\top x_0 + \sum_{\ell} v^\top a_\ell + \sum_{\ell} v^\top m_\ell$$

This allows us to decompose the activation of feature $v$ in the final layer into contributions from each component.

### Contribution Magnitudes

For each layer $\ell$ and component type $c \in \{\text{attn}, \text{mlp}\}$:

$$\text{Contribution}(\ell, c, v) = \frac{|v^\top h_\ell^{(c)}|}{\sum_{\ell', c'} |v^\top h_{\ell'}^{(c')}|}$$

where $h_\ell^{(\text{attn})} = a_\ell$ and $h_\ell^{(\text{mlp})} = m_\ell$.

This measures the fraction of the total activation along $v$ contributed by each component.

## Implementation: Residual Stream Decomposition

```python
import torch
import torch.nn as nn
from typing import Dict, List

def decompose_residual_stream(model, inputs):
    """Decompose residual stream into contributions from each component.

    Args:
        model: Transformer model with accessible blocks
        inputs: torch.Tensor [batch, seq]

    Returns:
        dict with keys:
            - embedding: [batch, seq, d]
            - attn_contributions: list of [batch, seq, d], length L
            - mlp_contributions: list of [batch, seq, d], length L
            - final: [batch, seq, d]
            - reconstructed: [batch, seq, d] (should equal final)
    """
    cache = {}

    def make_hook(name):
        def hook(module, input, output):
            cache[name] = output.detach()
        return hook

    handles = []

    # Register hooks on each block's attention and MLP
    for i, block in enumerate(model.blocks):
        h1 = block.attn.register_forward_hook(make_hook(f"attn_{i}"))
        h2 = block.mlp.register_forward_hook(make_hook(f"mlp_{i}"))
        handles.extend([h1, h2])

    # Forward pass
    with torch.no_grad():
        embedding = model.embed(inputs)
        cache["embedding"] = embedding.detach()
        final_output = model(inputs)
        cache["final"] = final_output.detach()

    # Remove hooks
    for h in handles:
        h.remove()

    # Decompose
    L = len(model.blocks)
    attn_contribs = [cache[f"attn_{i}"] for i in range(L)]
    mlp_contribs = [cache[f"mlp_{i}"] for i in range(L)]

    # Verify decomposition: x_L = x_0 + sum(a_l) + sum(m_l)
    reconstructed = cache["embedding"]
    for i in range(L):
        reconstructed = reconstructed + attn_contribs[i] + mlp_contribs[i]

    # Check reconstruction accuracy
    recon_error = (reconstructed - cache["final"]).abs().max().item()
    assert recon_error < 1e-5, f"Reconstruction failed: max error {recon_error}"

    return {
        "embedding": cache["embedding"],
        "attn_contributions": attn_contribs,
        "mlp_contributions": mlp_contribs,
        "final": cache["final"],
        "reconstructed": reconstructed,
        "reconstruction_error": recon_error
    }


def analyze_direction_contributions(decomposition, direction, token_idx=0):
    """Analyze which layers contribute to a specific direction.

    Args:
        decomposition: output from decompose_residual_stream
        direction: torch.Tensor [d], unit vector
        token_idx: int, which token to analyze

    Returns:
        dict with contribution magnitudes per layer and component
    """
    direction = direction.to(decomposition["embedding"].device)
    direction = direction / (direction.norm() + 1e-8)

    emb_contrib = (decomposition["embedding"][:, token_idx] @ direction).item()

    attn_contribs = []
    for attn_out in decomposition["attn_contributions"]:
        c = (attn_out[:, token_idx] @ direction).item()
        attn_contribs.append(c)

    mlp_contribs = []
    for mlp_out in decomposition["mlp_contributions"]:
        c = (mlp_out[:, token_idx] @ direction).item()
        mlp_contribs.append(c)

    total = emb_contrib + sum(attn_contribs) + sum(mlp_contribs)

    return {
        "embedding": emb_contrib,
        "attention": attn_contribs,
        "mlp": mlp_contribs,
        "total": total,
        "normalized": {
            "embedding": emb_contrib / (abs(total) + 1e-8),
            "attention": [c / (abs(total) + 1e-8) for c in attn_contribs],
            "mlp": [c / (abs(total) + 1e-8) for c in mlp_contribs]
        }
    }


def plot_residual_contributions(contributions, direction_name=""):
    """Plot layer-wise contributions to a direction."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    L = len(contributions["attention"])
    layers = list(range(L))

    # Raw contributions
    axes[0].bar(layers, contributions["attention"], label="Attention", alpha=0.7)
    axes[0].bar(layers, contributions["mlp"], bottom=contributions["attention"], 
                label="MLP", alpha=0.7)
    axes[0].axhline(y=contributions["embedding"], color='r', linestyle='--', 
                    label="Embedding")
    axes[0].set_xlabel("Layer")
    axes[0].set_ylabel("Contribution")
    axes[0].set_title(f"Raw Contributions: {direction_name}")
    axes[0].legend()

    # Normalized contributions
    norm_attn = contributions["normalized"]["attention"]
    norm_mlp = contributions["normalized"]["mlp"]
    axes[1].bar(layers, norm_attn, label="Attention", alpha=0.7)
    axes[1].bar(layers, norm_mlp, bottom=norm_attn, label="MLP", alpha=0.7)
    axes[1].axhline(y=contributions["normalized"]["embedding"], color='r', 
                    linestyle='--', label="Embedding")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Normalized Contribution")
    axes[1].set_title(f"Normalized Contributions: {direction_name}")
    axes[1].legend()

    plt.tight_layout()
    return fig
```

## Measurement: Contribution Analysis

1. **Layer-wise heatmap**: For each layer and each of $k$ feature directions, plot $v^\top h_\ell$.
2. **Cumulative contribution**: Plot $\sum_{\ell' \leq \ell} v^\top h_{\ell'}$ vs. $\ell$ to see when information accumulates.
3. **Attention vs. MLP ratio**: For each layer, compare $|v^\top a_\ell|$ vs. $|v^\top m_\ell|$.

## Intervention: Residual Stream Patching

```python
def patch_residual_component(model, inputs, layer_idx, component_type,
                               clean_cache, corrupted_cache):
    """Patch a specific component of the residual stream.

    Args:
        component_type: 'attn' or 'mlp'
    """
    def hook_fn(module, input, output):
        return clean_cache[f"{component_type}_{layer_idx}"]

    target = (model.blocks[layer_idx].attn if component_type == 'attn' 
              else model.blocks[layer_idx].mlp)
    handle = target.register_forward_hook(hook_fn)

    with torch.no_grad():
        output = model(inputs)

    handle.remove()
    return output
```

## Falsification

A claim about residual stream contributions is falsified if:
- The decomposition does not reconstruct the final state (indicates missing terms like LayerNorm)
- A component claimed to be important for direction $v$ has $|v^\top h| \approx 0$
- Patching the component does not change behavior predicted to depend on $v$
- The contribution pattern is not stable across different inputs

## Alternative Explanations

- **Nonlinear interactions**: The residual stream decomposition is linear, but LayerNorm introduces nonlinearity that can couple components
- **Attention pattern dependence**: Attention outputs depend on the full residual stream, so $a_\ell$ is not independent of other components
- **MLP nonlinearity**: MLP outputs are nonlinear functions of the residual stream, making linear attribution approximate
- **Cross-layer dependencies**: Later layers may read information written by multiple earlier layers in nonlinear combinations

## Exercises

### Mathematical
1. Prove that $x_L = x_0 + \sum_{\ell=0}^{L-1} (a_\ell + m_\ell)$ by induction on $L$.
2. Show that if all $a_\ell$ and $m_\ell$ were orthogonal to $v$, then $v^\top x_L = v^\top x_0$.
3. Prove that the residual stream enables gradient flow: show that $\frac{\partial x_L}{\partial x_0} = I + \sum_{\ell} \frac{\partial a_\ell}{\partial x_0} + \sum_{\ell} \frac{\partial m_\ell}{\partial x_0}$.
4. Analyze how LayerNorm affects the residual stream decomposition. Is the decomposition still exact after LayerNorm?

### Implementation
5. Implement a hook system that can extract *every* intermediate tensor in a transformer forward pass without modifying the model code.
6. Write a function that verifies the residual stream decomposition to machine precision.

### Experimental
7. For a trained transformer, compute the residual stream decomposition for 100 inputs. Plot a heatmap of layer-wise contributions to the top-10 logit directions.
8. Measure when specific features "appear" in the residual stream by tracking at which layer their contribution first exceeds a threshold.

### Research
9. Investigate whether early layers write "low-level" features (position, syntax) while late layers write "high-level" features (semantics, world knowledge). Quantify this using the residual stream decomposition.
10. Study whether information written by early layers is preserved or overwritten by later layers. Design an experiment to measure information persistence.

## References

- Elhage, N., et al. (2021). "A Mathematical Framework for Transformer Circuits."
- Nanda, N. (2022). "A Comprehensive Mechanistic Interpretability Explainer."
- He, K., et al. (2016). "Deep Residual Learning for Image Recognition."

# Chapter 8 — Attention Heads

## Motivation

Attention heads are the primary information-routing mechanism in transformers. Each head can be decomposed into two distinct computations: selecting where to attend (QK) and determining what to write (OV). Understanding this decomposition is essential for reverse engineering transformer circuits.

## Learning Objectives

- Decompose attention heads into QK and OV circuits
- Analyze attention patterns as bilinear forms
- Compute the effective rank of QK and OV matrices
- Design interventions targeting specific QK or OV behavior
- Understand how QK and OV interact to produce head behavior

## Formal Definition: QK and OV Decomposition

For a single attention head at layer $\ell$:

### QK Circuit (Information Selection)

The attention score between destination position $j$ and source position $i$ is:

$$S_{ji} = \frac{q_j^\top k_i}{\sqrt{d_h}} = \frac{x_j^\top W_Q W_K^\top x_i}{\sqrt{d_h}}$$

Define $B_{QK} = W_Q W_K^\top \in \mathbb{R}^{d \times d}$. Then:

$$S_{ji} = \frac{x_j^\top B_{QK} x_i}{\sqrt{d_h}}$$

$B_{QK}$ defines a **bilinear form** that determines which pairs of tokens attend to each other. It encodes the head's "preference" for certain token relationships.

### OV Circuit (Information Movement)

The output at position $j$ is:

$$o_j = \sum_i A_{ji} W_O^\top W_V^\top x_i$$

Define $W_{OV} = W_V W_O \in \mathbb{R}^{d_h \times d}$ (or $W_V W_O^\top$ depending on convention). The head output is:

$$Z = A V W_O = A X W_V W_O = A X W_{OV}$$

### Mechanistic Decomposition

- **QK**: Determines *which* source tokens each destination token attends to
- **OV**: Determines *what information* is copied from source to destination

This decomposition is powerful because it separates the "where" from the "what" of attention. A head may have a simple QK circuit (attend to previous token) and a simple OV circuit (copy embedding), but their interaction produces complex behavior.

### Effective Rank

The effective rank of $B_{QK}$ and $W_{OV}$ indicates how "focused" the head is:
- Low effective rank $\approx$ 1: The head implements a simple, interpretable function
- High effective rank $\approx d$: The head implements a complex, distributed function

## Implementation: QK and OV Analysis

```python
import torch
import torch.linalg as LA

def decompose_head_weights(W_q, W_k, W_v, W_o):
    """Decompose head weights into QK and OV matrices.

    Args:
        W_q: [d_model, d_head]
        W_k: [d_model, d_head]
        W_v: [d_model, d_head]
        W_o: [d_head, d_model]

    Returns:
        dict with B_qk, W_ov, and their SVD analyses
    """
    # QK bilinear form
    B_qk = W_q @ W_k.T  # [d_model, d_model]

    # OV circuit
    W_ov = W_v @ W_o  # [d_model, d_model]

    # SVD analysis
    U_qk, S_qk, Vh_qk = LA.svd(B_qk)
    U_ov, S_ov, Vh_ov = LA.svd(W_ov)

    return {
        "B_qk": B_qk,
        "W_ov": W_ov,
        "qk_singular_values": S_qk,
        "ov_singular_values": S_ov,
        "qk_effective_rank": compute_effective_rank(S_qk),
        "ov_effective_rank": compute_effective_rank(S_ov),
        "qk_top_directions": U_qk[:, :5],
        "ov_top_directions": U_ov[:, :5]
    }


def compute_effective_rank(singular_values):
    """Compute effective rank from singular values."""
    p = singular_values / singular_values.sum()
    return torch.exp(-(p * torch.log(p + 1e-10)).sum()).item()


def compute_attention_pattern(x, W_q, W_k, mask=None):
    """Compute attention pattern for input x.

    Args:
        x: [batch, seq, d_model]
        W_q, W_k: [d_model, d_head]
        mask: [seq, seq], optional

    Returns:
        attention_weights: [batch, seq, seq]
    """
    Q = x @ W_q  # [batch, seq, d_head]
    K = x @ W_k  # [batch, seq, d_head]

    scores = (Q @ K.transpose(-1, -2)) / (Q.shape[-1] ** 0.5)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    weights = torch.softmax(scores, dim=-1)
    return weights


def analyze_ov_effect(x, attention_weights, W_v, W_o, target_direction):
    """Measure how much the OV circuit writes into target_direction.

    Args:
        x: [batch, seq, d_model]
        attention_weights: [batch, seq, seq]
        target_direction: [d_model], unit vector

    Returns:
        torch.Tensor [batch, seq], projection of head output onto target_direction
    """
    V = x @ W_v  # [batch, seq, d_head]
    head_output = attention_weights @ V @ W_o  # [batch, seq, d_model]

    projection = head_output @ target_direction  # [batch, seq]
    return projection
```

## Measurement: Head Specialization Metrics

1. **QK entropy**: $H(A_i) = -\sum_j A_{ij} \log A_{ij}$ averaged over positions
2. **OV rank**: Effective rank of $W_{OV}$
3. **Copying score**: For sequence $[A][B]...[A]$, measure attention from last $[A]$ to $[B]$
4. **Positional bias**: $\sum_i A_{i, i-1}$ (attention to previous token)
5. **Diagonal mass**: $\sum_i A_{ii}$ (attention to current token)

## Intervention: QK-Only and OV-Only Patching

```python
def patch_qk_only(model, layer_idx, head_idx, clean_inputs, corrupted_inputs):
    """Patch only the QK circuit (attention pattern) from clean to corrupted.

    Strategy: Use clean attention pattern with corrupted OV values.
    """
    # Compute clean attention pattern
    with torch.no_grad():
        clean_pattern = extract_attention_pattern(model, layer_idx, head_idx, clean_inputs)

    def hook_fn(module, input, output):
        x = input[0]  # [batch, seq, d]
        V = x @ module.W_v  # [batch, seq, d_head]
        patched_output = clean_pattern @ V @ module.W_o
        return patched_output

    target = model.blocks[layer_idx].attn
    handle = target.register_forward_hook(hook_fn)

    with torch.no_grad():
        result = model(corrupted_inputs)

    handle.remove()
    return result
```

## Falsification

A claimed QK or OV function is falsified if:
- The QK matrix does not have the structure predicted by the hypothesis
- The OV matrix does not project onto the claimed target directions
- Patching only QK or only OV does not produce the predicted behavior change
- The effective rank contradicts the claimed simplicity of the function

## Exercises

### Mathematical
1. Show that $B_{QK}$ is not necessarily symmetric. What does asymmetry imply about attention patterns?
2. Prove that the operator norm $\|W_{OV}\|_2$ bounds the maximum change in output norm per unit change in input norm.
3. Derive the gradient of the attention output with respect to $W_Q$. How does it depend on $B_{QK}$?

### Implementation
4. Implement a function that computes the "copying score" for each head in a transformer on a set of induction prompts.
5. Write a visualization function that plots QK and OV singular value spectra side by side for all heads.

### Experimental
6. For a 2-layer transformer, measure the effective rank of $W_{OV}$ for each head. Compare with human-labeled head types.

### Research
7. Investigate whether heads with low-rank $W_{OV}$ implement simpler, more interpretable functions than heads with full-rank $W_{OV}$.

## References

- Elhage, N., et al. (2021). "A Mathematical Framework for Transformer Circuits."
- Olsson, C., et al. (2022). "In-context Learning and Induction Heads."

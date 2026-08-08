# Chapter 8 — Attention Heads

## Motivation
Attention heads route information. Each head decomposes into QK (selection) and OV (movement).

## QK and OV Decomposition

**QK circuit (selection):**
$$S_{ji} = \frac{q_j^\top k_i}{\sqrt{d_h}} = \frac{x_j^\top W_Q W_K^\top x_i}{\sqrt{d_h}}$$

Define $B_{QK} = W_Q W_K^\top$. Then $S_{ji} = \frac{x_j^\top B_{QK} x_i}{\sqrt{d_h}}$.

**OV circuit (movement):**
$$Z = AVW_O = AXW_VW_O = AXW_{OV}$$

where $W_{OV} = W_V W_O$.

## Implementation

```python
import torch
import torch.linalg as LA

def decompose_head_weights(W_q, W_k, W_v, W_o):
    B_qk = W_q @ W_k.T
    W_ov = W_v @ W_o
    U_qk, S_qk, Vh_qk = LA.svd(B_qk)
    U_ov, S_ov, Vh_ov = LA.svd(W_ov)
    return {
        "B_qk": B_qk,
        "W_ov": W_ov,
        "qk_singular_values": S_qk,
        "ov_singular_values": S_ov,
        "qk_effective_rank": compute_effective_rank(S_qk),
        "ov_effective_rank": compute_effective_rank(S_ov)
    }

def compute_effective_rank(singular_values):
    p = singular_values / singular_values.sum()
    return torch.exp(-(p * torch.log(p + 1e-10)).sum()).item()

def compute_attention_pattern(x, W_q, W_k, mask=None):
    Q = x @ W_q
    K = x @ W_k
    scores = (Q @ K.transpose(-1, -2)) / (Q.shape[-1] ** 0.5)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    return torch.softmax(scores, dim=-1)

def analyze_ov_effect(x, attention_weights, W_v, W_o, target_direction):
    V = x @ W_v
    head_output = attention_weights @ V @ W_o
    return head_output @ target_direction
```

## Measurement
1. QK entropy
2. OV rank
3. Copying score
4. Positional bias

## Intervention
```python
def patch_qk_only(model, layer_idx, head_idx, clean_inputs, corrupted_inputs):
    clean_pattern = extract_attention_pattern(model, layer_idx, head_idx, clean_inputs)
    def hook_fn(module, input, output):
        x = input[0]
        V = x @ module.W_v
        return clean_pattern @ V @ module.W_o
    target = model.blocks[layer_idx].attn
    handle = target.register_forward_hook(hook_fn)
    with torch.no_grad():
        result = model(corrupted_inputs)
    handle.remove()
    return result
```

## Falsification
Falsified if QK matrix lacks predicted structure, or if OV does not project onto claimed directions.

## Exercises
- **Mathematical**: Show $B_{QK}$ need not be symmetric. Bound $\|W_{OV}\|_2$.
- **Implementation**: Compute copying score for each head. Plot QK/OV spectra.
- **Experimental**: Measure effective rank of $W_{OV}$ for all heads.
- **Research**: Do low-rank $W_{OV}$ heads implement simpler functions?

## References
- Elhage, N., et al. (2021). "A Mathematical Framework for Transformer Circuits."
- Olsson, C., et al. (2022). "In-context Learning and Induction Heads."

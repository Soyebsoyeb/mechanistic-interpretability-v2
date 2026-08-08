# Chapter 7 — The Residual Stream

## Motivation
The residual stream is the central communication channel. Every layer reads from it and writes to it.

## Formal Definition

For layer $\ell$:
$$x_{\ell+1} = x_\ell + a_\ell + m_\ell$$

**Unrolling:**
$$x_L = x_0 + \sum_{\ell=0}^{L-1} a_\ell + \sum_{\ell=0}^{L-1} m_\ell$$

This decomposition is exact.

**Directional analysis:**
$$v^\top x_L = v^\top x_0 + \sum_\ell v^\top a_\ell + \sum_\ell v^\top m_\ell$$

## Implementation

```python
def decompose_residual_stream(model, inputs):
    cache = {}
    def make_hook(name):
        def hook(module, input, output):
            cache[name] = output.detach()
        return hook
    handles = []
    for i, block in enumerate(model.blocks):
        h1 = block.attn.register_forward_hook(make_hook(f"attn_{i}"))
        h2 = block.mlp.register_forward_hook(make_hook(f"mlp_{i}"))
        handles.extend([h1, h2])
    with torch.no_grad():
        cache["embedding"] = model.embed(inputs).detach()
        _ = model(inputs)
    for h in handles:
        h.remove()
    L = len(model.blocks)
    attn_contribs = [cache[f"attn_{i}"] for i in range(L)]
    mlp_contribs = [cache[f"mlp_{i}"] for i in range(L)]
    reconstructed = cache["embedding"]
    for i in range(L):
        reconstructed = reconstructed + attn_contribs[i] + mlp_contribs[i]
    return {
        "embedding": cache["embedding"],
        "attn_contributions": attn_contribs,
        "mlp_contributions": mlp_contribs,
        "reconstructed": reconstructed
    }

def analyze_direction_contributions(decomposition, direction, token_idx=0):
    direction = direction.to(decomposition["embedding"].device)
    emb_contrib = (decomposition["embedding"][:, token_idx] @ direction).item()
    attn_contribs = [(a[:, token_idx] @ direction).item() for a in decomposition["attn_contributions"]]
    mlp_contribs = [(m[:, token_idx] @ direction).item() for m in decomposition["mlp_contributions"]]
    return {
        "embedding": emb_contrib,
        "attention": attn_contribs,
        "mlp": mlp_contribs,
        "total": emb_contrib + sum(attn_contribs) + sum(mlp_contribs)
    }
```

## Measurement
$$\text{Contribution}(\ell, c, v) = \frac{|v^\top h_\ell^{(c)}|}{\sum_{\ell', c'} |v^\top h_{\ell'}^{(c')}|}$$

## Intervention
```python
def patch_residual_component(model, inputs, layer_idx, component_type, clean_cache):
    def hook_fn(module, input, output):
        return clean_cache[f"{component_type}_{layer_idx}"]
    target = model.blocks[layer_idx].attn if component_type == 'attn' else model.blocks[layer_idx].mlp
    handle = target.register_forward_hook(hook_fn)
    with torch.no_grad():
        output = model(inputs)
    handle.remove()
    return output
```

## Falsification
Falsified if decomposition does not reconstruct final state, or if component has $|v^\top h| \approx 0$.

## Exercises
- **Mathematical**: Prove $x_L = x_0 + \sum (a_\ell + m_\ell)$ by induction.
- **Implementation**: Build hook system extracting every intermediate tensor.
- **Experimental**: Compute decomposition for 100 inputs. Plot layer-wise contributions.
- **Research**: Do early layers write low-level features, late layers high-level?

## References
- Elhage, N., et al. (2021). "A Mathematical Framework for Transformer Circuits."
- Nanda, N. (2022). "A Comprehensive Mechanistic Interpretability Explainer."

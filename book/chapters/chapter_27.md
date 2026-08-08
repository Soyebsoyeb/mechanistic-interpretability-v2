# Chapter 27 — Arithmetic and Modular Computation

## Motivation

Synthetic arithmetic tasks have known ground-truth algorithms, making them ideal for mechanistic study. Modular arithmetic is particularly interesting because it requires the model to learn periodic representations.

## Setup

Train transformers on $(a + b) \mod p$.

## Analysis Targets
- Embeddings: periodic structure
- Attention: which positions attend to which
- MLPs: Fourier-like representations
- Output decoding

## Implementation

```python
def generate_modular_data(p=113, n_samples=10000):
    a = torch.randint(0, p, (n_samples,))
    b = torch.randint(0, p, (n_samples,))
    c = (a + b) % p
    return torch.stack([a, b], dim=1), c

def analyze_periodic_structure(embedding_matrix, p):
    fft = torch.fft.fft(embedding_matrix, dim=0)
    magnitudes = torch.abs(fft)
    top_freqs = torch.topk(magnitudes.mean(dim=1), 5)
    return top_freqs
```

## Falsification
Falsified if model does not use Fourier-like representations when predicted.

## Exercises
- **Mathematical**: Prove modular addition can be computed via trigonometric identities.
- **Implementation**: Implement Fourier analysis of embeddings.
- **Experimental**: Train and reverse-engineer modular addition transformer.
- **Research**: Do larger $p$ require different circuit structures?

## References

- Nanda, N., et al. (2023). "Progress Measures for Grokking via Mechanistic Interpretability."

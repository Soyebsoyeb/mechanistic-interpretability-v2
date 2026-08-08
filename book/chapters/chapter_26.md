# Chapter 26 — Indirect Object Identification

## Motivation

IOI is a well-studied circuit where the model identifies the indirect object in sentences like "John gave Mary a book. She..." It serves as a benchmark for circuit discovery methods.

## Procedure
1. Construct prompts with competing candidates
2. Measure output probabilities
3. Identify candidate heads
4. Compute attribution
5. Patch candidate activations
6. Test circuit edges
7. Reconstruct algorithm
8. Test on held-out templates

## Implementation

```python
ioi_templates = [
    "[A] gave [B] a book. [Pronoun]",
    "[A] sent [B] a letter. [Pronoun]",
    "[A] told [B] a story. [Pronoun]"
]

def measure_ioi_probability(model, template, names, pronoun_pos):
    prompt = template.replace("[A]", names[0]).replace("[B]", names[1])
    tokens = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        logits = model(**tokens).logits
    pronoun_logits = logits[0, -1]
    correct_prob = torch.softmax(pronoun_logits, dim=-1)[pronoun_pos]
    return correct_prob.item()
```

## Falsification
Falsified if circuit does not generalize to new names or templates.

## Exercises
- **Mathematical**: Write information flow graph for IOI.
- **Implementation**: Implement full IOI circuit reconstruction.
- **Experimental**: Validate IOI circuit in Pythia models.
- **Research**: How does IOI circuit scale with model size?

## References

- Wang, K., et al. (2022). "Interpretability in the Wild: A Circuit for Indirect Object Identification."

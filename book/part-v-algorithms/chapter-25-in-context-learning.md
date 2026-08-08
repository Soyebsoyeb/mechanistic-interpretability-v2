# Chapter 25 — In-Context Learning

## Motivation
Transformers learn from examples in context. Understanding the mechanism requires controlled synthetic tasks.

## Approach
Use tasks with known rules: $y = ax + b$, classification mappings.

## Questions
- Which components retrieve examples?
- Where are labels represented?
- How is the inferred rule stored?
- Which components update predictions?

## Implementation

```python
def generate_icl_prompts(rule_fn, n_examples=5, n_test=100):
    prompts = []
    for _ in range(n_test):
        examples = [(torch.randn(1), rule_fn(torch.randn(1))) for _ in range(n_examples)]
        test_x = torch.randn(1)
        prompt = format_examples(examples) + format_test(test_x)
        prompts.append((prompt, rule_fn(test_x)))
    return prompts
```

## Falsification
Falsified if claimed example-retrieval heads do not attend to examples.

## Exercises
- **Mathematical**: Formalize ICL as Bayesian inference.
- **Implementation**: Build ICL task with known linear rule.
- **Experimental**: Identify example-retrieval heads.
- **Research**: Do ICL mechanisms differ between pretraining and fine-tuning?

## References
- Xie, S. M., et al. (2021). "An Explanation of In-context Learning as Implicit Bayesian Inference."

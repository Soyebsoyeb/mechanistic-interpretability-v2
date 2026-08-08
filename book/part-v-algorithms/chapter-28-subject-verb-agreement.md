# Chapter 28 — Subject-Verb Agreement

## Motivation
Subject-verb agreement tests long-distance syntactic dependencies.

## Setup
Examples: "The key to the cabinets is/are..."

Vary: subject number, distractor number, distance, syntactic structure.

## Implementation

```python
sv_templates = [
    "The [subject] [prep] the [distractor] [verb]",
    "The [subject] that the [distractor] saw [verb]"
]

def measure_agreement_accuracy(model, templates, subjects, distractors, verbs):
    correct = 0
    for template in templates:
        for subj in subjects:
            for dist in distractors:
                for verb in verbs:
                    prompt = template.replace("[subject]", subj).replace("[distractor]", dist)
                    # Check verb prediction
                    pass
    return correct / total
```

## Falsification
Falsified if claimed number-representation heads do not affect agreement.

## Exercises
- **Mathematical**: Formalize agreement as feature propagation.
- **Implementation**: Build controlled agreement dataset.
- **Experimental**: Identify number-tracking circuit.
- **Research**: How does agreement circuit handle nested dependencies?

## References
- Linzen, T., et al. (2016). "Assessing the Ability of LSTMs to Learn Syntax-Sensitive Dependencies."

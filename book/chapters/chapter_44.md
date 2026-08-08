# Chapter 44 — Attribution Frameworks

## Generic Interface
```python
score = attribution(
    model=model,
    input=tokens,
    target=target,
    components=components
)
```

## Methods
- Gradient attribution
- Activation attribution
- Integrated gradients
- Direct logit attribution
- Patching-based attribution

**Always compare methods.**

## Implementation
```python
class AttributionFramework:
    def __init__(self, model, method='gradient'):
        self.model = model
        self.method = method

    def compute(self, inputs, target, components):
        if self.method == 'gradient':
            return self._gradient_attribution(inputs, target, components)
        elif self.method == 'activation':
            return self._activation_attribution(inputs, target, components)

    def _gradient_attribution(self, inputs, target, components):
        self.model.zero_grad()
        output = self.model(inputs)
        loss = compute_target_loss(output, target)
        loss.backward()
        scores = {}
        for name in components:
            module = dict(self.model.named_modules())[name]
            if hasattr(module, 'weight') and module.weight.grad is not None:
                scores[name] = module.weight.grad.norm().item()
        return scores
```

## Exercises
- **Implementation**: Compare all attribution methods on a task.

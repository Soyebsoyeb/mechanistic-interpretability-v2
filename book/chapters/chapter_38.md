# Chapter 38 — RLHF and Preference Optimization

## Motivation
Alignment changes model behavior. Understanding these changes mechanistically is critical for safety.

## Comparison
Compare $\theta_{\text{pretrain}}$ vs. $\theta_{\text{aligned}}$.

## Questions
- Which features change?
- Which circuits are preserved?
- Which representations emerge?
- Which behavioral changes are localized?

## Implementation
```python
def compare_models(pretrained, aligned, test_inputs):
    pre_acts = extract_activations(pretrained, test_inputs)
    post_acts = extract_activations(aligned, test_inputs)
    feature_changes = (pre_acts - post_acts).norm(dim=-1).mean()
    diff = pre_acts.mean(dim=0) - post_acts.mean(dim=0)
    top_changed = torch.topk(diff.norm(dim=-1), 10)
    return feature_changes, top_changed
```

## Exercises
- **Experimental**: Identify features that emerge after RLHF.
- **Research**: Are safety-relevant features localized or distributed?

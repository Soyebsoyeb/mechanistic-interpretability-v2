import torch

def activation_patch(model, clean_inputs, corrupted_inputs, component_name, metric_fn):
    clean_cache = {}
    def clean_hook(module, inputs, output):
        clean_cache['out'] = output.detach()
    target = dict(model.named_modules())[component_name]
    handle = target.register_forward_hook(clean_hook)
    with torch.no_grad():
        clean_metric = metric_fn(model(clean_inputs))
    handle.remove()

    with torch.no_grad():
        corrupted_metric = metric_fn(model(corrupted_inputs))

    def patch_hook(module, inputs, output):
        return clean_cache['out']
    handle = target.register_forward_hook(patch_hook)
    with torch.no_grad():
        patched_metric = metric_fn(model(corrupted_inputs))
    handle.remove()

    score = (patched_metric - corrupted_metric) / (clean_metric - corrupted_metric + 1e-10)
    return {
        "clean": clean_metric.item(),
        "corrupted": corrupted_metric.item(),
        "patched": patched_metric.item(),
        "score": score.item()
    }

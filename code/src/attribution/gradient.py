import torch

def gradient_attribution(model, inputs, target, components):
    model.zero_grad()
    output = model(inputs)
    loss = torch.nn.functional.cross_entropy(output.view(-1, output.size(-1)), target.view(-1))
    loss.backward()
    scores = {}
    for name in components:
        module = dict(model.named_modules())[name]
        if hasattr(module, 'weight') and module.weight.grad is not None:
            scores[name] = module.weight.grad.norm().item()
    return scores

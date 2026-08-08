# Chapter 31 — Large Language Models

## Practical Challenges
- Activation extraction at scale
- Storage and memory management
- GPU batching and model sharding
- Tokenizer versioning
- Reproducibility and dataset licensing

## Architecture
Separate: model loading, activation extraction, analysis, visualization.

## Implementation
```python
class LLMInterpreter:
    def __init__(self, model_name, device="cuda"):
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def extract_activations(self, texts, layers=None, batch_size=8):
        if layers is None:
            layers = range(self.model.config.n_layer)
        all_activations = {l: [] for l in layers}
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            tokens = self.tokenizer(batch, return_tensors="pt", padding=True).to(self.model.device)
            with torch.no_grad():
                outputs = self.model(**tokens, output_hidden_states=True)
            for l in layers:
                all_activations[l].append(outputs.hidden_states[l].cpu())
        return {l: torch.cat(v, dim=0) for l, v in all_activations.items()}
```

## Exercises
- **Implementation**: Build memory-efficient activation streaming.
- **Experimental**: Extract and analyze activations from Llama-2-7B.
- **Research**: How do feature distributions change with scale?

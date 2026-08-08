# Appendix H — Reproducibility Checklist

## Before Running
- [ ] Model identifier recorded
- [ ] Model revision (commit hash or version) recorded
- [ ] Tokenizer name and revision recorded
- [ ] Dataset source and license recorded
- [ ] Preprocessing pipeline documented
- [ ] Code version (git commit hash) recorded
- [ ] Dependencies pinned in requirements.txt

## During Running
- [ ] Random seeds recorded (PyTorch, NumPy, Python)
- [ ] Hardware (GPU model, CUDA version) recorded
- [ ] Experiment configuration saved as JSON/YAML

## After Running
- [ ] Raw results saved (not just figures)
- [ ] Figures generated programmatically from raw results
- [ ] Statistical uncertainty reported (confidence intervals, standard errors)
- [ ] Effect sizes reported
- [ ] Negative results and failure cases documented
- [ ] Limitations explicitly stated
- [ ] Code committed with descriptive message

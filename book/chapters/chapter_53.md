# Chapter 53 — Build an Automated Interpretability Agent

## Architecture
Agent loop:
1. Choose experiment
2. Extract activations
3. Identify candidate features
4. Generate hypotheses
5. Propose interventions
6. Execute tests
7. Update hypothesis
8. Produce research report

Must distinguish: OBSERVATION, HYPOTHESIS, EVIDENCE, INTERPRETATION, UNCERTAINTY.

## Template
```python
class AutomatedInterpreterAgent:
    def __init__(self, model, sae=None):
        self.model = model
        self.sae = sae
        self.state = "observing"

    def run(self, task_description, max_iterations=10):
        for i in range(max_iterations):
            if self.state == "observing":
                obs = self.observe(task_description)
                self.state = "hypothesizing"
            elif self.state == "hypothesizing":
                hypothesis = self.generate_hypothesis(obs)
                self.state = "testing"
            elif self.state == "testing":
                evidence = self.test_hypothesis(hypothesis)
                if evidence["supports"]:
                    self.state = "reporting"
                else:
                    self.state = "hypothesizing"
            elif self.state == "reporting":
                return self.generate_report()
```

## Deliverable
Research report with clear uncertainty quantification.

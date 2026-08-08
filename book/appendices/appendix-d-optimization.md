# Appendix D — Optimization

- Gradient: $\nabla_\theta L$
- Gradient descent: $\theta' = \theta - \eta \nabla_\theta L$
- Second-order: $L(\theta+\Delta) \approx L(\theta) + \nabla L^\top \Delta + \frac{1}{2} \Delta^\top H \Delta$
- Momentum: $v_{t+1} = \beta v_t + \nabla_\theta L$, $\theta_{t+1} = \theta_t - \eta v_{t+1}$
- Adam: adaptive learning rates per parameter

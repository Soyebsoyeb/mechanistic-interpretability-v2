# Appendix D — Optimization

## Gradient
$\nabla_\theta L$ is the vector of partial derivatives.

## Gradient Descent
$\theta' = \theta - \eta \nabla_\theta L$.

## Stochastic Gradient Descent
$\theta_{t+1} = \theta_t - \eta_t \nabla_\theta L_i(\theta_t)$.

## Momentum
$v_{t+1} = \beta v_t + \nabla_\theta L(\theta_t)$, $\theta_{t+1} = \theta_t - \eta v_{t+1}$.

## Adam
$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$
$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$
$\hat{m}_t = m_t/(1-\beta_1^t)$, $\hat{v}_t = v_t/(1-\beta_2^t)$
$\theta_{t+1} = \theta_t - \eta \hat{m}_t/(\sqrt{\hat{v}_t} + \epsilon)$

## Second-Order Approximation
$L(\theta+\Delta) \approx L(\theta) + \nabla L^\top \Delta + \frac{1}{2} \Delta^\top H \Delta$
where $H$ is the Hessian.

## Convexity
$L$ is convex if $L(\lambda x + (1-\lambda)y) \leq \lambda L(x) + (1-\lambda)L(y)$.

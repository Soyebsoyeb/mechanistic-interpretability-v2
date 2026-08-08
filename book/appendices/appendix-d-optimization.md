# Appendix D — Optimization for Mechanistic Interpretability

## D.1 Introduction

Training a neural network is an optimization problem: find parameters $\theta$ minimizing a loss $L(\theta)$. This appendix treats gradient-based optimization rigorously enough to explain *why* the standard tricks (momentum, Adam, learning-rate limits) work, rather than just stating the update rules. The central tool is the **Hessian** — the matrix of second derivatives — and everything in Appendix A about eigenvalues and eigenvectors (§A.5) applies to it directly: the Hessian's eigenvalues govern how fast gradient descent converges, and its eigenvectors give the "natural coordinates" in which the optimization problem decouples into independent 1-D problems.

If you can answer these questions, you're ready:
- Why does gradient descent need a learning rate smaller than $2/\lambda_{\max}$ of the Hessian, and where does that number come from?
- Why does an *ill-conditioned* loss landscape make plain gradient descent slow, and why does momentum help?
- Why does Adam divide by $(1-\beta_2^t)$, and not just use $\sqrt{v_t}$ directly?
- Why is a saddle point, not a local maximum, the generic obstacle in high-dimensional loss landscapes?

### D.1.1 Conventions

$\theta \in \mathbb{R}^n$ denotes the full parameter vector. $\nabla_\theta L$ (or just $\nabla L$ when $\theta$ is clear from context) is the **gradient**, a column vector in $\mathbb{R}^n$ with $(\nabla L)_i = \partial L/\partial \theta_i$. $H = \nabla^2 L \in \mathbb{R}^{n\times n}$ is the **Hessian**, with $H_{ij} = \partial^2 L/\partial\theta_i\partial\theta_j$. Whenever $L$ is twice continuously differentiable, mixed partials commute (Schwarz's / Clairaut's theorem), so $H$ is always **symmetric** — meaning every fact from Appendix A §A.5.4 (real eigenvalues, orthonormal eigenbasis $H = Q\Lambda Q^\top$) applies to it unconditionally, not as a special case.

---

## D.2 Gradients

### D.2.1 Definition

$$
\nabla_\theta L = \begin{bmatrix} \partial L/\partial \theta_1 \\ \vdots \\ \partial L/\partial \theta_n \end{bmatrix}
$$

The first-order Taylor approximation of $L$ near $\theta$ is:

$$
L(\theta + \Delta) \approx L(\theta) + \nabla L^\top \Delta
$$

### D.2.2 The Gradient Points in the Direction of Steepest Ascent

**Claim**: among all unit vectors $u$, the directional derivative $\nabla L^\top u$ is maximized by $u = \nabla L / \|\nabla L\|$.

**Proof**: by the Cauchy–Schwarz form of the dot product (Appendix A §A.2.2), $\nabla L^\top u = \|\nabla L\|\,\|u\|\cos\theta = \|\nabla L\|\cos\theta$ for unit $u$, which is maximized exactly when $\cos\theta = 1$, i.e. $u$ points in the same direction as $\nabla L$. $\blacksquare$

This is *why* gradient descent moves in the direction $-\nabla L$: it is, to first order, the direction of steepest local *decrease*.

> **MI connection**: This is the same dot-product/cosine machinery from Appendix A §A.2.2 that governs attention scores and feature similarity — "steepest ascent direction" and "most similar direction" are the same geometric fact (maximizing a dot product subject to a fixed norm) applied to two different quantities.

### D.2.3 Backpropagation

Backpropagation is the chain rule (calculus, not the probability chain rule of Appendix B §B.3.1) applied systematically: for a composition $L = f_k \circ f_{k-1} \circ \cdots \circ f_1$, the gradient with respect to an intermediate layer's parameters is a product of Jacobians propagated backward from the loss. We do not re-derive this here; the point relevant to this appendix is only that $\nabla_\theta L$ is computable exactly and efficiently for the compositions neural networks are built from, which is what makes everything below practical rather than merely theoretical.

---

## D.3 Gradient Descent

### D.3.1 Update Rule

$$
\theta_{t+1} = \theta_t - \eta\, \nabla_\theta L(\theta_t)
$$

where $\eta > 0$ is the **learning rate** (step size).

### D.3.2 Convergence Analysis on a Quadratic

To understand *why* this converges (and how fast), analyze the idealized case of a quadratic loss centered at the minimum $\theta^\star = 0$:

$$
L(\theta) = \tfrac{1}{2}\theta^\top H \theta, \qquad \nabla L(\theta) = H\theta
$$

with $H$ symmetric positive definite (all eigenvalues $\lambda_i > 0$ — this makes $\theta=0$ the unique global minimum; see §D.4.3). Substituting into the update rule:

$$
\theta_{t+1} = \theta_t - \eta H \theta_t = (I - \eta H)\theta_t
$$

Diagonalize $H = Q\Lambda Q^\top$ (Appendix A §A.5.4) and change coordinates to $\phi_t = Q^\top \theta_t$ (i.e. work in the eigenbasis of $H$). Since $Q^\top(I-\eta H)Q = I - \eta\Lambda$ is diagonal, the update **decouples into $n$ independent 1-D problems**, one per eigenvalue:

$$
\phi_{t+1}^{(i)} = (1 - \eta\lambda_i)\,\phi_t^{(i)} \quad \implies \quad \phi_t^{(i)} = (1-\eta\lambda_i)^t\, \phi_0^{(i)}
$$

This single line explains most of the qualitative behavior of gradient descent:

- **Convergence requires** $|1 - \eta\lambda_i| < 1$ for every $i$, i.e. $0 < \eta < 2/\lambda_i$ for every eigenvalue. Since this must hold for the *largest* eigenvalue, the binding constraint is
  $$
  \eta < \frac{2}{\lambda_{\max}}
  $$
  — exceeding this causes the corresponding coordinate to diverge, oscillating with growing amplitude.
- **Convergence speed is set by the slowest-decaying coordinate**, i.e. the eigenvalue $\lambda_i$ for which $|1-\eta\lambda_i|$ is closest to 1. For a fixed $\eta$, small eigenvalues (flat, "sloppy" directions of the loss) shrink slowly; large eigenvalues (steep, "stiff" directions) shrink quickly. The overall convergence rate is governed by whichever is worse — usually the smallest eigenvalue, once $\eta$ is chosen near its stability limit for the largest one.
- **The condition number** $\kappa = \lambda_{\max}/\lambda_{\min}$ controls how bad this tension is. The learning rate that minimizes the *worst-case* per-step contraction factor $\max_i |1-\eta\lambda_i|$ is
  $$
  \eta^\star = \frac{2}{\lambda_{\max}+\lambda_{\min}}, \qquad \text{giving worst-case rate} \quad \rho = \frac{\lambda_{\max}-\lambda_{\min}}{\lambda_{\max}+\lambda_{\min}} = \frac{\kappa - 1}{\kappa+1}
  $$
  As $\kappa \to \infty$ (very ill-conditioned loss), $\rho \to 1$: convergence stalls, because no single $\eta$ can be both large enough to move quickly along flat directions and small enough to remain stable along steep ones.

### D.3.3 Worked Example

Reuse the matrix from Appendix A §A.5.6, $H = \begin{bmatrix}4 & 1\\ 1 & 4\end{bmatrix}$, with eigenvalues $\lambda = 3, 5$ found there. The stability limit is $\eta < 2/5 = 0.4$. The optimal fixed learning rate is

$$
\eta^\star = \frac{2}{5+3} = 0.25, \qquad \rho = \frac{5-3}{5+3} = 0.25
$$

so under optimal tuning, the distance to the optimum shrinks by a factor of $0.25$ every iteration along the worst-case direction — this is a mild condition number ($\kappa = 5/3 \approx 1.67$), and we will see in §D.5.3 that momentum improves substantially even on a problem this well-conditioned.

> **MI connection**: In a real network, $H$ is the loss Hessian evaluated at the current parameters, with dimension equal to the parameter count — far too large to diagonalize explicitly. But the qualitative lesson transfers directly: loss landscapes with very different curvature across directions (a large ratio between the largest and smallest relevant Hessian eigenvalues) are exactly the settings where plain SGD is slow and where preconditioning methods (§D.5–D.6) earn their keep. Empirically measured Hessian eigenvalue *spectra* of trained networks (via Lanczos-type methods) typically show a small number of very large eigenvalues and a long tail of small ones — a highly ill-conditioned landscape by this analysis — which is one standing empirical motivation for adaptive and momentum-based optimizers over plain gradient descent.

---

## D.4 Second-Order Methods and the Hessian

### D.4.1 Second-Order Taylor Expansion

$$
L(\theta + \Delta) \approx L(\theta) + \nabla L(\theta)^\top \Delta + \tfrac{1}{2}\Delta^\top H(\theta) \Delta
$$

where $\nabla L(\theta)$ and $H(\theta)$ are evaluated *at the current point* $\theta$ (not at $\theta+\Delta$), and the approximation has error $O(\|\Delta\|^3)$ for $L$ three-times differentiable. Because $H$ is symmetric (§D.1.1), the quadratic term $\Delta^\top H \Delta$ is a genuine quadratic form and everything in §D.3.2 about eigendecomposing it applies locally around any point $\theta$, not just at a global quadratic loss.

### D.4.2 Newton's Method

Rather than taking a small step in the gradient direction, Newton's method minimizes the second-order approximation of D.4.1 *exactly*. Treating it as a function of $\Delta$ and setting its gradient to zero:

$$
\nabla_\Delta \left[\nabla L^\top \Delta + \tfrac12 \Delta^\top H \Delta\right] = \nabla L + H\Delta = 0 \quad \implies \quad \Delta^\star = -H^{-1}\nabla L
$$

giving the update

$$
\theta_{t+1} = \theta_t - H(\theta_t)^{-1}\, \nabla L(\theta_t)
$$

This is exactly gradient descent (§D.3.1) with a per-direction learning rate of $1/\lambda_i$ along each Hessian eigendirection — automatically choosing the *optimal* step size for every direction simultaneously, which is why Newton's method converges in a single step on an exactly quadratic loss (verify: substituting $H^{-1}\nabla L = H^{-1}H\theta = \theta$ into the update from a point $\theta$ gives $\theta_{t+1}=0=\theta^\star$ directly). The cost is that $H^{-1}$ requires forming and inverting an $n\times n$ matrix — intractable at neural-network parameter counts, which is why momentum (§D.5) and diagonal-adaptive methods (§D.6) are used as cheap approximations to second-order curvature information instead.

### D.4.3 Classifying Critical Points

At a critical point ($\nabla L = 0$), the second-order term alone determines local behavior. Since $H$ is symmetric with real eigenvalues $\lambda_1,\ldots,\lambda_n$ (Appendix A §A.5.4):

- **All $\lambda_i > 0$** ($H$ positive definite): $\Delta^\top H\Delta > 0$ for every $\Delta \ne 0$, so $L$ increases in every direction — a **local minimum**.
- **All $\lambda_i < 0$** ($H$ negative definite): a **local maximum**.
- **Mixed signs**: $L$ increases along positive-eigenvalue directions and decreases along negative-eigenvalue ones — a **saddle point**.

> **MI connection**: In a loss landscape with $n$ parameters, a critical point is a local minimum only if *every one* of $n$ Hessian eigenvalues happens to be positive. If eigenvalue signs were independently random, the probability of this drops exponentially in $n$ — which is the standard heuristic explanation for why saddle points, not local maxima or spurious local minima, are believed to be the generic obstacle to optimization in high-dimensional networks (empirically, trained-network Hessians at convergence tend to show many near-zero and a mix of positive/small-negative eigenvalues rather than strict positive-definiteness). This also underlies "flat vs. sharp minima" discussions in generalization: the *magnitude* of the positive eigenvalues at a minimum (how curved the basin is) is measured by exactly the Hessian spectrum this section classifies critical points with.

---

## D.5 Momentum

### D.5.1 Update Rule

$$
v_{t+1} = \beta v_t + \nabla_\theta L(\theta_t), \qquad \theta_{t+1} = \theta_t - \eta\, v_{t+1}
$$

with $v_0 = 0$ and $\beta \in [0,1)$ a momentum coefficient ($\beta=0$ recovers plain gradient descent). Unrolling the recursion, $v_{t+1}$ is an exponentially-weighted sum of *all* past gradients:

$$
v_{t+1} = \sum_{i=0}^{t} \beta^{i}\, \nabla_\theta L(\theta_{t-i})
$$

so the update direction is a smoothed running average of recent gradients rather than only the current one — the physical analogy is a heavy ball rolling downhill, which accumulates velocity and does not instantly reverse direction the way a memoryless (zero-mass) particle following $-\eta\nabla L$ alone would.

### D.5.2 Why Momentum Helps: Analysis on the Same Quadratic

Repeating the eigenbasis analysis of §D.3.2 on $L(\theta)=\tfrac12\theta^\top H\theta$ for the momentum update gives, per eigendirection $\lambda_i$, a linear recurrence

$$
\phi_{t+1}^{(i)} = (1+\beta)\phi_t^{(i)} - \beta\phi_{t-1}^{(i)} - \eta\lambda_i \phi_t^{(i)}
$$

whose characteristic roots (a standard but slightly involved calculation, stated here without full derivation) are minimized in magnitude, simultaneously across *all* eigenvalues $\lambda_i \in [\lambda_{\min}, \lambda_{\max}]$, by the **Polyak optimal parameters**:

$$
\eta^\star = \frac{4}{\left(\sqrt{\lambda_{\max}} + \sqrt{\lambda_{\min}}\right)^2}, \qquad \beta^\star = \left(\frac{\sqrt{\lambda_{\max}} - \sqrt{\lambda_{\min}}}{\sqrt{\lambda_{\max}} + \sqrt{\lambda_{\min}}}\right)^2
$$

giving worst-case per-step contraction $\rho_{\text{mom}} = \sqrt{\beta^\star} = \frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}$ — compare to plain gradient descent's $\rho_{\text{GD}} = \frac{\kappa-1}{\kappa+1}$ from §D.3.2. Since $\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1} < \frac{\kappa-1}{\kappa+1}$ for every $\kappa > 1$, momentum is *always* asymptotically faster under optimal tuning, and the gap widens as $\kappa$ grows: momentum's dependence on the condition number is $O(\sqrt{\kappa})$ where plain gradient descent's is $O(\kappa)$.

**Worked example (continuing §D.3.3)**: with $\lambda_{\max}=5,\ \lambda_{\min}=3$, $\sqrt{\lambda_{\max}}=2.236,\ \sqrt{\lambda_{\min}}=1.732$:

$$
\eta^\star = \frac{4}{(2.236+1.732)^2} = \frac{4}{15.75} \approx 0.254, \qquad \beta^\star = \left(\frac{0.504}{3.968}\right)^2 \approx 0.0161
$$

$$
\rho_{\text{mom}} = \sqrt{0.0161} \approx 0.127 \qquad \text{vs.} \qquad \rho_{\text{GD}} = 0.25 \text{ from §D.3.3}
$$

Momentum roughly doubles the convergence rate here even though the condition number ($\kappa\approx1.67$) is mild — the gap would be far larger for the highly ill-conditioned spectra typical of real loss landscapes (§D.3.3).

### D.5.3 Nesterov Momentum

A variant evaluates the gradient at a *look-ahead* point $\theta_t - \eta\beta v_t$ rather than at $\theta_t$ itself:

$$
v_{t+1} = \beta v_t + \nabla_\theta L(\theta_t - \eta\beta v_t), \qquad \theta_{t+1} = \theta_t - \eta v_{t+1}
$$

This does not change the asymptotic $O(\sqrt{\kappa})$ dependence of §D.5.2 on strongly convex quadratics, but improves the constant factor and, unlike heavy-ball momentum, provably accelerates the *general* smooth convex (not just quadratic) case — a distinction that does not matter for the local quadratic analysis above but does for global convergence guarantees.

---

## D.6 Adaptive Methods

### D.6.1 AdaGrad

Maintains a running *sum* of squared gradients per parameter and scales each coordinate's step inversely to it:

$$
G_t = G_{t-1} + g_t \odot g_t, \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{G_t}+\epsilon} \odot g_t
$$

where $g_t = \nabla_\theta L(\theta_t)$, $\odot$ is elementwise product, and $\epsilon$ prevents division by zero. Because $G_t$ only grows, the effective learning rate monotonically decays — which helps early on but can halt learning prematurely on long training runs.

### D.6.2 RMSProp

Replaces AdaGrad's running *sum* with an exponential moving average, so old gradients are eventually forgotten instead of permanently accumulated:

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)\, g_t \odot g_t, \qquad \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{v_t}+\epsilon}\odot g_t
$$

### D.6.3 Adam

Adam combines momentum (a first-moment EMA of the gradient) with RMSProp-style adaptive scaling (a second-moment EMA), plus a bias correction for both:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t, \qquad v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t \odot g_t
$$

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

$$
\theta_{t+1} = \theta_t - \eta\, \frac{\hat{m}_t}{\sqrt{\hat{v}_t}+\epsilon}
$$

with typical defaults $\beta_1 = 0.9,\ \beta_2 = 0.999,\ \epsilon = 10^{-8}$.

**Why the bias correction is needed**: unrolling $m_t = (1-\beta_1)\sum_{i=1}^t \beta_1^{t-i} g_i$ from $m_0=0$, and assuming (for the purpose of this calculation only) that the true gradient's expectation $E[g_i] = \mu$ is roughly constant over the averaging window:

$$
E[m_t] = (1-\beta_1)\sum_{i=1}^t \beta_1^{t-i}\, \mu = \mu\,(1-\beta_1)\sum_{i=0}^{t-1}\beta_1^{i} = \mu\,(1-\beta_1)\cdot\frac{1-\beta_1^t}{1-\beta_1} = \mu\,(1-\beta_1^t)
$$

using the finite geometric series identity. So $E[m_t] = \mu(1-\beta_1^t) \ne \mu$ — $m_t$ is a *biased* estimate of the true gradient, especially early in training when $t$ is small and $\beta_1^t$ is still far from zero (with $\beta_1=0.9$, $\beta_1^1=0.9$, so $m_1$ underestimates $\mu$ by a factor of 10 before correction). Dividing by $(1-\beta_1^t)$ exactly cancels this factor, giving $E[\hat{m}_t] = \mu$. The identical argument applies to $v_t$ and $\beta_2$.

> **MI connection**: The elementwise division by $\sqrt{\hat v_t}$ makes Adam's effective step size roughly proportional to the **signal-to-noise ratio** of each parameter's gradient (large, consistent gradients get a *relatively* smaller normalized step than small, noisy ones would otherwise suggest, since $\sqrt{\hat v_t}$ scales with the gradient's typical magnitude in that coordinate). This diagonal rescaling is a cheap surrogate for the whitening transform of Appendix B §B.6.3 ($\Sigma^{-1/2}$) — it only corrects per-coordinate scale (a diagonal approximation to the Hessian or gradient covariance), not cross-coordinate correlations, but this alone is usually enough to compensate for the fact that different parameter groups in a transformer (embeddings, attention projections, MLP weights, layer-norm gains) naturally operate on very different gradient scales, which is one standard explanation for why plain SGD is markedly harder to tune for transformer training than Adam-family optimizers.

---

## D.7 Stochastic Gradient Descent

### D.7.1 The Minibatch Gradient is Unbiased

The true objective is an expectation over the data distribution, $L(\theta) = E_{(x,y)\sim\mathcal{D}}[\ell(\theta;x,y)]$ (Appendix B §B.5.3). For a minibatch $\{(x_i,y_i)\}_{i=1}^B$ sampled i.i.d. from $\mathcal{D}$, the minibatch gradient estimate satisfies, by linearity of expectation (Appendix B §B.5.3, which holds regardless of any independence assumption):

$$
E\left[\frac{1}{B}\sum_{i=1}^B \nabla_\theta \ell(\theta;x_i,y_i)\right] = \nabla_\theta L(\theta)
$$

so SGD follows the *true* gradient in expectation, with the minibatch estimate differing from it only by mean-zero noise.

### D.7.2 Variance Scales as $1/B$

If each per-example gradient has (coordinate-wise) variance $\sigma^2$ and the examples are drawn independently, the variance of the batch-averaged estimator is $\sigma^2/B$ (Appendix B §B.6.1, $\text{Var}(aX)=a^2\text{Var}(X)$ applied to a sum of $B$ independent terms each scaled by $1/B$). Doubling the batch size halves the gradient noise variance but costs twice the compute per step — this trade-off, not the bias (§D.7.1 shows there is none), is the main quantitative reason batch size matters.

---

## D.8 Convexity

$L$ is **convex** if for all $\theta_1,\theta_2$ and $t\in[0,1]$:

$$
L(t\theta_1 + (1-t)\theta_2) \le t\,L(\theta_1) + (1-t)\,L(\theta_2)
$$

Equivalently, for differentiable $L$: $L(\theta_2) \ge L(\theta_1) + \nabla L(\theta_1)^\top(\theta_2-\theta_1)$ for all $\theta_1,\theta_2$ (the tangent line at any point lies below the function everywhere). Equivalently again, for twice-differentiable $L$: $H(\theta) \succeq 0$ (positive semi-definite, §D.4.3) at every $\theta$. For convex $L$, every local minimum is a global minimum, and gradient descent with a small enough fixed step size is guaranteed to converge to it.

**Neural network losses are not convex** in $\theta$ (compositions of even simple non-linear functions generally aren't), so none of these global guarantees apply directly. What *does* transfer is the purely local analysis of §D.3–D.5: near any point (including near a minimum actually reached during training), the second-order Taylor expansion of §D.4.1 is a legitimate local convex (if $H\succeq0$ there) approximation, and the eigenvalue-based convergence/momentum analysis of this appendix describes the *local* dynamics of optimization even though no equivalent global statement is available.

---

## D.9 Common Identities Reference

$$
\theta_{t+1} = \theta_t - \eta\,\nabla_\theta L(\theta_t) \qquad \text{(gradient descent)}
$$

$$
\eta_{\text{stable}} < \frac{2}{\lambda_{\max}(H)}, \qquad \eta^\star_{\text{GD}} = \frac{2}{\lambda_{\max}+\lambda_{\min}}, \qquad \rho_{\text{GD}} = \frac{\kappa-1}{\kappa+1}
$$

$$
\theta_{t+1} = \theta_t - H(\theta_t)^{-1}\nabla L(\theta_t) \qquad \text{(Newton's method)}
$$

$$
v_{t+1} = \beta v_t + \nabla_\theta L(\theta_t), \quad \theta_{t+1} = \theta_t - \eta v_{t+1} \qquad \text{(momentum)}
$$

$$
\beta^\star = \left(\frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}\right)^2, \qquad \rho_{\text{mom}} = \frac{\sqrt{\kappa}-1}{\sqrt{\kappa}+1}
$$

$$
m_t = \beta_1 m_{t-1}+(1-\beta_1)g_t,\quad v_t=\beta_2 v_{t-1}+(1-\beta_2)g_t^2,\quad \theta_{t+1}=\theta_t - \eta\frac{m_t/(1-\beta_1^t)}{\sqrt{v_t/(1-\beta_2^t)}+\epsilon} \qquad \text{(Adam)}
$$

---

## D.10 Summary: MI-Relevant Optimization Concepts

| Concept | MI Application |
|---------|----------------|
| Gradient = steepest ascent | Same dot-product geometry as attention/similarity (Appendix A §A.2.2) |
| Hessian eigendecomposition | Diagnoses ill-conditioning; same machinery as Appendix A §A.5, Appendix B §B.6.3 |
| Learning-rate stability bound | $\eta < 2/\lambda_{\max}(H)$ — direct consequence of the Hessian spectrum |
| Saddle points vs. minima | Why saddle points dominate high-dimensional loss landscapes |
| Momentum's $O(\sqrt\kappa)$ rate | Formal reason momentum accelerates ill-conditioned training |
| Adam's bias correction | Removes a provable, derivable bias in the raw moment estimates |
| Adam's diagonal rescaling | Cheap surrogate for the whitening transform of Appendix B §B.6.3 |
| Minibatch gradient variance | $\sigma^2/B$ — quantifies the batch-size/compute trade-off |
| Local convexity near minima | Justifies applying the (globally non-convex) eigenvalue analysis locally |

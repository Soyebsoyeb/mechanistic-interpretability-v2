# Appendix D — Optimization

## D.1 Introduction

This appendix is cited elsewhere for two specific facts: the dot-product argument for why $-\nabla_\theta L$ is the steepest-descent direction (Appendix F §F.2.2, alongside Appendix A §A.2.2), and the notion of curvature mismatch across a Hessian's eigenvalues that Appendix F §F.6.2 draws an analogy to when motivating LayerNorm. Both are made precise below (§D.2.2, §D.3.2), along with the convergence and bias-correction arguments the original section headers only stated as formulas. The section order below is rearranged from a flat list into a dependency order — gradient, then curvature/convexity (needed to *state* what "converges" means), then the optimizers themselves, each of which is now justified against that background rather than presented as an unmotivated update rule.

---

## D.2 The Gradient

### D.2.1 Definition

For $L:\mathbb{R}^n\to\mathbb{R}$ differentiable at $\theta$, $\nabla_\theta L \in \mathbb{R}^n$ is the vector of partial derivatives, $(\nabla_\theta L)_i = \partial L/\partial\theta_i$. This is a coordinate-dependent statement in general (it identifies the gradient with the Euclidean-coordinate vector of partials, implicitly using the standard inner product $u^\top v$ on $\mathbb{R}^n$, §A.2.1) — every claim in this appendix inherits that choice, which is the default and matches how $\theta$ is represented in an implementation, but is worth flagging since "gradient" is only defined relative to a choice of inner product on the underlying space.

### D.2.2 Steepest Descent Direction

**Claim.** Among all unit vectors $u$ ($\|u\|=1$), the direction that maximizes the instantaneous rate of increase of $L$ at $\theta$ is $u = \nabla_\theta L/\|\nabla_\theta L\|$, and the direction that maximizes the rate of *decrease* is $u=-\nabla_\theta L/\|\nabla_\theta L\|$ — i.e. $-\nabla_\theta L$ is, up to normalization, the steepest-descent direction, which is what licenses using it (rather than some other direction with negative dot product with the gradient) as the update direction in §D.4.

**Proof.** The directional derivative of $L$ at $\theta$ in direction $u$ is $D_uL(\theta) := \lim_{h\to0}\frac{L(\theta+hu)-L(\theta)}{h} = \nabla_\theta L^\top u$ (first-order Taylor expansion, §D.3.1, with the limit isolating exactly the linear term). By the dot-product-as-similarity identity of Appendix A §A.2.2, $\nabla_\theta L^\top u = \|\nabla_\theta L\|\,\|u\|\cos\phi = \|\nabla_\theta L\|\cos\phi$ (using $\|u\|=1$), where $\phi$ is the angle between $u$ and $\nabla_\theta L$. This is maximized over unit $u$ exactly when $\cos\phi=1$, i.e. $u$ points in the same direction as $\nabla_\theta L$, giving $u = \nabla_\theta L/\|\nabla_\theta L\|$ and maximum rate $\|\nabla_\theta L\|$; it is minimized (most negative, i.e. steepest decrease) exactly when $\cos\phi=-1$, giving $u=-\nabla_\theta L/\|\nabla_\theta L\|$. $\blacksquare$

So gradient descent's update direction is not an arbitrary choice consistent with "go downhill" — among *all* unit-norm directions, $-\nabla_\theta L$ is the *unique* one (up to the edge case $\nabla_\theta L=0$, where every direction has zero directional derivative and $\theta$ is a stationary point) achieving the maximum possible instantaneous rate of decrease, by the same Cauchy–Schwarz-type extremization used throughout Appendix A.

---

## D.3 Second-Order Structure

### D.3.1 The Hessian and Second-Order Taylor Expansion

For $L$ twice differentiable at $\theta$, the **Hessian** $H\in\mathbb{R}^{n\times n}$ has entries $H_{ij} = \partial^2L/\partial\theta_i\partial\theta_j$. If the second partial derivatives are continuous near $\theta$ (the usual case for the smooth losses considered here), **Clairaut's theorem** gives $H_{ij}=H_{ji}$, i.e. **$H$ is symmetric** — which is exactly the precondition needed to invoke Appendix A §A.4.2's spectral theorem: $H$ has real eigenvalues and an orthonormal eigenbasis, a fact §D.3.2 depends on.

**Taylor's theorem** (multivariate, second order, stated without proof) gives, for small $\Delta$,
$$
L(\theta+\Delta) = L(\theta) + \nabla_\theta L^\top\Delta + \tfrac12\Delta^\top H\Delta + o(\|\Delta\|^2),
$$
so the quadratic approximation $L(\theta+\Delta)\approx L(\theta)+\nabla_\theta L^\top\Delta+\tfrac12\Delta^\top H\Delta$ used elsewhere drops a remainder term that is provably small *relative to* $\|\Delta\|^2$ as $\Delta\to0$, but is not controlled in absolute terms for a fixed, non-infinitesimal step — the approximation degrades, potentially badly, for the step sizes actually taken in practice, which is exactly why §D.5's convergence analysis needs a global (not just local, infinitesimal) assumption on how fast $\nabla L$ can change.

### D.3.2 Curvature Mismatch and Conditioning

Restrict attention to a region where $H\succeq0$ (§A.9) — e.g. near a local minimum, or globally for a convex $L$ (§D.4) — so $H$'s eigenvalues $0\le\lambda_1\le\cdots\le\lambda_n$ are well-defined and non-negative (real, by §D.3.1's symmetry). Define the **condition number** $\kappa := \lambda_n/\lambda_1$ (taking $\lambda_1>0$; $\kappa=\infty$ if $\lambda_1=0$, a degenerate flat direction).

**Why $\kappa$ governs gradient descent's behavior.** In the eigenbasis of $H$ (which exists and is orthonormal by §D.3.1 + §A.4.2), the quadratic model of §D.3.1 decouples into $n$ independent one-dimensional quadratics, one per eigenvalue: along the eigendirection with eigenvalue $\lambda_i$, a gradient step behaves like 1-D gradient descent on $f(x)=\tfrac12\lambda_ix^2$, which converges (does not diverge) only for step size $\eta < 2/\lambda_i$, and converges *fastest* at $\eta = 1/\lambda_i$. A single global learning rate $\eta$ must therefore satisfy $\eta<2/\lambda_n$ (the largest eigenvalue, or the update diverges along that direction), while the *rate* of progress along the smallest-eigenvalue direction is governed by $\eta\lambda_1$ — with $\eta$ capped near $1/\lambda_n$, progress along the $\lambda_1$ direction proceeds at rate $\approx\lambda_1/\lambda_n = 1/\kappa$ per step. When $\kappa\gg1$ (a curvature *mismatch* — some directions much more sharply curved than others), no single $\eta$ can be both large enough to make fast progress along shallow directions and small enough to avoid divergence/oscillation along steep ones; this is the precise mechanism behind gradient descent's well-known "zig-zagging" in narrow valleys, and behind Appendix F §F.6.2's analogy — there, the mismatch is across the many additive contributions accumulating in the residual stream across *depth* rather than across a Hessian's eigen-directions in *parameter space*, but both are instances of "a single scalar knob (learning rate, or normalization scale) cannot separately compensate for quantities that vary by orders of magnitude across different directions/components of the same system."

---

## D.4 Convexity

**Definition.** $L$ is convex if, for all $x,y$ and $\lambda\in[0,1]$, $L(\lambda x+(1-\lambda)y) \le \lambda L(x)+(1-\lambda)L(y)$ — the graph of $L$ restricted to any line segment lies on or below the chord connecting its endpoints.

**Claim (first-order characterization).** For differentiable $L$, $L$ is convex iff, for all $x,y$,
$$
L(y) \ge L(x) + \nabla L(x)^\top(y-x)
$$
— i.e. the tangent-plane approximation at any point is a *global underestimate* of $L$ everywhere. *(Proof omitted; standard, obtained by taking a limit of the chord-slope definition as $\lambda\to1$ from the definition above.)*

**Claim (second-order characterization).** For twice-differentiable $L$, $L$ is convex on a convex domain iff $H(x)\succeq0$ (§A.9) for every $x$ in that domain — nonnegative curvature in every direction, everywhere.

**Claim.** If $L$ is convex, every local minimum is a global minimum.

**Proof.** Suppose $\theta^\star$ is a local min but not a global min: some $\theta'$ has $L(\theta')<L(\theta^\star)$. Consider the segment $\theta_\lambda := \lambda\theta'+(1-\lambda)\theta^\star$ for $\lambda\in(0,1)$ small. By convexity, $L(\theta_\lambda)\le\lambda L(\theta')+(1-\lambda)L(\theta^\star) < \lambda L(\theta^\star)+(1-\lambda)L(\theta^\star) = L(\theta^\star)$ (using $L(\theta')<L(\theta^\star)$ strictly), so $L(\theta_\lambda)<L(\theta^\star)$ for every small $\lambda>0$, and $\theta_\lambda\to\theta^\star$ as $\lambda\to0$ — contradicting $\theta^\star$ being a *local* minimum (no neighborhood of $\theta^\star$ can then have $L(\theta^\star)$ as its minimum value). $\blacksquare$

This is the entire reason convexity is worth checking for: it converts "found *a* stationary point" into "found *the* global optimum," a guarantee unavailable for the generally non-convex losses actual neural networks induce (where $H$ is indefinite at many points, having both positive and negative eigenvalues — saddle points — so the analysis of §D.5 below for non-convex $L$ can only promise convergence to a *stationary point*, not a global or even local minimum).

---

## D.5 Gradient Descent

$$
\theta' = \theta - \eta\nabla_\theta L
$$

By §D.2.2 this moves in the (locally) steepest-decrease direction; the remaining question is how large $\eta$ can be while still guaranteeing $L$ actually decreases, given that §D.3.1's quadratic approximation is only a local one.

**Assumption ($L$-smoothness).** Suppose $\nabla L$ is Lipschitz continuous with constant $L_{\text{smooth}}$ (i.e. $\|\nabla L(x)-\nabla L(y)\|\le L_{\text{smooth}}\|x-y\|$ for all $x,y$ — equivalently, if $L$ is twice differentiable, that all eigenvalues of $H$ are bounded by $L_{\text{smooth}}$ in absolute value everywhere, a global version of the local curvature bound in §D.3.2).

**Claim (descent lemma).** Under this assumption, $L(\theta-\eta\nabla L(\theta)) \le L(\theta) - \eta\left(1-\tfrac{L_{\text{smooth}}\eta}{2}\right)\|\nabla L(\theta)\|^2$.

*(Proof omitted — a standard consequence of integrating the Lipschitz bound on $\nabla L$ along the line from $\theta$ to $\theta-\eta\nabla L(\theta)$; the key point is what it implies, below.)*

**Consequence.** The coefficient $1-\tfrac{L_{\text{smooth}}\eta}{2}$ is positive exactly when $\eta < 2/L_{\text{smooth}}$, in which case each gradient step is *guaranteed* to decrease $L$ (strictly, unless already at a stationary point $\nabla L(\theta)=0$) — this is the rigorous, global version of §D.3.2's local, per-eigendirection "$\eta<2/\lambda_i$" stability threshold, with $L_{\text{smooth}}$ playing the role of a worst-case bound on the largest curvature encountered anywhere along the path, not just at one point. It guarantees monotonic decrease and convergence to a stationary point in the non-convex case; combined with convexity (§D.4), it additionally guarantees convergence to the global minimum.

---

## D.6 Stochastic Gradient Descent

$$
\theta_{t+1} = \theta_t - \eta_t\nabla_\theta L_i(\theta_t)
$$

where $L_i$ is the loss on a single example (or minibatch) $i$, drawn randomly (typically uniformly) from the dataset, rather than the full-dataset loss $L = \frac1N\sum_iL_i$ used in §D.5.

**Claim (unbiasedness).** If $i$ is drawn uniformly from $\{1,\ldots,N\}$, $E_i[\nabla_\theta L_i(\theta_t)] = \nabla_\theta L(\theta_t)$.

**Proof.** $E_i[\nabla_\theta L_i(\theta_t)] = \frac1N\sum_{i=1}^N\nabla_\theta L_i(\theta_t) = \nabla_\theta\left(\frac1N\sum_iL_i(\theta_t)\right) = \nabla_\theta L(\theta_t)$, using linearity of expectation (Appendix B §B.4) and linearity of the gradient operator (differentiation commutes with finite sums). $\blacksquare$

So each stochastic gradient step moves, *in expectation*, in the true steepest-descent direction of §D.2.2 — but any single step is a noisy estimate, with variance depending on how much $\nabla L_i$ varies across examples $i$ at the current $\theta_t$ (large when different examples disagree sharply about the best direction to move; the "batch size" and "learning-rate schedule" choices in practice are, in this framing, choices about how much of that variance to average out vs. tolerate per step). Unlike full-batch gradient descent (§D.5), a *constant* $\eta$ does not generally converge exactly to a stationary point even for convex $L$, because the noise floor persists regardless of proximity to the optimum; the classical **Robbins–Monro conditions**, $\sum_t\eta_t=\infty$ (steps large enough in total to reach the optimum from anywhere) and $\sum_t\eta_t^2<\infty$ (steps shrinking fast enough for the accumulated noise to vanish), are a standard sufficient condition for convergence guarantees under a decaying schedule $\eta_t\to0$ — stated here without proof, as the underlying stochastic-approximation argument is beyond this appendix's scope.

---

## D.7 Momentum

$$
v_{t+1} = \beta v_t + \nabla_\theta L(\theta_t), \qquad \theta_{t+1} = \theta_t - \eta v_{t+1}
$$

**Unrolling the recursion**: with $v_0=0$, $v_{t+1} = \sum_{k=0}^{t}\beta^k\nabla_\theta L(\theta_{t-k})$ (direct induction: $v_1=\nabla L(\theta_0)$ matches $k=0$ alone; assuming the form holds for $v_t$, $v_{t+1}=\beta v_t+\nabla L(\theta_t) = \beta\sum_{k=0}^{t-1}\beta^k\nabla L(\theta_{t-1-k}) + \nabla L(\theta_t) = \sum_{k=1}^{t}\beta^k\nabla L(\theta_{t-k}) + \nabla L(\theta_t)$, which is exactly $\sum_{k=0}^t\beta^k\nabla L(\theta_{t-k})$). So $v_{t+1}$ is an **exponentially-weighted moving average** of past gradients, with weight $\beta^k$ on the gradient from $k$ steps ago — recent gradients dominate (weight $\approx1$), older ones decay geometrically.

**Why this helps with the curvature mismatch of §D.3.2**: decompose (as in §D.3.2) into the Hessian's eigendirections. Along a high-curvature direction where the raw gradient oscillates in sign step-to-step (overshoot, correct, overshoot the other way — the mechanism behind zig-zagging), the exponential average partially cancels these alternating-sign contributions, damping the oscillation; along a low-curvature direction where consecutive gradients point consistently the same way, the average instead *reinforces* them, effectively accumulating a larger step than a single gradient would give. This asymmetric effect — damping where consecutive gradients disagree, amplifying where they agree — is precisely what makes momentum improve conditioning-sensitive convergence beyond what a single global $\eta$ can achieve alone in §D.5's analysis, without needing to explicitly estimate $H$'s eigenstructure at all.

---

## D.8 Adam

$$
m_t = \beta_1m_{t-1}+(1-\beta_1)g_t, \qquad v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$
$$
\hat m_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat v_t=\frac{v_t}{1-\beta_2^t}, \qquad \theta_{t+1}=\theta_t-\eta\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

where $g_t := \nabla_\theta L(\theta_t)$ (or a stochastic estimate of it, §D.6), and all operations on $v_t,g_t^2,\sqrt{\hat v_t}$ are elementwise.

### D.8.1 Why the Bias Correction Is Needed

$m_t$ is the same kind of exponential moving average as momentum's $v_t$ (§D.7), but with $m_0=0$ and mixing weight $(1-\beta_1)$: unrolling, $m_t = (1-\beta_1)\sum_{k=0}^{t-1}\beta_1^k g_{t-k}$.

**Claim.** If the true gradient is (approximately) stationary across the averaging window, $E[g_{t-k}]\approx g$ for each $k=0,\ldots,t-1$, then $E[m_t] \approx (1-\beta_1^t)\,g$ — i.e. $m_t$ is a **biased** estimate of $g$, systematically too small by the factor $(1-\beta_1^t)$, especially for small $t$ (early in training, where $\beta_1^t$ is still close to $1$).

**Proof.** $E[m_t] = (1-\beta_1)\sum_{k=0}^{t-1}\beta_1^k\,E[g_{t-k}] \approx (1-\beta_1)g\sum_{k=0}^{t-1}\beta_1^k = (1-\beta_1)g\cdot\frac{1-\beta_1^t}{1-\beta_1} = (1-\beta_1^t)\,g$, using the geometric series formula $\sum_{k=0}^{t-1}\beta_1^k = (1-\beta_1^t)/(1-\beta_1)$ and linearity of expectation (Appendix B §B.4). $\blacksquare$

**Consequence.** Dividing by exactly this factor, $\hat m_t = m_t/(1-\beta_1^t)$, gives $E[\hat m_t]\approx g$ — an (approximately) unbiased estimate. The identical argument applied to $v_t$ against a stationary second moment $E[g_{t-k}^2]\approx v$ gives $E[\hat v_t]\approx v$. Both corrections rely on the *stationarity* assumption stated above — $g_t$'s distribution actually changing little across the $\sim1/(1-\beta)$-step effective window of the moving average — which is only approximately true, and is at its *least* true early in training, exactly when the multiplicative correction $1/(1-\beta_1^t)$ is largest and therefore most consequential if the assumption is violated; this is a known limitation of the bias-correction argument, not a fully general guarantee. $\epsilon$ in the final update plays the same purely-numerical-stability role as LayerNorm's $\epsilon$ (Appendix F §F.6.1): it guarantees the denominator $\sqrt{\hat v_t}+\epsilon$ is bounded away from $0$ even for a coordinate whose gradient has been consistently near-zero, and is not part of the statistical content of the second-moment estimate itself.

---

## D.9 Summary Table

| Concept | Definition | Key fact |
|---|---|---|
| Gradient ($\nabla_\theta L$) | Vector of partials | Coordinate/inner-product dependent (Euclidean assumed) |
| Steepest descent (§D.2.2) | — | $-\nabla_\theta L/\|\nabla_\theta L\|$ uniquely maximizes decrease rate, via Cauchy–Schwarz (App. A §A.2.2) |
| Hessian $H$ | $\partial^2L/\partial\theta_i\partial\theta_j$ | Symmetric (Clairaut) $\Rightarrow$ real eigenvalues, orthonormal eigenbasis (App. A §A.4.2) |
| Condition number $\kappa=\lambda_n/\lambda_1$ | — | Governs the tension between max stable $\eta$ ($<2/\lambda_n$) and progress rate along shallow directions ($\propto1/\kappa$); §D.3.2 |
| Convexity | Chord definition | Local min = global min; equivalent to $H\succeq0$ everywhere (App. A §A.9) |
| Gradient descent | $\theta-\eta\nabla L$ | Guaranteed decrease for $\eta<2/L_{\text{smooth}}$ (descent lemma, §D.5) |
| SGD | $\theta-\eta_t\nabla L_i$ | Unbiased in expectation (App. B §B.4); needs $\eta_t\to0$ (Robbins–Monro) for exact convergence |
| Momentum | EMA of gradients | Damps oscillation along high-curvature directions, reinforces along low-curvature ones (§D.7) |
| Adam bias correction | $\hat m_t=m_t/(1-\beta_1^t)$ | Exactly corrects the geometric-series bias under a stationarity assumption, weakest early in training (§D.8.1) |

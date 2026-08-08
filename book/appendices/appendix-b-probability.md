# Appendix B — Probability for Mechanistic Interpretability

## B.1 Introduction

This appendix reviews the probability theory used throughout this book. As with Appendix A, we emphasize *intuition* and *connections to neural networks* over exhaustive measure-theoretic development.

If you can answer these questions, you're ready:
- What does it mean for a model's output to *be* a probability distribution?
- Why does cross-entropy loss look the way it does?
- What does KL divergence measure, and why is it asymmetric?
- How does a covariance matrix connect back to the eigenvectors of Appendix A?

### B.1.1 Conventions

- Random variables are capital letters ($X, Y$); the values they take are lowercase ($x, y$).
- $p(x)$ denotes a probability mass function (PMF) when $X$ is discrete, and a probability density function (PDF) when $X$ is continuous. Where a statement holds for both, we write $p(x)$ and mean "PMF or PDF" generically; where it matters we say so.
- All expectations are assumed to exist (i.e. the relevant sum or integral converges absolutely) unless noted.
- We work with a fixed underlying probability space $(\Omega, \mathcal{F}, P)$; $\Omega$ is the sample space, $\mathcal{F}$ the event space, and $P$ the probability measure. We rarely need this explicitly, but it is what makes statements like "$P(A)$" well-defined for an event $A \subseteq \Omega$.
- $\log$ denotes the natural logarithm unless a base is given; in the entropy sections we use $\log_2$ explicitly when reporting numbers in *bits*.

---

## B.2 Probability Spaces and Random Variables

### B.2.1 Events and the Axioms

An **event** $A$ is a subset of the sample space $\Omega$. A probability measure $P$ assigns each event a number satisfying the **Kolmogorov axioms**:

$$
P(A) \ge 0 \quad \text{for all } A, \qquad P(\Omega) = 1
$$

$$
P\left(\bigcup_{i=1}^\infty A_i\right) = \sum_{i=1}^\infty P(A_i) \quad \text{if } A_1, A_2, \ldots \text{ are pairwise disjoint}
$$

Everything else in probability theory is a consequence of these three facts.

### B.2.2 Random Variables

A **random variable** $X$ is a function $X: \Omega \to \mathbb{R}$ (or $\mathbb{R}^n$, for a random *vector*). It lets us talk about numerical outcomes without referring back to $\Omega$ directly: we write $P(X = x)$ as shorthand for $P(\{\omega \in \Omega : X(\omega) = x\})$.

- **Discrete** $X$: takes values in a countable set; described by a PMF $p(x) = P(X = x)$, with $\sum_x p(x) = 1$.
- **Continuous** $X$: described by a PDF $p(x)$ with $P(a \le X \le b) = \int_a^b p(x)\,dx$ and $\int_{-\infty}^\infty p(x)\,dx = 1$. Note $p(x)$ is a *density*, not a probability — $p(x)$ can exceed 1, and $P(X = x) = 0$ for any single point.

> **MI connection**: A model's final layer typically outputs logits $z \in \mathbb{R}^V$ over a vocabulary of size $V$. After softmax, $p(x) = \frac{e^{z_x}}{\sum_{x'} e^{z_{x'}}}$ is literally a PMF over the discrete random variable "next token." Every claim in this appendix about discrete PMFs applies directly to that output.

### B.2.3 Cumulative Distribution Function

The CDF $F(x) = P(X \le x)$ is defined for both discrete and continuous $X$ and fully determines the distribution. It is non-decreasing, right-continuous, with $F(-\infty) = 0$ and $F(\infty) = 1$.

---

## B.3 Conditional Probability and Independence

### B.3.1 Definition

For events $A, B$ with $P(B) > 0$:

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)}
$$

**Geometric meaning**: $P(A \mid B)$ restricts attention to the world in which $B$ has already happened, and asks what fraction of *that* world is also in $A$.

Rearranging gives the **multiplication rule**:

$$
P(A \cap B) = P(A \mid B)\, P(B) = P(B \mid A)\, P(A)
$$

which extends to $n$ events as the **chain rule**:

$$
P(A_1 \cap \cdots \cap A_n) = P(A_1)\, P(A_2 \mid A_1)\, P(A_3 \mid A_1, A_2) \cdots P(A_n \mid A_1, \ldots, A_{n-1})
$$

> **MI connection**: Autoregressive language models are literally an application of the chain rule. The joint probability of a sequence of tokens $x_1, \ldots, x_T$ is factored as
> $$
> p(x_1, \ldots, x_T) = \prod_{t=1}^T p(x_t \mid x_1, \ldots, x_{t-1})
> $$
> and the model is trained to approximate each conditional $p(x_t \mid x_{<t})$ directly — this factorization is *why* next-token prediction is a well-posed objective for modeling whole sequences, not an arbitrary design choice.

### B.3.2 Independence

Events $A, B$ are **independent** if $P(A \cap B) = P(A)P(B)$, equivalently $P(A \mid B) = P(A)$ (learning that $B$ happened tells you nothing about $A$). Random variables $X, Y$ are independent if $p(x, y) = p(x)p(y)$ for all $x, y$.

**Caution**: independence is not the same as zero covariance (§B.6.4) — independence implies $\text{Cov}(X,Y) = 0$, but the converse is false in general.

---

## B.4 Bayes' Theorem

### B.4.1 Statement

$$
P(A \mid B) = \frac{P(B \mid A)\, P(A)}{P(B)}
$$

This follows immediately from the multiplication rule in §B.3.1: both $P(B\mid A)P(A)$ and $P(A\mid B)P(B)$ equal $P(A \cap B)$. In the language usually attached to this formula:

$$
\underbrace{P(A \mid B)}_{\text{posterior}} = \frac{\overbrace{P(B \mid A)}^{\text{likelihood}} \; \overbrace{P(A)}^{\text{prior}}}{\underbrace{P(B)}_{\text{evidence}}}
$$

### B.4.2 Law of Total Probability

If $B_1, \ldots, B_n$ partition $\Omega$ (pairwise disjoint, and $\bigcup_i B_i = \Omega$), then for any event $A$:

$$
P(A) = \sum_{i=1}^n P(A \mid B_i)\, P(B_i)
$$

This is what lets us compute the "evidence" term $P(B)$ in Bayes' theorem when we only know the conditionals: $P(B) = \sum_i P(B \mid A_i) P(A_i)$.

### B.4.3 Worked Example: Base Rates and Feature Detectors

Suppose a linear probe is trained to detect whether a rare interpretable feature $F$ is active in a given activation. Say the feature is genuinely present on 1% of inputs, the probe has a 95% true-positive rate, and a 5% false-positive rate:

$$
P(F) = 0.01, \qquad P(\text{pos} \mid F) = 0.95, \qquad P(\text{pos} \mid \neg F) = 0.05
$$

By the law of total probability:

$$
P(\text{pos}) = P(\text{pos}\mid F)P(F) + P(\text{pos}\mid \neg F)P(\neg F) = (0.95)(0.01) + (0.05)(0.99) = 0.059
$$

By Bayes' theorem:

$$
P(F \mid \text{pos}) = \frac{(0.95)(0.01)}{0.059} = \frac{0.0095}{0.059} \approx 0.161
$$

Even with a probe that looks accurate (95% sensitivity, 95% specificity), a positive reading only means the feature is genuinely active about **16%** of the time — because the feature is rare, false positives from the 99% of inactive cases dominate. This is the *base rate fallacy*, and it is a standing hazard when interpreting probe activations, autoencoder feature firings, or any rare-event detector without accounting for how rare the event actually is.

> **MI connection**: This is precisely why interpretability work reports precision/recall (or the full confusion matrix) rather than just "the probe fires when the feature is present" — the latter describes $P(\text{pos}\mid F)$, not the quantity practitioners actually care about, $P(F \mid \text{pos})$.

---

## B.5 Expectation

### B.5.1 Definition

For discrete $X$ with PMF $p(x)$:

$$
E[X] = \sum_x x\, p(x)
$$

For continuous $X$ with PDF $p(x)$:

$$
E[X] = \int_{-\infty}^{\infty} x\, p(x)\, dx
$$

**Geometric meaning**: $E[X]$ is the probability-weighted center of mass of the distribution.

### B.5.2 Law of the Unconscious Statistician (LOTUS)

For a function $g$:

$$
E[g(X)] = \sum_x g(x)\, p(x) \qquad \text{(discrete)}, \qquad E[g(X)] = \int g(x)\, p(x)\, dx \qquad \text{(continuous)}
$$

This lets us compute the expectation of a *transformed* random variable without first deriving the distribution of $g(X)$ — a fact used constantly in defining variance, entropy, and loss functions (all of which are expectations of some function of a random variable).

### B.5.3 Linearity of Expectation

For any random variables $X, Y$ and constants $a, b$:

$$
E[aX + bY] = a\,E[X] + b\,E[Y]
$$

**Critically, this holds regardless of whether $X$ and $Y$ are independent** — it is a consequence of the sum/integral being linear, not of any probabilistic structure. This is what makes expectation tractable for sums of dependent quantities (e.g. correlated tokens in a sequence) where variance (§B.6) is not so forgiving.

> **MI connection**: A training loss is an expectation — $\mathcal{L}(\theta) = E_{(x,y)\sim \mathcal{D}}[\ell(f_\theta(x), y)]$ — and minibatch SGD works *because* the minibatch average is an unbiased estimator of this expectation: $E\left[\frac{1}{B}\sum_{i=1}^B \ell(f_\theta(x_i), y_i)\right] = \mathcal{L}(\theta)$ by linearity, for any batch size $B$.

---

## B.6 Variance and Covariance

### B.6.1 Variance

$$
\text{Var}(X) = E\left[(X - E[X])^2\right]
$$

Equivalently, by expanding the square and using linearity (§B.5.3):

$$
\text{Var}(X) = E[X^2] - (E[X])^2
$$

Properties, for constants $a, b$:

$$
\text{Var}(aX + b) = a^2\, \text{Var}(X)
$$

$$
\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\,\text{Cov}(X, Y)
$$

Note the last identity reduces to $\text{Var}(X) + \text{Var}(Y)$ **only when $\text{Cov}(X,Y) = 0$** — unlike expectation, variance does not distribute over sums of dependent variables without a correction term.

### B.6.2 Covariance

$$
\text{Cov}(X, Y) = E\left[(X - E[X])(Y - E[Y])\right] = E[XY] - E[X]E[Y]
$$

$\text{Cov}(X, X) = \text{Var}(X)$, so variance is a special case of covariance.

**Geometric meaning**: covariance measures whether $X$ and $Y$ tend to deviate from their means *in the same direction* (positive covariance), *opposite directions* (negative covariance), or with no consistent relationship (zero covariance — though as noted in §B.3.2, this does not imply independence).

### B.6.3 The Covariance Matrix

For a random vector $X = (X_1, \ldots, X_n) \in \mathbb{R}^n$, the **covariance matrix** $\Sigma \in \mathbb{R}^{n \times n}$ collects all pairwise covariances:

$$
\Sigma_{ij} = \text{Cov}(X_i, X_j), \qquad \Sigma = E\left[(X - \mu)(X - \mu)^\top\right], \quad \mu = E[X]
$$

$\Sigma$ is symmetric ($\Sigma_{ij} = \Sigma_{ji}$) and positive semi-definite: $v^\top \Sigma v = \text{Var}(v^\top X) \ge 0$ for every $v \in \mathbb{R}^n$, since variance can't be negative. This means $\Sigma$ admits the spectral decomposition of Appendix A §A.5.4:

$$
\Sigma = Q \Lambda Q^\top, \qquad \Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n),\ \lambda_i \ge 0
$$

The eigenvectors (columns of $Q$) are the **principal directions** of the distribution — the directions of maximal and minimal variance — and the eigenvalues are the variances *along* those directions. This is exactly what PCA computes: it is the eigendecomposition of an empirical covariance matrix.

> **MI connection**: Layer-norm-adjacent techniques and whitening transforms use $\Sigma^{-1/2} = Q\Lambda^{-1/2}Q^\top$ to rescale activations so every direction has unit variance — directly reusing the eigendecomposition machinery of Appendix A §A.5–A.6. Feature directions found by an SAE or probe are typically evaluated *relative to* the ambient activation covariance, since a direction that looks "large" in raw coordinates may simply lie along a high-variance direction of $\Sigma$ rather than being meaningfully distinct.

### B.6.4 Correlation

The (Pearson) correlation coefficient normalizes covariance to $[-1, 1]$:

$$
\rho(X, Y) = \frac{\text{Cov}(X, Y)}{\sqrt{\text{Var}(X)\,\text{Var}(Y)}}
$$

$|\rho| = 1$ if and only if $Y$ is an exact affine function of $X$ ($Y = aX + b$ for some constants $a \ne 0, b$) — this follows from the Cauchy–Schwarz inequality applied to the centered variables.

---

## B.7 Common Distributions

### B.7.1 Bernoulli and Categorical

$X \sim \text{Bernoulli}(p)$: $P(X=1) = p$, $P(X=0) = 1-p$, with $E[X] = p$ and $\text{Var}(X) = p(1-p)$.

The **Categorical** distribution generalizes this to $K$ outcomes with probabilities $p_1, \ldots, p_K$ summing to 1 — this is exactly the distribution a softmax layer parameterizes over the vocabulary or class set.

### B.7.2 The Gaussian (Normal) Distribution

Univariate:

$$
p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right), \qquad E[X] = \mu,\ \ \text{Var}(X) = \sigma^2
$$

Multivariate, for $X \in \mathbb{R}^n$ with mean $\mu$ and covariance $\Sigma$ (§B.6.3), assuming $\Sigma$ is invertible:

$$
p(x) = \frac{1}{(2\pi)^{n/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2}(x-\mu)^\top \Sigma^{-1} (x-\mu)\right)
$$

The term $(x-\mu)^\top \Sigma^{-1}(x-\mu)$ is the **Mahalanobis distance** — it measures distance from the mean in units of standard deviation *along each principal direction of $\Sigma$* (§B.6.3), rather than raw Euclidean distance, so it correctly accounts for the fact that the distribution may be far more spread out along some directions than others.

> **MI connection**: Modeling a layer's activation distribution as (approximately) multivariate Gaussian is the working assumption behind many interpretability tools — e.g. flagging out-of-distribution or anomalous activations by Mahalanobis distance from the training-set mean, or generating synthetic "typical" activations for causal interventions.

### B.7.3 Softmax as a Categorical Distribution

Given logits $z \in \mathbb{R}^K$:

$$
p(x = k) = \text{softmax}(z)_k = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}
$$

This is precisely the Categorical distribution of §B.7.1 with $p_k = \text{softmax}(z)_k$; every fact about categorical random variables (entropy, KL divergence to another categorical, etc.) applies directly to a model's predicted next-token distribution.

---

## B.8 Entropy and Information Theory

### B.8.1 Entropy

For a discrete random variable $X$ with PMF $p$:

$$
H(X) = -\sum_x p(x) \log p(x) = E\left[-\log p(X)\right]
$$

(using base-2 logs gives units of **bits**; natural log gives **nats**). $H(X) \ge 0$, with $H(X) = 0$ if and only if $X$ is deterministic (all mass on one outcome), and $H(X)$ is maximized by the uniform distribution over its support.

**Geometric/coding-theoretic meaning**: $H(X)$ is the expected number of bits needed to optimally encode a sample of $X$ (Shannon's source coding theorem).

**Example**: A fair coin ($p = 0.5$) has $H(X) = -0.5\log_2 0.5 - 0.5 \log_2 0.5 = 1$ bit. A biased coin with $p = 0.9$ has

$$
H(X) = -0.9\log_2(0.9) - 0.1\log_2(0.1) \approx 0.9(0.152) + 0.1(3.322) \approx 0.469 \text{ bits}
$$

— less than the fair coin, since the outcome is more predictable.

### B.8.2 Cross-Entropy

For two distributions $p$ (the true/target distribution) and $q$ (a model's predicted distribution) over the same outcomes:

$$
H(p, q) = -\sum_x p(x) \log q(x)
$$

$H(p, q) \ge H(p)$ always, with equality iff $p = q$ (this is Gibbs' inequality, a consequence of Jensen's inequality — see §B.8.3).

> **MI connection**: The standard next-token training loss *is* cross-entropy: with $p$ the one-hot true-token distribution and $q = \text{softmax}(z)$ the model's predicted distribution, $H(p, q) = -\log q(x_{\text{true}})$, since $p$ places all its mass on the single true token. Minimizing this over the data distribution is minimizing $H(p, q)$ averaged over examples — which by the inequality above is minimized exactly when the model's predictive distribution matches the true conditional distribution of the data.

### B.8.3 KL Divergence

$$
D_{\text{KL}}(p \,\|\, q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = H(p, q) - H(p)
$$

**Properties**:
- $D_{\text{KL}}(p\|q) \ge 0$ always, with equality iff $p = q$ (Gibbs' inequality, via Jensen's inequality applied to the concave $\log$).
- $D_{\text{KL}}(p\|q) \ne D_{\text{KL}}(q\|p)$ in general — KL divergence is **not symmetric** and is not a true distance metric (it also fails the triangle inequality).
- $D_{\text{KL}}(p\|q)$ is undefined (or infinite) wherever $q(x) = 0$ but $p(x) > 0$: KL divergence penalizes $q$ heavily for assigning near-zero probability to outcomes $p$ considers possible.

**Example**: for two Bernoulli-like distributions $p = (0.9, 0.1)$ and $q = (0.5, 0.5)$ over two outcomes:

$$
D_{\text{KL}}(p\|q) = 0.9\log_2\!\frac{0.9}{0.5} + 0.1\log_2\!\frac{0.1}{0.5} \approx 0.9(0.848) + 0.1(-2.322) \approx 0.531 \text{ bits}
$$

> **MI connection**: KL divergence appears throughout model training and evaluation beyond the base loss — as a regularizer keeping a fine-tuned policy close to a reference model (e.g. in RLHF), as the objective in knowledge distillation (matching a student's output distribution to a teacher's), and as a similarity measure between two models' or two layers' predictive distributions on the same input. Its asymmetry matters here: penalizing $D_{\text{KL}}(q_{\text{student}} \| p_{\text{teacher}})$ versus $D_{\text{KL}}(p_{\text{teacher}} \| q_{\text{student}})$ produces different training dynamics (roughly, "mode-seeking" versus "mode-covering" behavior).

### B.8.4 Mutual Information

$$
I(X; Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)} = D_{\text{KL}}\big(p(x,y) \,\|\, p(x)p(y)\big)
$$

Equivalently, in terms of entropy:

$$
I(X;Y) = H(X) - H(X\mid Y) = H(Y) - H(Y \mid X)
$$

where $H(X\mid Y) = E_Y[H(X \mid Y=y)]$ is the conditional entropy. $I(X;Y) \ge 0$, with $I(X;Y) = 0$ if and only if $X$ and $Y$ are independent — mutual information is exactly "how many bits of uncertainty about $X$ are removed by observing $Y$."

> **MI connection**: Mutual information gives a principled (if often intractable to compute exactly) way to ask "does this direction/neuron/probe output actually carry information about this feature?" — a probe that achieves high classification accuracy but low estimated mutual information with the target feature suggests the probe is exploiting some correlated-but-distinct signal rather than reading out the feature itself.

---

## B.9 Law of Large Numbers and the Central Limit Theorem

### B.9.1 Law of Large Numbers (LLN)

For i.i.d. $X_1, \ldots, X_n$ with finite mean $\mu$, the sample mean converges to the true mean as $n \to \infty$:

$$
\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i \ \xrightarrow{\ n\to\infty\ }\ \mu
$$

(in probability, under only a finite-mean assumption — this is the weak LLN).

### B.9.2 Central Limit Theorem (CLT)

If additionally $\text{Var}(X_i) = \sigma^2 < \infty$, the *fluctuations* of the sample mean around $\mu$ are asymptotically Gaussian:

$$
\sqrt{n}\,(\bar{X}_n - \mu) \ \xrightarrow{\ d\ }\ \mathcal{N}(0, \sigma^2)
$$

regardless of the shape of the original distribution of $X_i$ — this universality is what makes the Gaussian distribution so central to statistics.

> **MI connection**: A residual stream activation at a given position is, informally, a sum of many roughly-independent contributions accumulated from earlier layers and attention heads. The CLT is the usual informal justification for why individual activation dimensions in large models tend to look approximately Gaussian in practice, and why techniques built on a Gaussian assumption (Mahalanobis-distance anomaly detection, §B.7.2; whitening, §B.6.3) tend to be reasonable approximations even without a proof that any specific activation is exactly Gaussian-distributed.

---

## B.10 Common Identities Reference

For quick reference:

$$
P(A \mid B) = \frac{P(B \mid A) P(A)}{P(B)}
$$

$$
E[aX + bY] = aE[X] + bE[Y] \quad \text{(always, no independence needed)}
$$

$$
\text{Var}(X) = E[X^2] - (E[X])^2
$$

$$
\text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y) + 2\,\text{Cov}(X,Y)
$$

$$
\text{Cov}(X,Y) = E[XY] - E[X]E[Y]
$$

$$
D_{\text{KL}}(p\|q) = H(p,q) - H(p) \ge 0
$$

$$
I(X;Y) = H(X) - H(X\mid Y) = D_{\text{KL}}\big(p(x,y)\,\|\,p(x)p(y)\big)
$$

$$
\Sigma = Q\Lambda Q^\top \quad \text{(covariance matrix; same machinery as Appendix A §A.5.4)}
$$

---

## B.11 Summary: MI-Relevant Probability Concepts

| Concept | MI Application |
|---------|----------------|
| Chain rule | Autoregressive factorization of sequence models |
| Bayes' theorem / base rates | Interpreting probe and feature-detector outputs correctly |
| Linearity of expectation | Why minibatch loss is an unbiased estimate of the true loss |
| Covariance matrix | PCA, whitening, activation-space geometry (ties to Appendix A eigendecomposition) |
| Multivariate Gaussian | Modeling activation distributions, Mahalanobis anomaly detection |
| Cross-entropy | The standard next-token training loss |
| KL divergence | RLHF regularization, distillation, distributional comparison between models |
| Mutual information | Principled measure of whether a probe/direction carries real signal about a feature |
| CLT | Informal justification for approximate Gaussianity of activations |

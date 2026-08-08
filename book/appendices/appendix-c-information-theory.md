# Appendix C — Information Theory for Mechanistic Interpretability

## C.1 Introduction

Appendix B introduced entropy, cross-entropy, KL divergence, and mutual information as consequences of expectation. This appendix goes one level deeper: we derive their joint/conditional forms rigorously, prove the identities that are usually just asserted (chain rules, non-negativity, the data processing inequality), and connect them to results that are used directly in interpretability work — most importantly the **data processing inequality**, which is the formal reason a layer cannot manufacture information it wasn't given, and the **information bottleneck** framework for thinking about what a layer keeps versus discards.

If you can answer these questions, you're ready:
- Why can't mutual information be negative, and why can conditioning only decrease entropy (never increase it)?
- What does it mean, precisely, for a neural network layer to be "information non-increasing"?
- Why is KL divergence the "right" measure of distributional mismatch for a training loss, rather than just an intuitive choice?
- How does Fano's inequality connect a probe's achievable accuracy to how much information a representation actually contains?

### C.1.1 Conventions

We reuse the conventions of Appendix B §B.1.1. All logarithms are base 2 (units of **bits**) unless stated otherwise; switching to natural log (units of **nats**) only rescales every quantity below by a constant factor of $\ln 2$, so no identity in this appendix depends on the choice of base. We write $p(x,y)$ for the joint PMF of $(X,Y)$, $p(x) = \sum_y p(x,y)$ for the marginal, and $p(x\mid y) = p(x,y)/p(y)$ for the conditional (defined wherever $p(y) > 0$).

---

## C.2 Entropy, Joint Entropy, and Conditional Entropy

### C.2.1 Entropy (recap)

$$
H(X) = -\sum_x p(x)\log p(x) = E\left[-\log p(X)\right] \ge 0
$$

as in Appendix B §B.8.1.

### C.2.2 Joint Entropy

For a pair $(X, Y)$, treat $(X,Y)$ as a single random variable with PMF $p(x,y)$:

$$
H(X, Y) = -\sum_{x,y} p(x,y) \log p(x,y)
$$

### C.2.3 Conditional Entropy

$$
H(X \mid Y) = \sum_y p(y)\, H(X \mid Y=y) = -\sum_y p(y) \sum_x p(x\mid y) \log p(x \mid y) = -\sum_{x,y} p(x,y) \log p(x\mid y)
$$

**Geometric meaning**: $H(X\mid Y)$ is the *expected remaining uncertainty* about $X$, averaged over the possible values $Y$ could take, after you learn $Y$.

### C.2.4 Chain Rule for Entropy

$$
H(X, Y) = H(X) + H(Y \mid X) = H(Y) + H(X \mid Y)
$$

**Derivation**: since $p(x,y) = p(x)p(y\mid x)$,

$$
H(X,Y) = -\sum_{x,y} p(x,y)\log p(x,y) = -\sum_{x,y} p(x,y)\big[\log p(x) + \log p(y\mid x)\big]
$$

$$
= -\sum_{x,y} p(x,y)\log p(x) \; - \sum_{x,y} p(x,y)\log p(y\mid x) = H(X) + H(Y\mid X)
$$

using $\sum_y p(x,y) = p(x)$ in the first term. This generalizes to $n$ variables exactly as the chain rule for probability does (Appendix B §B.3.1):

$$
H(X_1, \ldots, X_n) = \sum_{i=1}^n H(X_i \mid X_1, \ldots, X_{i-1})
$$

### C.2.5 Conditioning Reduces Entropy

$$
H(X \mid Y) \le H(X), \qquad \text{with equality iff } X \perp Y
$$

This says learning $Y$ can never *increase* your expected uncertainty about $X$ — on average, information never hurts. (It can increase uncertainty for a *specific* observed value $y$; the inequality only holds after averaging over $Y$.) We prove this in §C.3.2 as an immediate corollary of the non-negativity of mutual information.

**Worked example**: Let $X, Y \in \{0,1\}$ have the joint distribution

| | $Y=0$ | $Y=1$ |
|---|---|---|
| $X=0$ | 0.4 | 0.1 |
| $X=1$ | 0.1 | 0.4 |

Marginals: $p(X=0) = p(X=1) = 0.5$, so $H(X) = 1$ bit. Conditioned on $Y=0$: $p(X=0\mid Y=0) = 0.4/0.5 = 0.8$, $p(X=1\mid Y=0)=0.2$, giving $H(X\mid Y=0) = -0.8\log_2 0.8 - 0.2\log_2 0.2 \approx 0.722$ bits; by symmetry $H(X\mid Y=1) \approx 0.722$ bits too. So $H(X\mid Y) = 0.722$ bits $< H(X) = 1$ bit, as the inequality requires: knowing $Y$ makes $X$ noticeably more predictable, since the table's mass concentrates on the diagonal.

---

## C.3 Mutual Information

### C.3.1 Definition and Equivalent Forms

$$
I(X;Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)}
$$

Using the chain rule (§C.2.4), this is equivalent to each of:

$$
I(X;Y) = H(X) - H(X\mid Y) = H(Y) - H(Y\mid X) = H(X) + H(Y) - H(X,Y)
$$

The three expressions agree because $H(X,Y) = H(X) + H(Y\mid X) = H(Y) + H(X\mid Y)$; substituting either form into $I(X;Y) = H(X) - H(X\mid Y)$ and simplifying recovers the definition above. **Symmetry**, $I(X;Y) = I(Y;X)$, is immediate from this form and is not obvious from the original definition alone — it says the information $Y$ gives about $X$ exactly equals the information $X$ gives about $Y$, even though $H(X\mid Y)$ and $H(Y\mid X)$ are generally *not* equal to each other.

### C.3.2 Non-negativity

$$
I(X;Y) = D_{\text{KL}}\big(p(x,y) \,\|\, p(x)p(y)\big) \ge 0
$$

directly from the general non-negativity of KL divergence (Appendix B §B.8.3, proved via Jensen's inequality — see §C.4.1 below for the proof spelled out). Equality holds iff $p(x,y) = p(x)p(y)$ everywhere, i.e. iff $X \perp Y$. Combined with $I(X;Y) = H(X) - H(X\mid Y) \ge 0$, this proves the "conditioning reduces entropy" claim of §C.2.5.

### C.3.3 Chain Rule for Mutual Information

$$
I(X; Y, Z) = I(X;Y) + I(X;Z\mid Y)
$$

where the **conditional mutual information** is defined analogously to conditional entropy:

$$
I(X;Z\mid Y) = \sum_y p(y)\, I(X;Z \mid Y=y) = H(X\mid Y) - H(X \mid Y, Z)
$$

**Derivation**: apply the entropy chain rule (§C.2.4) twice,

$$
I(X;Y,Z) = H(X) - H(X\mid Y,Z)
$$

$$
= \big[H(X) - H(X\mid Y)\big] + \big[H(X\mid Y) - H(X\mid Y,Z)\big] = I(X;Y) + I(X;Z\mid Y)
$$

Because $I(X;Z\mid Y) \ge 0$ (§C.3.2, applied conditionally), this immediately gives $I(X;Y,Z) \ge I(X;Y)$: **observing more variables can never decrease** the mutual information with $X$. It can, however, be that $I(X;Z\mid Y) = 0$ even though $I(X;Z) > 0$ on its own — i.e. $Z$ can carry information about $X$ that becomes redundant once $Y$ is already known. The chain rule tracks exactly this kind of redundancy, which the two-variable, unconditional $I(X;Y)$ and $I(X;Z)$ cannot.

> **MI connection**: If $Y$ is one layer's activation and $Z$ is a later layer's activation (both derived from the same input $X$ via a fixed, possibly stochastic computation), the chain rule decomposes "how much information the pair $(Y,Z)$ carries about $X$" into what $Y$ already provides, plus whatever *additional* information $Z$ contributes on top of $Y$. This is the natural information-theoretic language for asking whether a later layer is adding new signal about some feature or simply re-representing what an earlier layer already captured.

---

## C.4 KL Divergence, Revisited

### C.4.1 Proof of Non-negativity (Gibbs' Inequality)

**Claim**: $D_{\text{KL}}(p\|q) \ge 0$ for any two distributions $p, q$ over the same support, with equality iff $p = q$.

**Proof**: Jensen's inequality states that for a concave function $f$ and random variable $Z$, $E[f(Z)] \le f(E[Z])$. Apply this with $f = \log$ (concave) and $Z = q(X)/p(X)$ where $X \sim p$:

$$
-D_{\text{KL}}(p\|q) = \sum_x p(x) \log\frac{q(x)}{p(x)} = E_{X\sim p}\left[\log \frac{q(X)}{p(X)}\right] \le \log E_{X\sim p}\left[\frac{q(X)}{p(X)}\right] = \log \sum_x p(x)\frac{q(x)}{p(x)} = \log \sum_x q(x) = \log 1 = 0
$$

so $D_{\text{KL}}(p\|q) \ge 0$. Since $\log$ is *strictly* concave, Jensen's inequality is an equality only when $Z = q(X)/p(X)$ is (almost surely) constant — and since it must average to 1, that constant is 1, i.e. $q(x) = p(x)$ everywhere $p(x) > 0$. $\blacksquare$

### C.4.2 Asymmetry and Its Consequences

As noted in Appendix B §B.8.3, $D_{\text{KL}}(p\|q) \ne D_{\text{KL}}(q\|p)$ in general. Concretely:
- $D_{\text{KL}}(p\|q)$ is large wherever $q$ assigns low probability to an outcome $p$ considers likely — minimizing it over $q$ (with $p$ fixed) tends to produce a $q$ that **covers every mode** of $p$, even at the cost of also covering low-probability regions ("mode-covering" / mean-seeking).
- $D_{\text{KL}}(q\|p)$ is large wherever $q$ assigns *high* probability to an outcome $p$ considers unlikely — minimizing it over $q$ tends to produce a $q$ that **locks onto a single mode** of $p$ and ignores the rest ("mode-seeking").

### C.4.3 Jensen–Shannon Divergence

A symmetrized, bounded alternative: with $m = \tfrac12(p+q)$,

$$
D_{\text{JS}}(p\|q) = \tfrac12 D_{\text{KL}}(p\|m) + \tfrac12 D_{\text{KL}}(q\|m)
$$

Unlike KL, $D_{\text{JS}}(p\|q) = D_{\text{JS}}(q\|p)$, and $0 \le D_{\text{JS}}(p\|q) \le 1$ bit (when using $\log_2$), with the upper bound achieved when $p, q$ have disjoint support. $\sqrt{D_{\text{JS}}}$ is a true metric (satisfies the triangle inequality), which plain KL is not.

---

## C.5 Cross-Entropy, Revisited

Recall from Appendix B §B.8.2:

$$
H(p,q) = -\sum_x p(x)\log q(x) = H(p) + D_{\text{KL}}(p\|q)
$$

Since $H(p)$ does not depend on $q$, minimizing $H(p,q)$ over $q$ (with $p$ — the true data distribution — fixed) is *exactly equivalent* to minimizing $D_{\text{KL}}(p\|q)$ over $q$. This is why cross-entropy training pushes a model's predicted distribution toward the mode-covering behavior described in §C.4.2: the standard token-prediction loss is a forward-KL objective in $q$, not a reverse-KL one.

**Properness**: $H(p,q) \ge H(p)$ for all $q$, with equality iff $q = p$ (§C.4.1). This means cross-entropy is a *proper scoring rule* — the loss is minimized only by reporting the true conditional distribution itself, not by hedging toward some other distribution, which is what justifies reading a trained model's softmax outputs as calibrated probability estimates (to the extent the model has actually converged and has the capacity to represent $p$).

---

## C.6 The Data Processing Inequality

### C.6.1 Statement

Suppose $X \to Y \to Z$ forms a **Markov chain** — meaning $Z$ depends on $X$ only through $Y$, i.e. $Z$ is conditionally independent of $X$ given $Y$: $p(z \mid x, y) = p(z\mid y)$. Then:

$$
I(X;Z) \le I(X;Y)
$$

and symmetrically $I(X;Z) \le I(Y;Z)$.

**Proof**: expand $I(X;Y,Z)$ using the chain rule (§C.3.3) two different ways:

$$
I(X;Y,Z) = I(X;Y) + I(X;Z\mid Y) = I(X;Z) + I(X;Y\mid Z)
$$

Because $X \to Y \to Z$ is Markov, $Z \perp X \mid Y$, so $I(X;Z\mid Y) = 0$. Since $I(X;Y\mid Z) \ge 0$ always (§C.3.2, applied conditionally), we get

$$
I(X;Y) = I(X;Z) + I(X;Y\mid Z) \ge I(X;Z)
$$

$\blacksquare$

### C.6.2 Interpretation

No deterministic or stochastic processing of $Y$ can increase the information it contains about $X$ — it can only preserve or destroy it. Post-processing cannot manufacture information from nothing.

> **MI connection**: This is arguably the single most load-bearing fact from information theory for interpretability. If $X$ is the model's input and $Y_\ell$ denotes the activation at layer $\ell$, then $X \to Y_1 \to Y_2 \to \cdots \to Y_L$ is a Markov chain (each layer's output is computed *only* from the previous layer's output), so
> $$
> I(X; Y_1) \ge I(X; Y_2) \ge \cdots \ge I(X; Y_L)
> $$
> Mutual information about the input can only decrease (or stay the same) as it passes through layers — it never increases. This does *not* mean later layers are "less useful": $I(X;Y_\ell)$ measures raw retained information, not how usefully it is organized for a downstream task, and a layer can *discard task-irrelevant* information about $X$ while making the *task-relevant* part linearly readable for the first time. The DPI bounds the total information budget; it says nothing about accessibility. Likewise, it applies to the *entire* representation $Y_\ell$, not to any single direction or neuron within it — a specific probe direction extracted from $Y_\ell$ can have arbitrarily small mutual information with $X$ even when $I(X;Y_\ell)$ itself is large, simply because the probe is looking at only one projection of a high-dimensional representation.

---

## C.7 Fano's Inequality

### C.7.1 Statement

Suppose we try to guess $X$ from $Y$ using some (possibly randomized) estimator $\hat{X} = g(Y)$, and let $P_e = P(\hat{X} \ne X)$ be the resulting error probability. If $X$ takes values in a set of size $|\mathcal{X}|$, then:

$$
H(X \mid Y) \le H_b(P_e) + P_e \log(|\mathcal{X}| - 1)
$$

where $H_b(p) = -p\log p - (1-p)\log(1-p)$ is the binary entropy function. A looser but often more usable form:

$$
P_e \ \ge\ \frac{H(X\mid Y) - 1}{\log |\mathcal{X}|}
$$

### C.7.2 Interpretation

Fano's inequality lower-bounds the error rate of *any* classifier attempting to recover $X$ from $Y$, purely in terms of the residual uncertainty $H(X\mid Y)$ — no assumption about the classifier's architecture or training procedure is needed. If $H(X\mid Y)$ is large (i.e. $Y$ carries little information about $X$, since $H(X\mid Y) = H(X) - I(X;Y)$ from §C.3.1), *no* estimator, however powerful, can achieve low error.

> **MI connection**: This gives a principled ceiling on probe accuracy. If a representation $Y$ (an activation, a residual-stream slice) has low mutual information with a target feature $X$, Fano's inequality guarantees that *no* probe — linear, non-linear, or otherwise — can classify $X$ from $Y$ with high accuracy, regardless of how the probe is trained. Conversely, if a well-trained probe achieves near-perfect accuracy, that alone is (weak, one-directional) evidence that $I(X;Y)$ must have been substantial to begin with — Fano's inequality bounds error *from below* using $I(X;Y)$, so a *low* observed error certifies a *lower bound* on the information that must have been present, without needing to estimate $I(X;Y)$ directly, which is typically intractable in high dimensions.

---

## C.8 Differential Entropy (Continuous Random Variables)

For a continuous $X$ with density $p(x)$, the direct analogue of entropy is:

$$
h(X) = -\int p(x) \log p(x)\, dx
$$

This is called **differential entropy**, and it behaves differently from discrete entropy in one important way: **$h(X)$ can be negative**, and it is not invariant under change of variables (rescaling $X$ changes $h(X)$ by an additive constant, since it is measuring density relative to Lebesgue measure, not counting outcomes). Consequently, absolute values of $h(X)$ are not directly comparable to bits of a discrete random variable; only *differences* in differential entropy (as in mutual information, defined below) retain the clean information-theoretic meaning.

For a multivariate Gaussian $X \sim \mathcal{N}(\mu, \Sigma)$ (Appendix B §B.7.2):

$$
h(X) = \frac{1}{2}\log\big((2\pi e)^n |\Sigma|\big) = \frac{n}{2}\log(2\pi e) + \frac{1}{2}\log|\Sigma|
$$

using $\log |\Sigma| = \sum_i \log \lambda_i$ (Appendix A §A.8.3, determinant as product of eigenvalues) — so differential entropy of a Gaussian grows with the *log-volume* of its covariance ellipsoid, directly tying back to the eigendecomposition $\Sigma = Q\Lambda Q^\top$ from Appendix B §B.6.3. Mutual information for continuous variables is defined via differential entropy in the same way as the discrete case, $I(X;Y) = h(X) - h(X\mid Y)$, and remains non-negative and well-behaved even though $h(X)$ and $h(X\mid Y)$ individually may not be.

---

## C.9 The Information Bottleneck

### C.9.1 The Trade-off

Given an input $X$, a target $Y$ one is ultimately trying to predict, and a learned representation $T$ of $X$ (e.g. a hidden layer), the **information bottleneck** framing poses representation learning as a trade-off between two mutual informations:

$$
\min_{p(t\mid x)} \; I(X;T) \; - \; \beta\, I(T;Y)
$$

for some trade-off parameter $\beta > 0$: compress $T$ to retain as little information about the raw input $X$ as possible ($I(X;T)$ small), while retaining as much information about the *target* $Y$ as possible ($I(T;Y)$ large). By the data processing inequality (§C.6), since typically $X \to T \to Y$ is not the causal direction but $T$ is a function of $X$, we always have $I(T;Y) \le I(X;Y)$ — a representation can never carry more information about $Y$ than the raw input itself does; the bottleneck objective is about how much of that ceiling is retained per bit spent on $I(X;T)$.

### C.9.2 MI Connection

> The information bottleneck gives one formal lens on why intermediate layers might discard input information (§C.6.2) without harming task performance: if the discarded bits of $I(X;Y_\ell)$ are exactly the bits that are irrelevant to the training objective, then $I(Y_\ell; Y_{\text{target}})$ can stay high (or even be preserved exactly) even as $I(X; Y_\ell)$ shrinks layer over layer. Empirically testing this trade-off directly is difficult, since estimating mutual information in high-dimensional continuous activation spaces is itself a hard and actively researched problem — but the framework gives precise vocabulary for a claim ("this layer is compressing task-irrelevant information") that would otherwise be purely qualitative.

---

## C.10 Common Identities Reference

$$
H(X,Y) = H(X) + H(Y\mid X) = H(Y) + H(X\mid Y)
$$

$$
I(X;Y) = H(X) - H(X\mid Y) = H(X) + H(Y) - H(X,Y) \ge 0
$$

$$
I(X;Y,Z) = I(X;Y) + I(X;Z\mid Y) \ge I(X;Y)
$$

$$
D_{\text{KL}}(p\|q) \ge 0, \quad \text{with equality iff } p = q
$$

$$
H(p,q) = H(p) + D_{\text{KL}}(p\|q)
$$

$$
X \to Y \to Z \text{ Markov} \implies I(X;Z) \le I(X;Y)
$$

$$
P_e \ge \frac{H(X\mid Y) - 1}{\log|\mathcal{X}|} \quad \text{(Fano)}
$$

$$
h(\mathcal{N}(\mu,\Sigma)) = \frac{n}{2}\log(2\pi e) + \frac{1}{2}\log|\Sigma|
$$

---

## C.11 Summary: MI-Relevant Information-Theoretic Concepts

| Concept | MI Application |
|---------|----------------|
| Chain rule for entropy/MI | Decomposing what a later layer adds beyond an earlier one |
| Conditioning reduces entropy | Formal basis for "more context ⟹ less residual uncertainty" |
| KL asymmetry | Explains mode-covering (forward-KL / cross-entropy training) vs. mode-seeking (reverse-KL) behavior |
| Data processing inequality | Formal limit: layers cannot increase information about the input, only preserve or discard it |
| Fano's inequality | Lower-bounds probe error from residual entropy; grounds claims about what a representation *can't* contain |
| Differential entropy | Entropy of continuous activations; ties covariance-matrix eigenvalues to information content |
| Information bottleneck | Formal vocabulary for "this layer compresses task-irrelevant input information" |

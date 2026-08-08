# Appendix C — Information Theory

## C.1 Introduction

Every quantity in this appendix is built from entropy by one of two operations — conditioning, or comparing one distribution to another — and every one of the identities below (mutual information's three equivalent forms, cross-entropy's decomposition, the data processing inequality) is a consequence of two underlying facts: the chain rule for entropy, and the nonnegativity of KL divergence. Both are proved once here (§C.3.2, §C.5.1) and then reused, rather than treating each downstream identity as a separate fact to memorize. Convention: $\log$ is base $2$ unless noted (entropy in bits), though every identity below holds under any fixed base, including $\ln$ (nats) — the standard choice in the cross-entropy training-loss connection of §C.6. By convention, $0\log 0 := 0$ (justified by $\lim_{p\to0^+}p\log p = 0$), so terms with $p(x)=0$ contribute nothing rather than being undefined.

---

## C.2 Entropy

**Definition.** $H(X) = -\sum_x p(x)\log p(x)$, the expected value of $-\log p(X)$ — i.e. $H(X) = E[-\log p(X)]$, the expected "surprisal" of a draw from $p$.

**Claim.** $H(X) \ge 0$, with equality iff $X$ is deterministic (i.e. $p(x)=1$ for exactly one $x$).

**Proof.** Every term $-p(x)\log p(x) \ge 0$ since $p(x)\in[0,1] \Rightarrow \log p(x)\le 0 \Rightarrow -p(x)\log p(x)\ge0$; a sum of nonnegative terms is nonnegative. Equality of the sum to $0$ requires every term to be $0$, i.e. for every $x$ either $p(x)=0$ or $\log p(x)=0$ (i.e. $p(x)=1$); since the $p(x)$ must sum to $1$, exactly one $x$ can have $p(x)=1$ and all others $p(x)=0$. $\blacksquare$

**Claim.** $H(X) \le \log|\mathcal X|$, where $|\mathcal X|$ is the number of values $X$ can take, with equality iff $X$ is uniform.

*(Stated without proof here; it follows from Jensen's inequality applied the same way as §C.5.1 below, and is the discrete-entropy analogue of "uniform is maximum-entropy.")*

---

## C.3 Joint and Conditional Entropy

### C.3.1 Definitions

$$
H(X,Y) = -\sum_{x,y}p(x,y)\log p(x,y), \qquad H(X\mid Y) = -\sum_{x,y}p(x,y)\log p(x\mid y).
$$

Note $H(X\mid Y)$ is **not** $E_y[H(X\mid Y=y)]$ written with a different-looking formula — it is exactly that, restated as a single sum: $H(X\mid Y) = \sum_y p(y)\left(-\sum_x p(x\mid y)\log p(x\mid y)\right) = \sum_y p(y)\,H(X\mid Y=y)$, using $p(x,y)=p(y)p(x\mid y)$ to factor the double sum. So conditional entropy is the *average*, weighted by $p(y)$, of the entropy of $X$'s conditional distribution given each value of $Y$ — not the entropy of some single conditional distribution.

### C.3.2 Chain Rule

**Claim.** $H(X,Y) = H(X) + H(Y\mid X)$.

**Proof.** $p(x,y) = p(x)p(y\mid x)$, so $\log p(x,y) = \log p(x) + \log p(y\mid x)$. Substituting:
$$
H(X,Y) = -\sum_{x,y}p(x,y)\big[\log p(x)+\log p(y\mid x)\big] = -\sum_{x,y}p(x,y)\log p(x) - \sum_{x,y}p(x,y)\log p(y\mid x).
$$
The first term is $-\sum_x\log p(x)\sum_y p(x,y) = -\sum_xp(x)\log p(x) = H(X)$ (marginalizing out $y$); the second term is, by definition, $H(Y\mid X)$. $\blacksquare$

This is the identity every other result in this appendix reduces to. In particular it immediately gives $H(X\mid Y) = H(X,Y)-H(Y)$ (the same claim with $X,Y$ relabeled), which is the form used to derive mutual information's third expression in §C.4.

---

## C.4 Mutual Information

**Definition/claim.** $I(X;Y) := H(X)-H(X\mid Y)$, and this is equal to each of $H(Y)-H(Y\mid X)$ and $H(X)+H(Y)-H(X,Y)$.

**Proof of equivalence.** By the chain rule (§C.3.2) in the form $H(X\mid Y) = H(X,Y)-H(Y)$: $H(X)-H(X\mid Y) = H(X)-H(X,Y)+H(Y) = H(X)+H(Y)-H(X,Y)$, which is manifestly symmetric in $X,Y$ — swapping $X\leftrightarrow Y$ leaves the expression unchanged, so it also equals $H(Y)-H(Y\mid X)$ by the same substitution with roles reversed. $\blacksquare$

So $I(X;Y)$ measures the reduction in uncertainty about $X$ from observing $Y$ (equivalently, about $Y$ from observing $X$ — the symmetry is not a coincidence, it's forced by the algebra above), and the third form exhibits it as exactly the gap between the joint entropy and what the joint entropy would need to be for $X,Y$ to be "as independent as their individual entropies allow."

**Claim.** $I(X;Y) \ge 0$, with equality iff $X,Y$ are independent. *(Proved as a corollary of KL nonnegativity, §C.5.2, since $I(X;Y) = D_{KL}(p(x,y)\,\|\,p(x)p(y))$ — shown there.)*

---

## C.5 KL Divergence

**Definition.** $D_{KL}(p\,\|\,q) = \sum_x p(x)\log\dfrac{p(x)}{q(x)}$, defined when $p$ is absolutely continuous with respect to $q$ (i.e. $q(x)=0 \Rightarrow p(x)=0$ for every $x$ — otherwise a term has $p(x)>0$ divided by $q(x)=0$ and the sum is $+\infty$ by convention, reflecting that $q$ assigns zero probability to an event $p$ considers possible).

### C.5.1 Nonnegativity (Gibbs' Inequality)

**Claim.** $D_{KL}(p\,\|\,q) \ge 0$, with equality iff $p=q$ (as distributions, i.e. $p(x)=q(x)$ for every $x$ with $p(x)>0$).

**Proof.** Using $\ln$ for this proof (nonnegativity is base-independent, since changing base only rescales by a positive constant $1/\ln(\text{base})$):
$$
-D_{KL}(p\|q) = \sum_x p(x)\ln\frac{q(x)}{p(x)} = E_p\!\left[\ln\frac{q(X)}{p(X)}\right] \le \ln E_p\!\left[\frac{q(X)}{p(X)}\right]
$$
by **Jensen's inequality** ($\ln$ is concave, so $E[\ln Z] \le \ln E[Z]$ for any nonnegative random variable $Z$, applied here to $Z=q(X)/p(X)$). And
$$
E_p\!\left[\frac{q(X)}{p(X)}\right] = \sum_{x:\,p(x)>0} p(x)\cdot\frac{q(x)}{p(x)} = \sum_{x:\,p(x)>0}q(x) \le \sum_x q(x) = 1,
$$
so $-D_{KL}(p\|q) \le \ln 1 = 0$, i.e. $D_{KL}(p\|q)\ge0$. Jensen's inequality is an equality here exactly when $Z=q(X)/p(X)$ is (a.s.) constant under $p$, which combined with both distributions summing to $1$ forces that constant to be exactly $1$, i.e. $p(x)=q(x)$ wherever $p(x)>0$. $\blacksquare$

This single inequality is the proof underlying both mutual information's nonnegativity (§C.4, below) and cross-entropy's minimization property (§C.6) — it is the one genuinely nontrivial fact in this appendix, and everything else here is bookkeeping built on top of it.

### C.5.2 Mutual Information as a KL Divergence

**Claim.** $I(X;Y) = D_{KL}\big(p(x,y)\,\|\,p(x)p(y)\big)$.

**Proof.** $D_{KL}(p(x,y)\|p(x)p(y)) = \sum_{x,y}p(x,y)\log\dfrac{p(x,y)}{p(x)p(y)} = \sum_{x,y}p(x,y)\log\dfrac{p(x\mid y)}{p(x)}$ (using $p(x,y)=p(y)p(x\mid y)$) $= \sum_{x,y}p(x,y)\log p(x\mid y) - \sum_{x,y}p(x,y)\log p(x) = -H(X\mid Y) - (-H(X)) = H(X)-H(X\mid Y) = I(X;Y)$. $\blacksquare$

This immediately gives $I(X;Y)\ge0$ (§C.4's remaining claim) as a direct instance of §C.5.1, since $I(X;Y)$ is *literally* the KL divergence between the true joint $p(x,y)$ and the "as-if-independent" product $p(x)p(y)$ — it is large exactly when the joint is far, in this specific asymmetric sense, from the product of its marginals, which is precisely "far from independent."

### C.5.3 Asymmetry

$D_{KL}(p\|q) \ne D_{KL}(q\|p)$ in general — KL divergence is **not** a metric (it also fails the triangle inequality in general), despite behaving somewhat like a "distance" via nonnegativity. The asymmetry has an operational meaning worth stating precisely: $D_{KL}(p\|q)$ heavily penalizes $q(x)$ being small (near $0$) wherever $p(x)$ is not small — the ratio $p(x)/q(x)$ blows up — but does *not* symmetrically penalize $q$ assigning mass to regions where $p(x)\approx0$ (there, $p(x)\log(p(x)/q(x)) \to 0$ regardless of $q(x)$, by the $0\log0=0$ convention). This is why minimizing $D_{KL}(p_{\text{data}}\|q_\theta)$ over $q_\theta$ (as in maximum-likelihood training, §C.6) forces $q_\theta$ to place mass wherever $p_{\text{data}}$ does ("mode-covering"), whereas minimizing $D_{KL}(q_\theta\|p_{\text{data}})$ (as used in some variational-inference objectives, not developed further here) has the opposite tendency ("mode-seeking") — these produce systematically different fitted distributions from the same two underlying distributions $p,q$, precisely because the divergence is not symmetric.

---

## C.6 Cross-Entropy

**Definition.** $H(p,q) = -\sum_x p(x)\log q(x)$.

**Claim.** $H(p,q) = H(p) + D_{KL}(p\,\|\,q)$.

**Proof.** $D_{KL}(p\|q) = \sum_xp(x)\log\dfrac{p(x)}{q(x)} = \sum_xp(x)\log p(x) - \sum_xp(x)\log q(x) = -H(p) + H(p,q)$, so $H(p,q) = H(p)+D_{KL}(p\|q)$. $\blacksquare$

**Consequence.** For a *fixed* true distribution $p$ (e.g. the empirical data distribution, unaffected by the model) and a model distribution $q_\theta$ being optimized over $\theta$, $H(p)$ is a constant with respect to $\theta$, so
$$
\arg\min_\theta H(p,q_\theta) = \arg\min_\theta D_{KL}(p\|q_\theta),
$$
and by §C.5.1 this minimum is exactly $0$ (achieved iff $q_\theta=p$) — **minimizing cross-entropy loss is, exactly, minimizing KL divergence from the data distribution to the model**, not merely correlated with it; the two objectives have identical minimizers and differ only by the $\theta$-independent additive constant $H(p)$. This is the precise justification for cross-entropy as a training loss: the standard language-model training objective $-\sum_x p_{\text{data}}(x)\log q_\theta(x)$, in the single-sample form actually optimized (empirical $p_{\text{data}}$ a point mass on the observed next token, so $H(p)=0$ trivially and the loss reduces to $-\log q_\theta(x_{\text{observed}})$), is exactly this construction specialized to a degenerate $p$.

---

## C.7 Data Processing Inequality

**Claim.** For any (possibly randomized) function $g$ applied to $Y$, $I(X;Y) \ge I(X;g(Y))$ — equivalently, for any Markov chain $X \to Y \to Z$ (meaning $X$ and $Z$ are conditionally independent given $Y$: $p(z\mid x,y)=p(z\mid y)$), $I(X;Y)\ge I(X;Z)$.

**Proof.** By the chain rule for mutual information (an extension of §C.3.2, stated without full derivation: $I(X;Y,Z) = I(X;Z) + I(X;Y\mid Z) = I(X;Y)+I(X;Z\mid Y)$, expanding the joint mutual information two ways), applied to the triple $X,Y,Z$ with $X\to Y\to Z$ a Markov chain:
$$
I(X;Y) + I(X;Z\mid Y) = I(X;Z) + I(X;Y\mid Z).
$$
Because $X\to Y\to Z$ is a Markov chain, $X$ and $Z$ are conditionally independent given $Y$, so $I(X;Z\mid Y) = 0$ (conditional mutual information of conditionally-independent variables is exactly $0$, by the same nonnegativity-and-equality-iff-independent argument as §C.4, applied within each conditioning slice). This leaves $I(X;Y) = I(X;Z) + I(X;Y\mid Z) \ge I(X;Z)$, since $I(X;Y\mid Z)\ge0$ (conditional mutual information is nonnegative for the same reason unconditional mutual information is — §C.5.1 applied conditionally). Taking $Z=g(Y)$ (a deterministic function of $Y$ trivially forms a Markov chain $X\to Y\to g(Y)$, since $g(Y)$ depends on $X$ only through $Y$) gives the stated form $I(X;Y)\ge I(X;g(Y))$. $\blacksquare$

**Interpretation.** No processing of $Y$ — deterministic or randomized, and regardless of how cleverly $g$ is chosen — can *increase* the information $Y$ (in its processed form) carries about $X$; it can only preserve or destroy it. Equality holds exactly when $I(X;Y\mid Z)=0$, i.e. when $g$ (or the second stage of the chain) is a **sufficient statistic** for $X$ given $Y$ — informally, $g(Y)$ throws away nothing about $X$ that $Y$ contained.

> **MI connection**: this is the formal reason a probe, a projection, or any other function applied to a model's internal activations $Y$ can never *reveal more* about an input feature $X$ than the activations themselves already contained — a probe achieving high accuracy is evidence about how much information $Y$ carries about $X$ (bounded above by $I(X;Y)$, per this inequality), not evidence of a computation the *model itself* performs; this is the information-theoretic sharpening of the faithfulness caveat raised in Appendix I §I.2.4 about linear probes measuring decodability rather than use. It is also why "does the model's output entropy collapse" or "how much does attending to token $j$ reduce uncertainty about the target" can be phrased directly as mutual-information quantities between residual-stream activations and downstream outputs (§C.4), with the data processing inequality bounding how such quantities can only shrink, never grow, as information passes through further layers reading only from the current residual stream.

---

## C.8 Summary Table

| Quantity | Definition | Key fact | Depends on |
|---|---|---|---|
| $H(X)$ | $-\sum_xp(x)\log p(x)$ | $\ge0$; $=0$ iff deterministic; $\le \log\lvert\mathcal X\rvert$, max iff uniform | — |
| $H(X,Y)$ | $-\sum p(x,y)\log p(x,y)$ | Chain rule: $H(X,Y)=H(X)+H(Y\mid X)$ (§C.3.2) | $H(X)$, $H(Y\mid X)$ |
| $H(X\mid Y)$ | $-\sum p(x,y)\log p(x\mid y)$ | $=E_y[H(X\mid Y{=}y)]$, an average over $Y$'s conditional entropies | — |
| $I(X;Y)$ | $H(X)-H(X\mid Y)$ | Symmetric in $X,Y$; $=D_{KL}(p(x,y)\|p(x)p(y))$; $\ge0$, $=0$ iff independent | Chain rule (§C.3.2), KL nonneg. (§C.5.1) |
| $D_{KL}(p\|q)$ | $\sum p(x)\log(p(x)/q(x))$ | $\ge0$ (Gibbs, via Jensen), $=0$ iff $p=q$; **asymmetric**, not a metric | Jensen's inequality |
| $H(p,q)$ | $-\sum p(x)\log q(x)$ | $=H(p)+D_{KL}(p\|q)$; minimizing over $q_\theta$ $\equiv$ minimizing $D_{KL}(p\|q_\theta)$ | §C.5.1 (for the minimizer argument) |
| DPI | $I(X;Y)\ge I(X;g(Y))$ | Equality iff $g(Y)$ is a sufficient statistic for $X$; proved via MI chain rule + nonnegativity | Chain rule for $I$, §C.5.1 |

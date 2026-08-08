# Appendix B — Probability

## B.1 Introduction

Several results elsewhere in this book lean on probability facts stated informally at the point of use: Appendix F §F.2.2 sums variances across coordinates to get the $\sqrt{d_k}$ scaling, §F.5.2 invokes concentration of near-orthogonal random directions, and §F.6.2 gestures at a sum of many terms growing "the same way a sum of many random contributions has growing variance." This appendix makes precise the facts those arguments actually rely on — most importantly, the variance-additivity-under-independence fact that both examples silently use, and the notion of convergence in distribution that the CLT's arrow notation invokes but rarely defines.

### B.1.1 Conventions

A **random variable** $X$ is a measurable function from an underlying probability space to $\mathbb{R}$; we do not develop measure theory here and instead work at the level of a probability mass function (pmf) $p(x) = P(X=x)$ for discrete $X$, or a probability density function (pdf) $p(x)$ with $P(a\le X\le b) = \int_a^b p(x)\,dx$ for continuous $X$. All sums/integrals below are assumed to converge (absolutely, for sums with possibly-negative terms) wherever an expectation is written; when they do not, the expectation is simply undefined, a caveat that matters for e.g. heavy-tailed distributions and is stated once here rather than qualifying every formula below.

---

## B.2 Conditional Probability

**Definition.** For events $A,B$ with $P(B)>0$,
$$
P(A\mid B) := \frac{P(A\cap B)}{P(B)}.
$$
The restriction $P(B)>0$ is not a technicality to skip past: $P(A\mid B)$ is simply **undefined** when $P(B)=0$, since the definition would require dividing by zero. (Conditioning on a probability-zero event, such as a specific value of a continuous random variable, requires a separate construction — a conditional density — that is not the same object as this definition specialized to $P(B)=0$; that construction is outside this appendix's scope.)

**Immediate consequence — the multiplication rule**: rearranging the definition, $P(A\cap B) = P(A\mid B)P(B) = P(B\mid A)P(A)$ (the second equality by applying the same definition with the roles of $A,B$ swapped, valid whenever $P(A)>0$ too). This symmetric restatement is the single algebraic step Bayes' theorem below is built from.

---

## B.3 Bayes' Theorem

**Claim.** For $P(A), P(B) > 0$,
$$
P(A\mid B) = \frac{P(B\mid A)P(A)}{P(B)}.
$$

**Proof.** By §B.2's definition, $P(A\mid B) = P(A\cap B)/P(B)$. By the multiplication rule (also §B.2), $P(A\cap B) = P(B\mid A)P(A)$. Substituting gives the claim directly. $\blacksquare$

The theorem is thus not a separate axiom — it is the definition of conditional probability applied twice, once in each direction, and is only as valid as that definition's precondition: both $P(A)>0$ and $P(B)>0$ are required for every quantity in the statement to be defined. In the form most often used in practice, with $A$ a hypothesis and $B$ observed data, and a partition $\{A_i\}$ of mutually exclusive, exhaustive hypotheses, the denominator is expanded via the law of total probability (§B.6):
$$
P(A_j\mid B) = \frac{P(B\mid A_j)P(A_j)}{\sum_i P(B\mid A_i)P(A_i)}.
$$

---

## B.4 Expectation

**Definition.** $E[X] = \sum_x x\,p(x)$ (discrete) or $E[X] = \int x\,p(x)\,dx$ (continuous), when the sum/integral converges absolutely.

**Claim (linearity).** For random variables $X,Y$ and constants $a,b\in\mathbb{R}$, $E[aX+bY] = aE[X]+bE[Y]$ — **with no independence assumption required.**

**Proof (discrete case, joint pmf $p(x,y)$).**
$$
E[aX+bY] = \sum_{x,y}(ax+by)p(x,y) = a\sum_{x,y}xp(x,y) + b\sum_{x,y}yp(x,y) = a\sum_x x\sum_yp(x,y) + b\sum_y y\sum_xp(x,y) = aE[X]+bE[Y],
$$
using $\sum_y p(x,y) = p(x)$ (marginalization) in the last step. $\blacksquare$

This unconditional linearity — it holds for *dependent* $X,Y$ just as well as independent ones — is what is silently used every time an expectation is pushed through a sum elsewhere in the book (e.g. any argument of the form "the expected total is the sum of expected parts"); it is variance and covariance, not expectation, where independence starts to matter (§B.5, §B.6).

---

## B.5 Variance

**Definition.** $\mathrm{Var}(X) = E[(X-E[X])^2]$.

**Claim.** $\mathrm{Var}(X) = E[X^2] - E[X]^2$.

**Proof.** Let $\mu = E[X]$. Expand: $E[(X-\mu)^2] = E[X^2 - 2\mu X + \mu^2] = E[X^2] - 2\mu E[X] + \mu^2$ (linearity, §B.4, with $\mu$ a constant) $= E[X^2] - 2\mu^2+\mu^2 = E[X^2]-\mu^2 = E[X^2]-E[X]^2$. $\blacksquare$

**Claim (scaling).** $\mathrm{Var}(cX) = c^2\mathrm{Var}(X)$ for constant $c$.

**Proof.** $\mathrm{Var}(cX) = E[(cX-cE[X])^2] = E[c^2(X-E[X])^2] = c^2E[(X-E[X])^2] = c^2\mathrm{Var}(X)$. $\blacksquare$ — this is the identity used directly in Appendix F §F.2.2 to show dividing scores by $\sqrt{d_k}$ divides their variance by $d_k$.

**Claim (variance of a sum, general case).** $\mathrm{Var}(X+Y) = \mathrm{Var}(X)+\mathrm{Var}(Y)+2\,\mathrm{Cov}(X,Y)$ (proof below §B.6, after covariance is defined). **Variance is additive over a sum only when the covariance term vanishes** — in particular when $X,Y$ are independent (§B.6 shows independence $\Rightarrow \mathrm{Cov}(X,Y)=0$, though the converse is not true in general: zero covariance does not imply independence). This is the precise condition under which Appendix F §F.2.2's step — summing $d_k$ per-coordinate variances to get $\mathrm{Var}(q_i^\top k_j) = \sum_m \mathrm{Var}(q_{i,m}k_{j,m}) = d_k$ — is valid: it requires the $d_k$ product terms to be pairwise uncorrelated (which the stated independence assumption there guarantees, being stronger than needed), not merely that each term individually has variance $1$.

---

## B.6 Covariance

**Definition.** $\mathrm{Cov}(X,Y) = E[(X-E[X])(Y-E[Y])]$.

**Claim.** $\mathrm{Cov}(X,Y) = E[XY]-E[X]E[Y]$ (proof identical in structure to §B.5's, expanding the product and applying linearity; omitted).

**Claim.** If $X,Y$ are independent, $\mathrm{Cov}(X,Y)=0$.

**Proof.** Independence means the joint pmf/pdf factors, $p(x,y)=p(x)p(y)$, so $E[XY] = \sum_{x,y}xy\,p(x,y) = \sum_{x,y}xy\,p(x)p(y) = \left(\sum_x xp(x)\right)\left(\sum_y yp(y)\right) = E[X]E[Y]$, hence $\mathrm{Cov}(X,Y)=E[XY]-E[X]E[Y]=0$. $\blacksquare$

The converse fails: e.g. $X\sim\mathrm{Uniform}\{-1,0,1\}$ and $Y=X^2$ are dependent ($Y$ is a deterministic function of $X$) but $\mathrm{Cov}(X,Y) = E[X^3]-E[X]E[X^2] = 0 - 0 = 0$ by symmetry — zero covariance only rules out *linear* association, not dependence in general. This is worth stating explicitly because "uncorrelated" is frequently used loosely as if it meant "independent" in informal derivations.

**Proof of variance-of-a-sum** (deferred from §B.5): $\mathrm{Var}(X+Y) = E[(X+Y)^2]-E[X+Y]^2 = E[X^2+2XY+Y^2] - (E[X]+E[Y])^2 = E[X^2]+2E[XY]+E[Y^2] - E[X]^2-2E[X]E[Y]-E[Y]^2$ $= \big(E[X^2]-E[X]^2\big)+\big(E[Y^2]-E[Y]^2\big)+2\big(E[XY]-E[X]E[Y]\big) = \mathrm{Var}(X)+\mathrm{Var}(Y)+2\,\mathrm{Cov}(X,Y)$. $\blacksquare$ By induction, for pairwise-independent $X_1,\ldots,X_n$ (pairwise suffices; full mutual independence is not required for this particular identity), $\mathrm{Var}\!\left(\sum_i X_i\right) = \sum_i\mathrm{Var}(X_i)$, since every cross term $\mathrm{Cov}(X_i,X_j)$, $i\ne j$, vanishes.

---

## B.7 Law of Total Probability

**Claim.** For a partition $\{B_i\}$ of the sample space (mutually exclusive, $B_i\cap B_j=\emptyset$ for $i\ne j$, and exhaustive, $\bigcup_i B_i = \Omega$), with $P(B_i)>0$ for every $i$,
$$
P(A) = \sum_i P(A\mid B_i)P(B_i).
$$

**Proof.** Since $\{B_i\}$ partitions $\Omega$, $\{A\cap B_i\}_i$ partitions $A$ (each point of $A$ lies in exactly one $B_i$, hence in exactly one $A\cap B_i$), so by countable additivity of probability, $P(A) = \sum_i P(A\cap B_i)$. By the multiplication rule (§B.2), $P(A\cap B_i) = P(A\mid B_i)P(B_i)$. Substituting gives the claim. $\blacksquare$

This is exactly what expands the denominator in the "hypothesis testing" form of Bayes' theorem given at the end of §B.3, and is the standard tool for computing an unconditional probability by case-splitting on a partition when the conditional probabilities within each case are easier to reason about directly than $P(A)$ itself.

---

## B.8 Central Limit Theorem

**Statement (Lindeberg–Lévy CLT).** Let $X_1, X_2, \ldots$ be i.i.d. (independent and identically distributed) random variables with $E[X_i]=\mu$ and $\mathrm{Var}(X_i)=\sigma^2 \in (0,\infty)$ (finite and nonzero — the theorem requires a finite second moment; it does not hold as stated for heavy-tailed distributions without finite variance, such as a Cauchy distribution). Then
$$
\frac{1}{\sqrt n}\sum_{i=1}^n (X_i-\mu) \xrightarrow{d} \mathcal N(0,\sigma^2).
$$

**What the arrow means.** $\xrightarrow{d}$ denotes **convergence in distribution**: writing $S_n := \frac{1}{\sqrt n}\sum_i(X_i-\mu)$ and $F_n$ for its cumulative distribution function, convergence in distribution to $\mathcal N(0,\sigma^2)$ (with CDF $\Phi_\sigma$) means $F_n(t) \to \Phi_\sigma(t)$ as $n\to\infty$, for every $t$ at which $\Phi_\sigma$ is continuous (which, for the normal distribution, is every $t\in\mathbb R$, since $\Phi_\sigma$ is everywhere continuous). This is a statement about the *distribution* of $S_n$ approaching the normal distribution's shape as $n$ grows — it is **not** a statement that the sequence of random variables $S_n$ itself converges to any single random variable (indeed $S_n$ for different $n$ are typically dependent and there is no pointwise limiting random variable in general), which is what distinguishes convergence in distribution from the stronger notions of convergence in probability or almost-sure convergence.

**Consistency check with §B.5–B.6.** Before taking any limit, the *exact* (not approximate) variance of the un-normalized sum is, by pairwise-independence and §B.6's induction, $\mathrm{Var}\!\left(\sum_{i=1}^n(X_i-\mu)\right) = \sum_{i=1}^n\mathrm{Var}(X_i-\mu) = n\sigma^2$, so the sum's standard deviation grows as $\sqrt n$ — precisely the scaling that motivates dividing by $\sqrt n$ before taking the limit: $\mathrm{Var}(S_n) = \mathrm{Var}\!\left(\frac{1}{\sqrt n}\sum_i(X_i-\mu)\right) = \frac1n\cdot n\sigma^2 = \sigma^2$ for *every* $n$ (using the scaling property of §B.5), so $S_n$'s variance is already exactly $\sigma^2$ at every finite $n$, not merely in the limit — what the CLT adds beyond this exact variance computation is that $S_n$'s entire *shape* (not just its variance) approaches Gaussian as $n\to\infty$, regardless of the shape of the original distribution of $X_i$.

This $\sqrt n$-growth-of-the-unnormalized-sum mechanism is exactly the one invoked informally in Appendix F §F.6.2 to motivate LayerNorm (a running sum of $L$ residual-stream updates having standard deviation growing like $\sqrt L$ under an independence-and-comparable-variance idealization) and is the same underlying computation as Appendix F §F.2.2's $\mathrm{Var}(q_i^\top k_j)=d_k$ — both are instances of "variance adds under independence, so standard deviation grows as the square root of the number of terms," stated once, precisely, here.

---

## B.9 Summary Table

| Result | Precondition | Used elsewhere for |
|---|---|---|
| $P(A\mid B) = P(A\cap B)/P(B)$ | $P(B)>0$ (undefined otherwise) | Base definition; multiplication rule $P(A\cap B)=P(A\mid B)P(B)$ |
| Bayes' theorem | $P(A),P(B)>0$ | Two applications of the multiplication rule, not a separate axiom |
| $E[aX+bY]=aE[X]+bE[Y]$ | None — holds for dependent $X,Y$ too | Any "expected total = sum of expected parts" argument |
| $\mathrm{Var}(cX)=c^2\mathrm{Var}(X)$ | None | App. F §F.2.2's $\sqrt{d_k}$-scaling variance computation |
| $\mathrm{Var}(X+Y)=\mathrm{Var}(X)+\mathrm{Var}(Y)+2\mathrm{Cov}(X,Y)$ | None (general identity) | Reduces to plain additivity when $\mathrm{Cov}=0$, e.g. under independence |
| Independence $\Rightarrow \mathrm{Cov}(X,Y)=0$ | Independence (converse false) | Justifies dropping cross terms in App. F §F.2.2's variance sum |
| Law of total probability | $\{B_i\}$ a partition, $P(B_i)>0$ | Expands the denominator in Bayes' theorem's hypothesis-testing form |
| CLT | i.i.d., finite nonzero variance $\sigma^2$ | Formalizes the $\sqrt n$-growth reasoning used informally in App. F §F.6.2 (LayerNorm) and §F.2.2 ($\sqrt{d_k}$ scaling) |

# Appendix F — Transformer Mathematics

## F.1 Introduction

As in the earlier appendices, each formula below is restated with the assumption or proof that actually justifies it — in particular, the two positional-embedding schemes at the end are ordinarily *asserted* to encode relative position; here that claim is proved rather than taken on faith, since it is the entire reason either scheme is used instead of a plain learned per-position embedding.

### F.1.1 Conventions

$X\in\mathbb{R}^{n\times d_{\text{model}}}$ holds $n$ token positions as rows. $W_Q,W_K\in\mathbb{R}^{d_{\text{model}}\times d_k}$, $W_V\in\mathbb{R}^{d_{\text{model}}\times d_v}$, $W_O\in\mathbb{R}^{d_v\times d_{\text{model}}}$. $\mathrm{softmax}$ acts row-wise: $\mathrm{softmax}(z)_j = e^{z_j}/\sum_ke^{z_k}$, and satisfies shift-invariance, $\mathrm{softmax}(z+c\mathbf 1)=\mathrm{softmax}(z)$ (immediate from the definition, since the constant $e^c$ cancels top and bottom) — the fact that licenses computing it in practice as $\mathrm{softmax}(z-\max z)$ for numerical stability without changing the result.

---

## F.2 Single-Head Self-Attention

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V, \qquad A=\mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right), \qquad Z=AV,\quad O=ZW_O.
$$

**Why $\sqrt{d_k}$.** If the entries of $q_i,k_j\in\mathbb{R}^{d_k}$ are independent, mean-$0$, variance-$1$ (the initialization regime), then $\mathrm{Var}(q_i^\top k_j) = \sum_{m=1}^{d_k}\mathrm{Var}(q_{i,m}k_{j,m}) = d_k$, using independence to make cross terms in $\mathrm{Var}(\sum_mq_{i,m}k_{j,m})$ vanish (Appendix B §B.6) and $\mathrm{Var}(q_{i,m}k_{j,m})=E[q_{i,m}^2]E[k_{j,m}^2]=1$ for each term. Dividing by $\sqrt{d_k}$ restores unit variance via $\mathrm{Var}(cX)=c^2\mathrm{Var}(X)$ (Appendix B §B.5). Without it, scores of standard deviation $\sqrt{d_k}\gg1$ push $\mathrm{softmax}$ into a regime where its Jacobian $\partial A_j/\partial z_k = A_j(\delta_{jk}-A_k)$ is uniformly near $0$ (every $A$ near one-hot), collapsing the gradient.

**Why $\mathrm{softmax}$, and what $Z=AV$ guarantees.** $\mathrm{softmax}$ guarantees $A_{ij}>0$ and $\sum_jA_{ij}=1$, so $z_i=\sum_jA_{ij}v_j$ is by definition a convex combination of $\{v_j\}$: $z_i\in\mathrm{conv}\{v_1,\ldots,v_n\}$, never an extrapolation beyond it, regardless of the weights. This is stronger than what a "normalize scores to sum to $1$" alternative would give, since raw scores can be negative and such a scheme would not guarantee positivity — only $\mathrm{softmax}$'s exponential form does, while additionally amplifying score gaps exponentially rather than linearly ($A_a/A_b=e^{z_a-z_b}$), which is what lets attention sharpen toward a near-hard selection while remaining everywhere differentiable.

**Causal masking**, when present, sets $\tilde z_{ij}=-\infty$ for $j>i$ before the softmax; since $e^{-\infty}=0$, this contributes exactly $0$ to both $A_{ij}$ and the row's normalizer, so $A_{i,\cdot}$ remains a valid distribution, now supported on $\{1,\ldots,i\}$ only.

---

## F.3 Multi-Head Attention

$$
\mathrm{MultiHead}(X) = \mathrm{Concat}(Z^{(1)},\ldots,Z^{(h)})\,W_O^{\text{multi}}
$$

**Claim.** This equals $\sum_{i=1}^hZ^{(i)}W_O^{(i)}$, where $W_O^{\text{multi}}$'s row-blocks are $W_O^{(1)},\ldots,W_O^{(h)}$.

**Proof.** Block matrix multiplication: for one token's concatenated row $z=[z^{(1)}\cdots z^{(h)}]$, $\big(zW_O^{\text{multi}}\big)_c = \sum_i\sum_mz^{(i)}_m(W_O^{(i)})_{mc} = \sum_i(z^{(i)}W_O^{(i)})_c$, applied row-by-row. $\blacksquare$

So multi-head attention's contribution to the residual stream is an **exact** sum of $h$ independent per-head read-write circuits — a linearity fact, not an approximation, and the reason individual heads can be analyzed in isolation.

---

## F.4 The MLP Block

$$
M(X) = W_2\,\sigma(W_1X+b_1)+b_2
$$

**Claim.** Without $\sigma$, $M$ is affine, and any finite composition of affine maps is affine.

**Proof.** For affine $f(x)=Ax+b$, $g(x)=Cx+d$: $g(f(x))=C(Ax+b)+d=(CA)x+(Cb+d)$, again affine; induct on the number of composed maps. $\blacksquare$ Hence $M(X)=W_2W_1X+(W_2b_1+b_2)$ without $\sigma$, and stacking such layers collapses to one fixed linear map regardless of depth — $\sigma$ is what allows AND/OR-like feature combinations and thresholding, none of which any affine map (of any depth) can represent.

---

## F.5 Residual Stream

$$
X_{\ell+1} = X_\ell+A_\ell+M_\ell
$$

**Claim.** $X_L = X_0+\sum_{\ell=0}^{L-1}(A_\ell+M_\ell)$.

**Proof.** Induction on $L$: base case trivial (empty sum); inductive step, $X_{L+1}=X_L+A_L+M_L = X_0+\sum_{\ell<L}(A_\ell+M_\ell)+A_L+M_L = X_0+\sum_{\ell\le L}(A_\ell+M_\ell)$. $\blacksquare$

Since logits $=X_LW_U$ and matrix multiplication distributes over this sum, the logits decompose into independent per-component contributions — the basis of direct logit attribution.

---

## F.6 Layer Normalization

$$
\mathrm{LayerNorm}(x) = \gamma\odot\frac{x-\mu}{\sqrt{\sigma^2+\epsilon}}+\beta
$$

with $\mu,\sigma^2$ the per-token mean and (biased, population) variance across $d_{\text{model}}$. $\epsilon>0$ is a numerical-stability constant only — it guarantees the denominator is bounded away from $0$ even when a token's entries are (near-)identical (so $\sigma^2\approx0$), and is not part of the statistical definition of standard deviation. Without normalization, each layer's additive update (§F.5) makes the running sum's scale grow with depth by the same variance-adds-under-independence mechanism as Appendix B §B.8's CLT discussion ($L$ roughly-independent, comparable-variance contributions give a sum with standard deviation growing like $\sqrt L$); LayerNorm resets scale before each read, keeping activation and gradient magnitudes stable regardless of depth. $\gamma$ can always be folded into the next layer's weight matrix exactly (a reparameterization, $W\gamma\to W'$); treating $(x-\mu)/\sqrt{\sigma^2+\epsilon}$ as locally linear for attribution purposes (freezing $\sigma$) is an approximation, not an identity.

---

## F.7 Absolute Position Embeddings (Sinusoidal)

$$
PE_{(\mathrm{pos},2i)} = \sin\!\left(\frac{\mathrm{pos}}{10000^{2i/d}}\right), \qquad PE_{(\mathrm{pos},2i+1)} = \cos\!\left(\frac{\mathrm{pos}}{10000^{2i/d}}\right)
$$

for $i=0,\ldots,d/2-1$. Write $\omega_i := 10000^{-2i/d}$, so dimension-pair $i$ oscillates at angular frequency $\omega_i$ as $\mathrm{pos}$ varies; $\omega_i$ decreases geometrically from $\omega_0=1$ to $\omega_{d/2-1}\approx 10000^{-1}$, giving wavelengths (in position units) ranging geometrically from $2\pi$ up to $\approx 10000\cdot2\pi$ — short enough at $i=0$ to distinguish adjacent positions, long enough at $i=d/2-1$ to remain distinguishable across the longest sequences the wavelength schedule is designed for.

**Claim (linear relative-position encoding).** For any fixed offset $k$, there is a matrix $M_k$ (depending on $k$ but *not* on $\mathrm{pos}$) such that, restricted to dimension-pair $i$, $PE_{(\mathrm{pos}+k,\,i)} = M_k^{(i)}\,PE_{(\mathrm{pos},\,i)}$ — i.e. the embedding at any offset $k$ is an exact **linear** function of the embedding at $\mathrm{pos}$, with the linear map itself independent of $\mathrm{pos}$.

**Proof.** Write $u=\sin(\mathrm{pos}\,\omega_i)$, $v=\cos(\mathrm{pos}\,\omega_i)$ (the pair-$i$ coordinates of $PE_{(\mathrm{pos},\cdot)}$). By the angle-addition formulas,
$$
\sin\big((\mathrm{pos}+k)\omega_i\big) = u\cos(k\omega_i)+v\sin(k\omega_i), \qquad \cos\big((\mathrm{pos}+k)\omega_i\big) = v\cos(k\omega_i)-u\sin(k\omega_i).
$$
So $\begin{pmatrix}\sin((\mathrm{pos}+k)\omega_i)\\\cos((\mathrm{pos}+k)\omega_i)\end{pmatrix} = \underbrace{\begin{pmatrix}\cos(k\omega_i) & \sin(k\omega_i)\\-\sin(k\omega_i)&\cos(k\omega_i)\end{pmatrix}}_{=:M_k^{(i)}}\begin{pmatrix}u\\v\end{pmatrix}$, and $M_k^{(i)}$ depends only on $k$ and $i$ (through $\omega_i$), not on $\mathrm{pos}$. $\blacksquare$ ($M_k^{(i)}$ is itself a rotation matrix, by $-k\omega_i$; §F.8 makes the same fact the entire *design principle*, rather than an incidental property, of a different scheme.)

This is the precise sense in which the original transformer paper's claim — that sinusoidal encoding lets the model learn to attend by relative position via a linear function of the encoding — is true: the linear map $M_k$ exists and is exhibited explicitly above; whether any given trained model's attention mechanism actually *implements* that linear map is a separate, empirical question (the claim only guarantees the relative-position information is linearly *available*, in the same sense flagged for probes in Appendix E §E.8/Appendix I §I.2.4, not that the network uses it that way). Also note, unlike a learned embedding table, $PE_{(\mathrm{pos},\cdot)}$ has every coordinate bounded in $[-1,1]$ for every $\mathrm{pos}$, including positions beyond any length seen in training — boundedness the linear relation above does not by itself guarantee (an arbitrary linear map can grow a vector's norm; the specific $M_k$ derived here happens to be a rotation, hence norm-preserving, which is why boundedness holds for every $k$ as well).

---

## F.8 Rotary Position Embedding (RoPE)

Partition $x\in\mathbb{R}^d$ ($d$ even) into $d/2$ pairs $(x_1,x_2),(x_3,x_4),\ldots$, and fix angles $\theta_i := 10000^{-2(i-1)/d}$, $i=1,\ldots,d/2$ (the same frequency schedule as §F.7). At sequence position $m$, RoPE applies an independent 2D rotation to each pair:
$$
R_{\Theta,m}^dx = \begin{pmatrix}x_1\\x_2\\x_3\\x_4\\\vdots\end{pmatrix}\odot\begin{pmatrix}\cos m\theta_1\\\cos m\theta_1\\\cos m\theta_2\\\cos m\theta_2\\\vdots\end{pmatrix} + \begin{pmatrix}-x_2\\x_1\\-x_4\\x_3\\\vdots\end{pmatrix}\odot\begin{pmatrix}\sin m\theta_1\\\sin m\theta_1\\\sin m\theta_2\\\sin m\theta_2\\\vdots\end{pmatrix},
$$
i.e., pair $i$ transforms as $\begin{pmatrix}x_{2i-1}'\\x_{2i}'\end{pmatrix} = \underbrace{\begin{pmatrix}\cos m\theta_i & -\sin m\theta_i\\ \sin m\theta_i & \cos m\theta_i\end{pmatrix}}_{=:R(m\theta_i)}\begin{pmatrix}x_{2i-1}\\x_{2i}\end{pmatrix}$ — an ordinary 2D rotation by angle $m\theta_i$, block-diagonal across the $d/2$ pairs, so $R^d_{\Theta,m}$ as a whole is an **orthogonal** $d\times d$ matrix (block-diagonal with orthogonal blocks).

RoPE is applied to $q_i=x_iW_Q$ and $k_j=x_jW_K$ *before* the attention dot product: $\tilde q_i := R^d_{\Theta,i}q_i$, $\tilde k_j:=R^d_{\Theta,j}k_j$.

**Claim (relative-position dependence).** $\tilde q_i^\top\tilde k_j$ depends on $i,j$ only through $i-j$.

**Proof.** $\tilde q_i^\top\tilde k_j = q_i^\top\big(R_{\Theta,i}^d\big)^\top R_{\Theta,j}^dk_j$. Since $R_{\Theta,m}^d$ is block-diagonal in $2\times2$ rotation blocks $R(m\theta_1),\ldots,R(m\theta_{d/2})$, so is $\big(R_{\Theta,i}^d\big)^\top R_{\Theta,j}^d$, with blocks $R(i\theta_l)^\top R(j\theta_l)$. Two standard rotation-matrix facts, both immediate from the angle-addition formulas: $R(\alpha)^\top=R(-\alpha)$ (a rotation matrix's transpose is its inverse, rotation by the negated angle) and $R(\alpha)R(\beta)=R(\alpha+\beta)$ (composing rotations adds angles). Hence $R(i\theta_l)^\top R(j\theta_l) = R(-i\theta_l)R(j\theta_l) = R\big((j-i)\theta_l\big)$, a function of $j-i$ alone. So
$$
\tilde q_i^\top\tilde k_j = \sum_{l=1}^{d/2}\begin{pmatrix}q_{i,2l-1}\\q_{i,2l}\end{pmatrix}^{\!\top}R\big((j-i)\theta_l\big)\begin{pmatrix}k_{j,2l-1}\\k_{j,2l}\end{pmatrix},
$$
in which $i,j$ appear only inside $R((j-i)\theta_l)$ — a function of $j-i$ — and otherwise only through the original (unrotated) content of $q_i,k_j$ themselves. $\blacksquare$

This is the exact property RoPE is designed to have — the pre-scaling attention score is a function of relative position $j-i$ and content, with no separate dependence on absolute position $i$ or $j$ individually — proved here rather than asserted, and it is a strictly stronger, exact statement than §F.7's linear-availability claim: here the relative-position dependence holds *inside the dot product actually computed by attention*, not merely as a linear map that a downstream computation would additionally need to apply. A further consequence of $R^d_{\Theta,m}$ being orthogonal: $\|\tilde q_i\|=\|q_i\|$ for every $i$ (orthogonal maps preserve norm, §A.5–A.7's machinery), so rotating for position never changes a vector's magnitude — only its direction — unlike additive absolute position embeddings, which can change the *combined* magnitude of content-plus-position by any amount depending on how the two vectors happen to add.

---

## F.9 Summary Table

| Concept | Key fact | Proved via |
|---|---|---|
| $\sqrt{d_k}$ scaling | Restores unit score variance regardless of $d_k$ | Variance additivity under independence (App. B §B.5–B.6) |
| Softmax attention | $Z=AV\in\mathrm{conv}\{v_j\}$; differentiable, unlike hard argmax | Row-stochasticity of $A$ |
| Multi-head sum | Concat-then-project $=$ exact sum of per-head circuits | Block matrix multiplication |
| MLP nonlinearity | Without $\sigma$, collapses to one affine map regardless of depth | Composition of affine maps is affine |
| Residual stream | $X_L=X_0+\sum(A_\ell+M_\ell)$; basis of direct logit attribution | Induction |
| LayerNorm | Bounds activation scale growth across depth | Same mechanism as CLT's $\sqrt L$ variance growth (App. B §B.8) |
| Sinusoidal PE | $PE_{\mathrm{pos}+k}$ is an exact linear (rotation) function of $PE_{\mathrm{pos}}$, $\forall\,\mathrm{pos}$ | Angle-addition formulas (§F.7) |
| RoPE | Rotated dot product $\tilde q_i^\top\tilde k_j$ depends on $i-j$ only, exactly, pre-softmax | Rotation composition $R(\alpha)^\top R(\beta)=R(\beta-\alpha)$ (§F.8) |

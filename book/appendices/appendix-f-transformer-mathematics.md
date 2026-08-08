# Appendix F — Transformer Mathematics

## F.1 Introduction

A transformer forward pass is, underneath the terminology, a sequence of linear reads from and additive writes to one shared object: the **residual stream**. This appendix derives the standard transformer block equation by equation, with an emphasis on *why* each piece is shaped the way it is, and — where the original treatment was informal — with the supporting algebra made explicit. The residual-stream framing is what makes the rest of this book's interpretability tools (direct logit attribution, the QK/OV circuit decomposition, path patching from Appendix E) apply to real transformers at all.

If you can answer these questions, you're ready:

- Why is the attention score $QK^\top$ divided by $\sqrt{d_k}$, and what breaks if it isn't — precisely, in terms of the softmax Jacobian?
- Why does using $\mathrm{softmax}$ (rather than, say, just normalizing the raw scores to sum to 1) matter for what attention can compute, and why is it numerically computed as $\mathrm{softmax}(z-\max z)$ rather than literally as written?
- Why can attention and MLP outputs simply be *added* into the residual stream rather than needing to overwrite it, and what does that imply about how many "things" a fixed-width residual stream can represent at once?
- Why do interpretability researchers analyze attention as two separate **QK circuit** and **OV circuit** matrices rather than as four separate matrices $W_Q, W_K, W_V, W_O$?
- What role does causal masking play, and why does it preserve $A$ as a valid probability distribution over a restricted support?

### F.1.1 Conventions

$X \in \mathbb{R}^{n\times d_{\text{model}}}$ holds $n$ token positions as rows, indexed $i,j \in \{1,\dots,n\}$, each row a $d_{\text{model}}$-dimensional residual-stream vector. $W_Q, W_K \in \mathbb{R}^{d_{\text{model}}\times d_k}$ and $W_V \in \mathbb{R}^{d_{\text{model}}\times d_v}$ are the per-head projection matrices; $W_O \in \mathbb{R}^{d_v \times d_{\text{model}}}$ projects a head's output back into the residual stream. In general $d_k \neq d_v$ is permitted (the score computation and the value/output computation are dimensionally independent); we do not assume $d_k = d_v$ anywhere below except where noted. All projections here are taken to be linear (bias-free); if bias terms $b_Q, b_K, b_V$ are present in an implementation, every claim below still holds with $Q = XW_Q + \mathbf{1}b_Q^\top$ etc., since biases do not affect any of the *linearity-in-$X$-across-terms* arguments used in §F.3 and §F.5.

$\mathrm{softmax}$ is applied **row-wise** unless stated otherwise: for a row $z\in\mathbb{R}^n$,
$$
\mathrm{softmax}(z)_j = \frac{e^{z_j}}{\sum_{k=1}^n e^{z_k}}.
$$
Two elementary but load-bearing facts about this map, used repeatedly below:

**(i) Shift invariance.** For any constant $c\in\mathbb{R}$, $\mathrm{softmax}(z + c\mathbf{1}) = \mathrm{softmax}(z)$, since
$$
\frac{e^{z_j+c}}{\sum_k e^{z_k+c}} = \frac{e^c e^{z_j}}{e^c\sum_k e^{z_k}} = \frac{e^{z_j}}{\sum_k e^{z_k}}.
$$
This is why every practical implementation computes $\mathrm{softmax}(z-\max_k z_k)$ rather than $\mathrm{softmax}(z)$ literally: the value is mathematically identical, but subtracting the row max caps the largest exponent at $e^0=1$, avoiding floating-point overflow for the large scores discussed in §F.2.2 and §F.8, without changing the result.

**(ii) Jacobian.** Writing $A_j := \mathrm{softmax}(z)_j$, the partial derivatives are
$$
\frac{\partial A_j}{\partial z_k} = A_j(\delta_{jk} - A_k),
$$
where $\delta_{jk}$ is the Kronecker delta. This is the precise fact behind the informal claim in §F.2.3/§F.8 that a saturated softmax has vanishing gradient: if some $A_j \to 1$ and all other entries $\to 0$ (the near-one-hot regime), then for every $k$, $A_j(\delta_{jk}-A_k) \to 0$, so *every* entry of the Jacobian vanishes, not merely the entries associated with the near-zero outputs.

---

## F.2 Single-Head Self-Attention

### F.2.1 Query, Key, Value Projections

$$
Q = XW_Q, \qquad K = XW_K, \qquad V = XW_V
$$

Each row of $Q$, $K$, $V$ is a linear readout of one token's residual-stream vector — $Q$ asks "what is this position looking for," $K$ answers "what does this position offer," and $V$ is "what this position will actually send" if attended to. Note $V$ is independent of $Q,K$ *as functions*: they share the input $X$ but are computed by disjoint parameter matrices $W_Q,W_K,W_V$, so which positions get attended to and what gets copied from them are governed by entirely separate degrees of freedom — a separation that becomes central in §F.2.4 and §F.7.

### F.2.2 Attention Scores and the $\sqrt{d_k}$ Scaling

The raw compatibility between query $i$ and key $j$ is their dot product, $(QK^\top)_{ij} = q_i^\top k_j = \sum_{m=1}^{d_k} q_{i,m}k_{j,m}$ — the same dot-product-as-similarity fact from Appendix A §A.2.2 and Appendix D §D.2.2, now measuring how well token $i$'s query matches token $j$'s key.

**Why scale by $\sqrt{d_k}$**: assume the $2d_k$ scalars $\{q_{i,m}\}_{m=1}^{d_k}\cup\{k_{j,m}\}_{m=1}^{d_k}$ are mutually independent, each with mean $0$ and variance $1$ (roughly true early in training under standard Xavier/Glorot-style initialization applied to $X$, $W_Q$, $W_K$ jointly, which is the regime this scaling is designed for; it is an idealization, since $Q$ and $K$ are deterministic functions of the same weights once training has proceeded, but it is the correct regime to reason about at initialization). Under this assumption, for each fixed $m$,
$$
E[q_{i,m}k_{j,m}] = E[q_{i,m}]\,E[k_{j,m}] = 0,\qquad
\mathrm{Var}(q_{i,m}k_{j,m}) = E[q_{i,m}^2 k_{j,m}^2] - 0 = E[q_{i,m}^2]\,E[k_{j,m}^2] = 1\cdot 1 = 1,
$$
using independence of $q_{i,m}$ and $k_{j,m}$ in both steps. Since the terms $\{q_{i,m}k_{j,m}\}_{m=1}^{d_k}$ are pairwise independent across $m$ as well (each depends on a disjoint coordinate pair), variances add under summation:
$$
\mathrm{Var}(q_i^\top k_j) = \mathrm{Var}\!\left(\sum_{m=1}^{d_k} q_{i,m}k_{j,m}\right) = \sum_{m=1}^{d_k}\mathrm{Var}(q_{i,m}k_{j,m}) = d_k.
$$
So the raw score's standard deviation grows as $\sqrt{d_k}$: with $d_k = 64$ (a typical head dimension), $\mathrm{sd}(q_i^\top k_j) = 8$, and scores routinely land around $\pm 8$ to $\pm 16$ (§F.8 makes this concrete). Dividing by $\sqrt{d_k}$ restores unit variance regardless of head dimension, since $\mathrm{Var}(cX) = c^2\mathrm{Var}(X)$ for a constant $c$:
$$
\mathrm{Var}\!\left(\frac{q_i^\top k_j}{\sqrt{d_k}}\right) = \frac{1}{d_k}\mathrm{Var}(q_i^\top k_j) = \frac{d_k}{d_k} = 1.
$$
This matters because $\mathrm{softmax}$ is scale-sensitive (its Jacobian, §F.1.1(ii), depends on the actual magnitude of $z$, not just its rank order): unscaled scores with standard deviation $\sqrt{d_k}\gg 1$ push the softmax into an almost one-hot regime, which — by the Jacobian identity above — drives *every* entry of $\partial A/\partial z$ toward $0$ simultaneously, collapsing attention to a near-hard argmax with essentially no gradient signal, before the model has had any chance to learn a useful *soft* pattern.

### F.2.3 Softmax, Causal Masking, and the Attention Pattern

$$
A = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)
$$

$\mathrm{softmax}$ is used rather than, say, dividing scores by their row sum, because it guarantees:

1. **Strict positivity.** $e^{z_j} > 0$ for every finite $z_j$, so $A_{ij} > 0$ for all $i,j$ (before any masking, see below).
2. **Row-normalization.** $\sum_j A_{ij} = \sum_j e^{z_{ij}}/\sum_k e^{z_{ik}} = 1$ by construction, so every row of $A$ is a genuine probability distribution over positions.
3. **Exponential amplification of gaps.** For two scores $z_a > z_b$ in the same row, $A_a/A_b = e^{z_a - z_b}$: the *ratio* of attention weights grows exponentially in the score gap (§F.8 gives a numeric instance), which is what lets attention approach a sharp, near-hard selection as the underlying scores become confident, while remaining differentiable everywhere by §F.1.1(ii) — unlike a hard $\arg\max$, which has zero gradient almost everywhere.

A "normalize the raw scores to sum to 1" alternative (e.g. $A_{ij} = z_{ij}/\sum_k z_{ik}$) satisfies row-normalization but not the other two properties in general: it does not guarantee positivity when scores can be negative (as raw dot products routinely are), and it amplifies gaps only linearly, not exponentially, giving the model far less ability to sharpen a pattern using the same underlying score magnitude.

**Causal masking.** In an autoregressive (decoder-only) transformer, position $i$ must not be allowed to attend to positions $j > i$, since at generation time those tokens do not yet exist. This is implemented *before* the softmax, not after: define the masked score matrix
$$
\tilde z_{ij} = \begin{cases} (QK^\top)_{ij}/\sqrt{d_k} & j \le i \\ -\infty & j > i\end{cases}, \qquad A = \mathrm{softmax}(\tilde z) \text{ (row-wise)}.
$$
Because $e^{-\infty} = 0$, every masked entry contributes exactly $0$ both to $A_{ij}$ itself and to the row's normalizing sum $\sum_k e^{\tilde z_{ik}}$ — so $A_{i,\cdot}$ remains a valid probability distribution, now supported only on $\{1,\dots,i\}$, with properties 1–3 above holding unchanged relative to that restricted support. In practice $-\infty$ is realized as a large negative finite number (e.g. the dtype's minimum representable value) applied *before* the numerically-stable max-subtraction of §F.1.1(i), so that masked entries still underflow to exactly $0$ after the shift.

### F.2.4 Output

$$
Z = AV, \qquad O = ZW_O
$$

Because each row of $A$ is a probability distribution (§F.2.3), $Z=AV$ computes, for each query position $i$, $z_i = \sum_j A_{ij}v_j$ with $A_{ij}\ge 0$ and $\sum_j A_{ij}=1$ — by definition, a **convex combination of value vectors**. Hence $z_i \in \mathrm{conv}\{v_1,\dots,v_n\}$, the convex hull of the value vectors at the (unmasked) attended positions: the output at position $i$ is confined to this hull and can never be an extrapolation beyond it, regardless of how the attention weights are distributed. $O=ZW_O$ then maps this $d_v$-dimensional combined value back up into the $d_{\text{model}}$-dimensional residual stream via a fixed linear map, ready to be added in (§F.5).

---

## F.3 Multi-Head Attention

### F.3.1 Concatenation Is Equivalent to Summing Independent Heads

$$
\mathrm{MultiHead}(X) = \mathrm{Concat}\big(Z^{(1)},\ldots,Z^{(h)}\big)\, W_O^{\text{multi}}
$$

Write $W_O^{\text{multi}} \in \mathbb{R}^{hd_v \times d_{\text{model}}}$ as $h$ vertically stacked blocks,
$$
W_O^{\text{multi}} = \begin{bmatrix}W_O^{(1)}\\ \vdots \\ W_O^{(h)}\end{bmatrix}, \qquad W_O^{(i)} \in \mathbb{R}^{d_v\times d_{\text{model}}}.
$$
For a single row (token) $z = [z^{(1)} \,\cdots\, z^{(h)}] \in \mathbb{R}^{hd_v}$ formed by concatenating that token's per-head outputs, block matrix multiplication gives, entrywise,
$$
\big(zW_O^{\text{multi}}\big)_c = \sum_{i=1}^h \sum_{m=1}^{d_v} z^{(i)}_m \big(W_O^{(i)}\big)_{mc} = \sum_{i=1}^h \big(z^{(i)}W_O^{(i)}\big)_c,
$$
i.e. $zW_O^{\text{multi}} = \sum_{i=1}^h z^{(i)}W_O^{(i)}$, and this holds row-by-row, so
$$
\mathrm{Concat}\big(Z^{(1)},\ldots,Z^{(h)}\big)\, W_O^{\text{multi}} = \sum_{i=1}^h Z^{(i)} W_O^{(i)}.
$$
So **concatenate-then-project is algebraically identical (an exact identity, not an approximation) to computing each head's contribution separately and adding them** — multi-head attention's entire contribution to the residual stream is a *sum* of $h$ independent per-head read-write circuits, each with its own $W_Q^{(i)}, W_K^{(i)}, W_V^{(i)}, W_O^{(i)}$, none of which interacts with the others except through sharing the same additive destination.

### F.3.2 Why Multiple Heads

Because each head is an independent circuit summed into the same stream (§F.3.1), different heads can implement entirely different query-key matching patterns — one head attending to the previous token, another to the most recent occurrence of the current token, another to a fixed syntactic relationship — all computed in parallel within a single layer and all writing simultaneously into the same $d_{\text{model}}$-dimensional space, rather than needing separate layers or separate slices of the residual stream reserved for each pattern.

> **MI connection**: This decomposition is exactly what licenses analyzing individual attention heads in isolation (e.g. identifying "this specific head is an induction head") rather than only the aggregate multi-head output — the sum-of-independent-heads identity of §F.3.1 is a linearity fact, proved exactly above, not an approximation, so per-head analysis is lossless.

---

## F.4 The MLP Block

### F.4.1 Definition

$$
M(X) = W_2\,\sigma(W_1 X + b_1) + b_2
$$

with $\sigma$ an elementwise nonlinearity (ReLU or GELU in standard transformers).

### F.4.2 Why the Nonlinearity Is Necessary

**Claim.** A finite composition of affine maps is itself affine.

**Proof.** Let $f(x) = Ax+b$ and $g(x) = Cx+d$ be affine. Then
$$
g(f(x)) = C(Ax+b)+d = (CA)x + (Cb+d),
$$
which is affine, with matrix $CA$ and bias $Cb+d$. By induction on the number of maps composed, any finite composition of affine maps is affine. $\blacksquare$

Without $\sigma$, $M(X) = W_2 W_1 X + (W_2 b_1 + b_2)$ is affine in $X$ (an instance of the claim with $A=W_1, b=b_1, C=W_2, d=b_2$). Consequently, stacking any number of such nonlinearity-free "MLP" layers — and, by the same argument applied to a linear attention block, any number of transformer layers with $\sigma$ removed — collapses to a single effective affine map $X \mapsto \bar W X + \bar b$, unable to implement anything beyond one fixed linear transformation of whatever attention has already routed into the residual stream. The nonlinearity is what allows the network to compute functions like logical AND/OR of two input features (e.g. via ReLU thresholds combined additively), general thresholding, or exact memorization of specific input patterns — none of which any affine map can represent regardless of depth, since an affine map is fully determined by its action on a $d_{\text{model}}$-dimensional affine subspace and cannot implement a decision boundary that isn't a hyperplane.

> **MI connection**: Each row of $W_1$ (a "neuron's" input weights) can be read as a **key direction** in the residual stream — the pre-activation $(W_1 x)_j = w_{1,j}^\top x$ is exactly a dot-product similarity (Appendix A §A.2.2) between the residual stream and that neuron's key direction, and the corresponding column of $W_2$ is the **value** written back into the stream, scaled by how strongly that key direction was matched. This "MLP-as-key-value-memory" view — a neuron fires (large pre-activation) when the residual stream aligns with its key, and writes a fixed value direction proportional to how strongly it fired — is the standard lens for interpreting individual MLP neurons.

---

## F.5 The Residual Stream and Additive Updates

### F.5.1 Residual Update Rule

$$
X_{\ell+1} = X_\ell + A_\ell + M_\ell
$$

Every layer **reads** from the current residual stream (via a linear projection — $W_Q,W_K,W_V$ for attention, $W_1$ for the MLP) and **writes** an additive update back into it; nothing already in the stream is overwritten or deleted.

**Claim.** $X_L = X_0 + \sum_{\ell=0}^{L-1}(A_\ell + M_\ell)$.

**Proof.** By induction on $L$. Base case $L=0$: $X_0 = X_0 + \sum_{\ell=0}^{-1}(\cdot) = X_0$ (empty sum), trivially true. Inductive step: assume $X_L = X_0 + \sum_{\ell=0}^{L-1}(A_\ell+M_\ell)$. Then by the update rule, $X_{L+1} = X_L + A_L + M_L = X_0 + \sum_{\ell=0}^{L-1}(A_\ell+M_\ell) + A_L + M_L = X_0 + \sum_{\ell=0}^{L}(A_\ell+M_\ell)$. $\blacksquare$

Because the final logits are a linear readout of $X_L$ (the unembedding matrix $W_U$, i.e. $\mathrm{logits} = X_L W_U$), and matrix multiplication distributes over the sum above,
$$
\mathrm{logits} = X_0 W_U + \sum_{\ell=0}^{L-1}\big(A_\ell W_U + M_\ell W_U\big),
$$
so the logits themselves decompose into a sum of independent per-component contributions — the formal basis for **direct logit attribution**: the contribution of any single head or MLP to the final output can be computed by projecting that one term through $W_U$ directly, without needing to run the rest of the network, precisely because addition and the final linear readout commute (this exactness is subject to the LayerNorm caveat of §F.6.3, since in practice $X_L$ is read through $\mathrm{LN}(X_L)$, not $X_L$ directly).

### F.5.2 Superposition

The residual stream has a fixed width $d_{\text{model}}$, but a trained network typically represents many more than $d_{\text{model}}$ distinct features across its components. This is only possible because §F.5.1 makes the stream additive rather than a fixed set of labeled slots: many components can write nearly-orthogonal directions into the same $d_{\text{model}}$-dimensional space, and as long as a given readout's dot product with the directions it doesn't care about stays small, those writes coexist without destructively interfering.

This is a quantitative, not merely qualitative, phenomenon: if $m$ unit vectors $u_1,\ldots,u_m \in \mathbb{R}^{d_{\text{model}}}$ are drawn independently and uniformly at random (e.g. by normalizing i.i.d. Gaussian vectors), then for any fixed pair $i\ne j$, $E[\langle u_i,u_j\rangle] = 0$, and standard concentration results (in the spirit of the Johnson–Lindenstrauss lemma) show $|\langle u_i,u_j\rangle| = O\!\left(\sqrt{\log m / d_{\text{model}}}\right)$ with high probability, uniformly over all $\binom{m}{2}$ pairs, provided $m$ grows at most sub-exponentially in $d_{\text{model}}$. So the number of *nearly*-orthogonal directions a $d_{\text{model}}$-dimensional space can host grows exponentially in $d_{\text{model}}$, far outstripping $d_{\text{model}}$ itself — this is the precise sense in which superposition lets a fixed-width stream represent more features than its dimension.

> **MI connection**: This is why interpretability of the residual stream is fundamentally a *decomposition* problem, not a *labeling* problem — there is no fixed dimension "reserved" for any one feature, and the same coordinates can be shared, in superposition, by features that rarely co-occur.

---

## F.6 LayerNorm

### F.6.1 Definition

$$
\mathrm{LN}(x) = \gamma \odot \frac{x-\mu}{\sigma + \epsilon} + \beta
$$

where, for a single token's vector $x\in\mathbb{R}^{d_{\text{model}}}$, $\mu = \frac{1}{d_{\text{model}}}\sum_{c=1}^{d_{\text{model}}} x_c$ and $\sigma = \sqrt{\frac{1}{d_{\text{model}}}\sum_{c=1}^{d_{\text{model}}}(x_c-\mu)^2}$ are the mean and (biased, population) standard deviation of $x$'s entries, computed *per token* — across the $d_{\text{model}}$ dimension, not across the batch or sequence axis — and $\gamma,\beta \in \mathbb{R}^{d_{\text{model}}}$ are learned elementwise scale and shift parameters shared across all tokens and positions. $\epsilon > 0$ is a small constant (e.g. $10^{-5}$) added purely for numerical stability, to guarantee the denominator is bounded away from $0$ even for a token whose entries happen to be (near-)identical, in which case $\sigma \approx 0$; it is not part of the statistical definition of standard deviation and does not appear in the idealized versions of the identities elsewhere in this appendix.

### F.6.2 Why Normalization Is Needed

Each residual-stream update is one more additive term (§F.5.1); left unchecked, the running sum's typical magnitude tends to grow with depth simply from accumulating more terms — informally, if the $L$ additive terms behaved like independent, mean-zero contributions of comparable variance $v$, the running sum's variance would grow as $\Theta(Lv)$ and its standard deviation as $\Theta(\sqrt{L})$, the same variance-of-a-sum mechanism used in §F.2.2 — an implicit ill-conditioning across depth analogous to the curvature mismatches of Appendix D §D.3.2, here across layers instead of parameters. Normalizing each token's vector to fixed mean and scale before it is read by the next layer keeps activation magnitudes, and consequently gradient magnitudes during backpropagation, in a stable range regardless of depth $L$.

### F.6.3 MI Connection: Folding LayerNorm

$\mathrm{LN}$'s division by a data-dependent $\sigma(x)$ makes it technically nonlinear in $x$, which complicates the clean additive-sum story of §F.5.1: strictly, a downstream component doesn't read $X_\ell$ directly, it reads $\mathrm{LN}(X_\ell)$. Two standard remedies are used in practice: (i) the elementwise scale $\gamma$ can always be **folded** into the very next layer's weight matrix ($W\gamma \to W'$, a pure reparameterization with no approximation, since $\mathrm{diag}(\gamma)$ commutes into $W$ by definition of matrix multiplication), removing that part of $\mathrm{LN}$ from consideration entirely; and (ii) for a fixed input, the remaining $(x-\mu)/(\sigma+\epsilon)$ operation is treated as an approximately linear map for the purposes of attribution (freezing $\sigma$ at its observed value rather than treating it as a function of $x$) — an approximation, not an identity, because $\sigma$ genuinely depends on $x$ and this dependence is discarded, but one that in practice preserves the additive decomposition of §F.5.1 well enough for direct logit attribution and activation patching to remain informative.

---

## F.7 Assembling a Full Transformer Block

A standard pre-LayerNorm block combines all of the above: attention and the MLP each read a *normalized* copy of the stream but write their update directly into the *unnormalized* running sum,

$$
X'_\ell = X_\ell + \mathrm{Attn}\big(\mathrm{LN}(X_\ell)\big)
$$

$$
X_{\ell+1} = X'_\ell + \mathrm{MLP}\big(\mathrm{LN}(X'_\ell)\big)
$$

with $\mathrm{Attn}(\cdot)$ expanding to the multi-head, (optionally causally-masked, §F.2.3) computation of §F.2–F.3, and $\mathrm{MLP}(\cdot)$ to §F.4. This is exactly why "reads are normalized, writes are additive to the raw stream" is the right way to state the residual-stream picture of §F.5: normalization only ever affects what a layer *sees*, never what accumulates — an invariant that follows directly from the update rule being $X_{\ell+1} = (\cdot) + \mathrm{MLP}(\mathrm{LN}(\cdot))$ rather than $X_{\ell+1} = \mathrm{LN}(X'_\ell + \mathrm{MLP}(\cdot))$.

### F.7.1 The QK Circuit and the OV Circuit

Substituting §F.2.1 into §F.2.2, and dropping the head superscript and the LayerNorm (folded per §F.6.3) for clarity, the attention score between positions $i$ and $j$ is
$$
\frac{q_i^\top k_j}{\sqrt{d_k}} = \frac{(x_i W_Q)^\top(x_j W_K)}{\sqrt{d_k}} = \frac{1}{\sqrt{d_k}}\, x_i^\top W_Q W_K^\top x_j = \frac{1}{\sqrt{d_k}}\, x_i^\top \big(W_Q W_K^\top\big) x_j,
$$
using $(x_iW_Q)^\top = x_i^\top W_Q^\top$... more precisely, $q_i^\top k_j = (W_Q^\top x_i)^\top(W_K^\top x_j)$ if $x_i$ is treated as a column vector and $Q=XW_Q$ means $q_i^\top = x_i^\top W_Q$; either convention collapses to the same bilinear form $x_i^\top(W_QW_K^\top)x_j$ once expanded, so the entire query/key computation reduces to a single bilinear form in the **QK circuit** matrix
$$
W_{QK} := W_Q W_K^\top \in \mathbb{R}^{d_{\text{model}}\times d_{\text{model}}} \qquad\text{via}\qquad \frac{q_i^\top k_j}{\sqrt{d_k}} = \frac{1}{\sqrt{d_k}}\,x_i^\top W_{QK}\,x_j,
$$
a fixed, input-independent object that alone determines *which* positions a head attends to, as a function purely of the two tokens' residual-stream vectors $x_i, x_j$, with no dependence on $V$ or $W_O$ at all.

Symmetrically, the value written to the residual stream by position $j$ (before being weighted by attention and summed) is $v_j W_O = (x_jW_V)W_O = x_j(W_VW_O)$, reducing the entire value/output computation to the **OV circuit** matrix
$$
W_{OV} := W_V W_O \in \mathbb{R}^{d_{\text{model}}\times d_{\text{model}}},
$$
a fixed linear map determining *what* gets copied into the residual stream (via $x_j \mapsto x_jW_{OV}$), entirely independent of which positions get attended to. Combining both, the head's total contribution to the residual stream at position $i$ (§F.2.4, ignoring masking) is
$$
o_i = \sum_j A_{ij}\,x_jW_{OV}, \qquad A_{ij} = \mathrm{softmax}_j\!\left(\frac{1}{\sqrt{d_k}}x_i^\top W_{QK}x_j\right),
$$
making explicit that $W_{QK}$ enters only inside the softmax (determining $A$) and $W_{OV}$ enters only outside it (determining what $A$ mixes) — the two circuits interact solely through this shared $A$, never through direct matrix composition with each other.

> **MI connection**: This QK/OV split (Elhage et al.'s transformer-circuits framework) is the reason a head's function can be summarized as "where it looks" ($W_{QK}$) and "what it copies" ($W_{OV}$) as two separate, independently interpretable objects — e.g. an *induction head*'s OV circuit is close to the identity map (it copies the attended token's identity forward), while its QK circuit implements "attend to the position after the last occurrence of the current token" via composition with an earlier head's output — and it is also what makes **virtual heads** (multi-layer compositions of two heads' QK or OV circuits) analyzable at all: because both circuits are just $d_{\text{model}}\times d_{\text{model}}$ matrices, composing two heads across layers is literal matrix multiplication of their respective $W_{QK}$ or $W_{OV}$ matrices.

---

## F.8 Worked Example: The Cost of Skipping the $\sqrt{d_k}$ Scale

Take a realistic head dimension $d_k = 64$, and suppose (as at initialization) $q_i, k_j$ have independent, unit-variance entries. By §F.2.2, the raw score $q_i^\top k_j$ has variance $64$, i.e. standard deviation $\sqrt{64}=8$ — so scores across a row of $QK^\top$ routinely span a range of $\pm 16$ or more (two standard deviations). Feeding scores that large directly into $\mathrm{softmax}$: for two scores differing by $16$, property 3 of §F.2.3 gives a weight ratio of $e^{16}\approx 8.9\times10^{6}$, so the attention weight on the lower-scoring position is smaller than $10^{-6}$ relative to the other — the row is, in practice, one-hot. By the Jacobian identity of §F.1.1(ii), with $A_j\approx 1$ and $A_k\approx 0$ for $k\ne j$, every entry $\partial A_\cdot/\partial z_\cdot \approx 0$: the gradient of $\mathrm{softmax}$ with respect to any entry is correspondingly tiny, not just for the near-zero outputs but for the near-one output too. Dividing first by $\sqrt{d_k}=8$ brings the same pair of scores back to a gap of $16/8=2$, where $e^2\approx 7.4$ — a soft, learnable weighting rather than a frozen near-hard selection. This is the concrete mechanism behind the qualitative claim in §F.2.2: the scaling isn't cosmetic, it is what keeps the attention pattern in a regime where gradient-based learning (Appendix D) can still move it.

---

## F.9 Common Identities Reference

**Single-head attention** (with optional causal mask $\tilde z_{ij} = -\infty$ for $j>i$)

$$
Q = XW_Q,\quad K = XW_K,\quad V = XW_V
$$

$$
A = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) \text{ (or masked } \tilde z\text{)}, \qquad Z = AV, \qquad O = ZW_O
$$

**Multi-head attention**

$$
\mathrm{MultiHead}(X) = \mathrm{Concat}(Z^{(1)},\ldots,Z^{(h)})\,W_O^{\text{multi}} = \sum_{i=1}^h Z^{(i)}W_O^{(i)}
$$

**MLP**

$$
M(X) = W_2\,\sigma(W_1X+b_1)+b_2
$$

**Residual stream**

$$
X_{\ell+1} = X_\ell + A_\ell + M_\ell, \qquad X_L = X_0 + \sum_{\ell=0}^{L-1}(A_\ell+M_\ell)
$$

**LayerNorm** (with numerical-stability constant $\epsilon$)

$$
\mathrm{LN}(x) = \gamma \odot \frac{x-\mu}{\sigma+\epsilon} + \beta
$$

**Pre-LN transformer block**

$$
X'_\ell = X_\ell + \mathrm{Attn}(\mathrm{LN}(X_\ell)), \qquad X_{\ell+1} = X'_\ell + \mathrm{MLP}(\mathrm{LN}(X'_\ell))
$$

**QK / OV circuits**

$$
W_{QK} = W_Q W_K^\top, \qquad W_{OV} = W_V W_O, \qquad o_i = \sum_j A_{ij}\,x_jW_{OV},\ \ A_{ij}=\mathrm{softmax}_j\!\left(\tfrac{1}{\sqrt{d_k}}x_i^\top W_{QK}x_j\right)
$$

**Softmax identities**

$$
\mathrm{softmax}(z+c\mathbf{1}) = \mathrm{softmax}(z), \qquad \frac{\partial\,\mathrm{softmax}(z)_j}{\partial z_k} = \mathrm{softmax}(z)_j\big(\delta_{jk} - \mathrm{softmax}(z)_k\big)
$$

---

## F.10 Summary: MI-Relevant Transformer Concepts

| Concept | MI Application |
|---|---|
| Dot-product attention score | Same similarity geometry as Appendix A §A.2.2 / Appendix D §D.2.2, applied to query–key matching |
| $\sqrt{d_k}$ scaling | Keeps softmax out of a saturated regime where the Jacobian $A_j(\delta_{jk}-A_k)$ vanishes uniformly, regardless of head dimension |
| Softmax shift invariance | Justifies the numerically-stable $\mathrm{softmax}(z-\max z)$ implementation as an exact, not approximate, rewrite |
| Causal masking | $-\infty$ scores $\Rightarrow$ exact zero post-softmax weight, preserving a valid probability distribution over a restricted (causal) support |
| Softmax row structure | Guarantees $Z=AV$ is a convex combination of value vectors — bounds what one attention step can produce to $\mathrm{conv}\{v_1,\dots,v_n\}$ |
| Sum-of-heads identity (§F.3.1) | Exact (proved via block matrix multiplication), not approximate, licence to analyze individual attention heads in isolation |
| Affine-composition-is-affine (§F.4.2) | Formal reason an MLP without a nonlinearity collapses to one linear map regardless of depth |
| MLP as key-value memory | Standard lens for interpreting individual neurons: key direction (row of $W_1$) vs. value direction (column of $W_2$) |
| Residual stream as additive sum | Proved by induction (§F.5.1); foundation of direct logit attribution — components' contributions to logits can be computed independently |
| Superposition | Quantified via near-orthogonality concentration (JL-style bound): number of near-orthogonal directions grows exponentially in $d_{\text{model}}$ |
| LayerNorm folding | $\gamma$-folding is an exact reparameterization; freezing $\sigma$ is an explicit approximation, not an identity |
| QK circuit ($W_QW_K^\top$) | Determines *where* a head attends, independent of what it copies |
| OV circuit ($W_VW_O$) | Determines *what* a head copies, independent of where it attends |
| Virtual heads (circuit composition) | Multi-layer head composition reduces to matrix multiplication of QK/OV circuits |

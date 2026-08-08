# Appendix A — Linear Algebra

## A.1 Introduction

This appendix is cited throughout the book (e.g. Appendix D §D.2.2, Appendix F §F.2.2 and §F.4.2) for specific facts — dot products as similarity, the spectral structure of symmetric matrices, the low-rank structure exposed by SVD — that are usually stated in a linear-algebra course but rarely restated precisely at the point of use. Each section below states the relevant object, its exact defining condition (not just a formula), and — where the fact is used elsewhere in the book to license an inferential step — a proof or proof sketch, since "why is this true" is what actually justifies reusing the fact in a new context.

Throughout, vectors are real column vectors unless noted, $\|x\| := \sqrt{x^\top x}$ denotes the Euclidean ($\ell_2$) norm, and $I$ denotes the identity matrix of whatever size context requires.

---

## A.2 Vectors, Dot Products, and Similarity

### A.2.1 Definition

For $x, y \in \mathbb{R}^d$, the dot product is $x^\top y = \sum_{i=1}^d x_i y_i$. It is bilinear (linear in each argument separately) and symmetric ($x^\top y = y^\top x$).

### A.2.2 Dot Product as Similarity

**Claim.** $x^\top y = \|x\|\,\|y\|\cos\theta$, where $\theta$ is the angle between $x$ and $y$; consequently, for fixed norms $\|x\|,\|y\|$, the dot product is maximized exactly when $x$ and $y$ point in the same direction ($\theta=0$), is $0$ exactly when they are orthogonal ($\theta = \pi/2$, §A.5), and is minimized (most negative) exactly when they point in opposite directions ($\theta=\pi$).

**Proof.** By the law of cosines applied to the triangle with sides $x$, $y$, $x-y$: $\|x-y\|^2 = \|x\|^2+\|y\|^2 - 2\|x\|\|y\|\cos\theta$. Expanding the left side algebraically, $\|x-y\|^2 = (x-y)^\top(x-y) = x^\top x - 2x^\top y + y^\top y = \|x\|^2 + \|y\|^2 - 2x^\top y$. Equating the two expressions for $\|x-y\|^2$ and canceling $\|x\|^2+\|y\|^2$ from both sides gives $-2x^\top y = -2\|x\|\|y\|\cos\theta$, i.e. $x^\top y = \|x\|\|y\|\cos\theta$. $\blacksquare$

This is the fact underlying every "dot product measures similarity" claim elsewhere in the book (attention scores, MLP neuron pre-activations, §A.2.2 as cited in Appendix F §F.2.2 and §F.4.2): it holds *exactly*, for any two nonzero vectors, with no distributional assumption — the distributional reasoning used in, e.g., Appendix F §F.2.2 to compute the *typical size* of a dot product under random initialization is a separate, additional argument layered on top of this exact geometric identity, not a restatement of it. Note the dot product conflates magnitude and direction: $x^\top y$ can be large either because $\theta$ is small or because $\|x\|,\|y\|$ are large, which is why *cosine similarity*, $\cos\theta = x^\top y/(\|x\|\|y\|)$, is used when only directional alignment (not magnitude) is of interest.

---

## A.3 Matrix Multiplication

For $A \in \mathbb{R}^{m\times n}$ and $B \in \mathbb{R}^{n\times k}$, the product $AB \in \mathbb{R}^{m\times k}$ is defined entrywise by $(AB)_{ij} = \sum_{p=1}^n A_{ip}B_{pj}$ — the inner dimensions ($n$) must match, and the result takes the outer dimensions ($m\times k$). Two properties used repeatedly elsewhere in the book:

- **Associativity**, $(AB)C = A(BC)$, which holds for any conformable $A,B,C$ (a direct consequence of the defining sum being independent of grouping) and is exactly what licenses collapsing a chain of linear maps into one matrix — e.g. Appendix F §F.7.1's reduction of $W_QW_K^\top$ and $W_VW_O$ into single fixed matrices relies on associativity to justify that the *order* in which the constituent products are computed doesn't change the result, only the cost of computing it.
- **Non-commutativity in general**: $AB \ne BA$ even when both products are defined (e.g. both square and same size) — matrix multiplication represents composition of linear maps, $x \mapsto A(Bx)$, and function composition is not commutative in general either. This is why the order of matrices in every identity throughout the book ($W_Q W_K^\top$, not $W_K^\top W_Q$; $XW_Q$, not $W_QX$) is load-bearing, not stylistic.
- **Interpretation**: $AB$ represents the composite linear map obtained by first applying the map $B$ represents, then the map $A$ represents, when vectors are treated as columns acted on by left-multiplication ($x \mapsto Bx \mapsto A(Bx) = (AB)x$); when vectors are treated as rows acted on by right-multiplication (the convention used for $Q=XW_Q$ throughout Appendix F, since $X$'s rows are individual token vectors), the composite of "apply $W_Q$ then $W_K^\top$" is $XW_QW_K^\top$, with the maps applied left-to-right matching the order they're written — the two conventions are transposes of each other and neither is more "correct," but a derivation must fix one and stay consistent.

---

## A.4 Eigenvectors and Eigenvalues

### A.4.1 Definition

For $A \in \mathbb{R}^{n\times n}$, a nonzero vector $v \in \mathbb{C}^n$ is an **eigenvector** with **eigenvalue** $\lambda \in \mathbb{C}$ if $Av = \lambda v$ — applying $A$ to $v$ scales $v$ without changing its direction (or, if $\lambda<0$, reverses it; if $\lambda$ is complex, $v$ typically has complex entries even when $A$ is real). The requirement $v \ne 0$ is essential to the definition: $v=0$ trivially satisfies $Av=\lambda v$ for *every* $\lambda$, so allowing it would make "eigenvalue" a vacuous notion.

Eigenvalues are exactly the roots of the **characteristic polynomial** $\det(A-\lambda I) = 0$: if $Av=\lambda v$ for some $v\ne0$, then $(A-\lambda I)v = 0$ has a nonzero solution, which happens exactly when $A-\lambda I$ is singular, i.e. $\det(A-\lambda I)=0$. This is a degree-$n$ polynomial in $\lambda$, so by the fundamental theorem of algebra it has exactly $n$ roots counted with multiplicity — every $n\times n$ real or complex matrix has exactly $n$ eigenvalues (with multiplicity) over $\mathbb{C}$, though a real matrix's eigenvalues need not be real (e.g. a rotation matrix in $\mathbb{R}^2$ has eigenvalues $\cos\theta \pm i\sin\theta$).

### A.4.2 The Symmetric Case

**Claim.** If $A=A^\top$ (real symmetric), every eigenvalue of $A$ is real.

**Proof sketch.** Let $Av=\lambda v$ with $v\ne0$ possibly complex, and let $\bar v$ denote the entrywise complex conjugate. Then $\bar v^\top A v = \lambda \bar v^\top v$. Since $A$ is real and symmetric, $\bar v^\top A v = \overline{v^\top A \bar v}$... more directly: $\bar v^\top A v = (A\bar v)^\top v$ using $A^\top=A$ and realness of $A$, and taking the conjugate of $Av=\lambda v$ gives $A\bar v = \bar\lambda \bar v$, so $\bar v^\top A v = \bar\lambda \bar v^\top v$. Combining $\bar v^\top A v = \lambda \bar v^\top v$ (from the original equation) with $\bar v^\top A v = \bar \lambda \bar v^\top v$ gives $(\lambda - \bar\lambda)\bar v^\top v = 0$. Since $v \ne 0$, $\bar v^\top v = \|v\|^2 > 0$, so $\lambda = \bar\lambda$, i.e. $\lambda \in \mathbb{R}$. $\blacksquare$

This is the **spectral theorem** setup: a real symmetric $n\times n$ matrix not only has all-real eigenvalues but additionally admits an *orthonormal* basis of eigenvectors, i.e. $A = Q\Lambda Q^\top$ with $Q$ orthogonal (§A.5) and $\Lambda$ real diagonal — the proof of the full theorem is beyond this appendix's scope, but the real-eigenvalue fact proved above is the essential first step, and the full statement is what §A.9 (PSD matrices) and the "eigenvalues of $W_{QK}, W_{OV}$" discussions elsewhere in the book rely on. **A general (non-symmetric) real matrix, such as the QK or OV circuits of Appendix F §F.7.1, is not guaranteed real eigenvalues or an orthonormal eigenbasis at all** — $W_{QK}=W_QW_K^\top$ is not symmetric in general (it need not even be square if $d_k \ne d_{\text{model}}$ in a generalized sense, though as defined in Appendix F it is $d_{\text{model}}\times d_{\text{model}}$), so eigen-analysis of attention circuits must be treated with this caveat, or an SVD (§A.7) used instead, which imposes no such symmetry requirement.

---

## A.5 Orthogonality

**Definition.** $x,y \ne 0$ are **orthogonal** if $x^\top y = 0$; equivalently, by §A.2.2, if $\cos\theta = 0$, i.e. the angle between them is $\pi/2$. (The zero vector is conventionally said to be orthogonal to every vector, since $0^\top y = 0$ always, but this is a convention needed to make "orthogonal complement" well-behaved as a subspace, not a claim that $0$ has a well-defined "direction" perpendicular to anything.)

**Claim (Pythagorean theorem).** If $x^\top y=0$, then $\|x+y\|^2 = \|x\|^2+\|y\|^2$.

**Proof.** $\|x+y\|^2 = (x+y)^\top(x+y) = x^\top x + 2x^\top y + y^\top y = \|x\|^2 + 0 + \|y\|^2$. $\blacksquare$

**Claim.** A set of nonzero pairwise-orthogonal vectors $\{v_1,\ldots,v_k\}$ is linearly independent.

**Proof.** Suppose $\sum_i c_iv_i = 0$. Taking the dot product of both sides with $v_j$: $\sum_i c_i (v_i^\top v_j) = 0$. Every term with $i\ne j$ vanishes by orthogonality, leaving $c_j\|v_j\|^2 = 0$; since $v_j\ne 0$, $\|v_j\|^2>0$, forcing $c_j=0$. This holds for every $j$, so all coefficients are $0$ — the only linear combination equal to $0$ is the trivial one. $\blacksquare$

A set of pairwise-orthogonal vectors that are additionally unit-norm ($\|v_i\|=1$) is **orthonormal**; the claim above shows any orthonormal set of size $k$ in $\mathbb{R}^d$ is a basis for a $k$-dimensional subspace, and no orthonormal set in $\mathbb{R}^d$ can have more than $d$ vectors (a corollary of linear independence bounding set size by the ambient dimension) — the exact fact that Appendix F §F.5.2 relaxes to "*near*-orthogonal" in order to argue that superposition can host more directions than $d_{\text{model}}$ by trading exactness for approximate orthogonality.

---

## A.6 Projection

For a matrix $U \in \mathbb{R}^{d\times k}$ whose columns $\{u_1,\ldots,u_k\}$ are orthonormal (so $U^\top U = I_k$, the $k\times k$ identity — *not* generally $UU^\top = I_d$ unless $k=d$), define $P := UU^\top \in \mathbb{R}^{d\times d}$.

**Claim.** $P$ is the orthogonal projection onto $\mathrm{span}\{u_1,\ldots,u_k\}$: $P$ is symmetric, idempotent ($P^2=P$), and for any $x$, $Px$ is the closest point in $\mathrm{span}\{u_1,\ldots,u_k\}$ to $x$.

**Proof.**
- *Symmetric*: $P^\top = (UU^\top)^\top = U U^\top = P$.
- *Idempotent*: $P^2 = UU^\top UU^\top = U(U^\top U)U^\top = UI_kU^\top = UU^\top = P$, using $U^\top U = I_k$.
- *Minimizes distance*: write $x = Px + (x-Px)$. For any $y \in \mathrm{span}\{u_i\}$, $y = Uc$ for some $c\in\mathbb{R}^k$, and $(x-Px)^\top(Px-y) = (x-Px)^\top U(U^\top x - c)$; but $U^\top(x-Px) = U^\top x - U^\top UU^\top x = U^\top x - U^\top x = 0$ using $U^\top U=I_k$ again, so $(x-Px)^\top U = 0$, making the cross term vanish. Hence $\|x-y\|^2 = \|x-Px\|^2 + \|Px-y\|^2 \ge \|x-Px\|^2$ by the Pythagorean theorem (§A.5) applied to the now-orthogonal pair $(x-Px)$ and $(Px-y)$, with equality exactly when $y=Px$. $\blacksquare$

**Corollary (eigenvalues of a projection).** Since $P^2=P$, any eigenvalue $\lambda$ of $P$ satisfies $\lambda^2=\lambda$ (apply $P$ to $Pv=\lambda v$ to get $P^2v = \lambda Pv$, i.e. $Pv = \lambda^2 v$, and also $Pv=\lambda v$, so $\lambda v = \lambda^2 v$, giving $\lambda=\lambda^2$ since $v\ne0$), so every eigenvalue of a projection matrix is exactly $0$ or $1$ — geometrically, $P$ acts as the identity on the subspace it projects onto and as zero on the orthogonal complement.

> **MI connection**: "Projecting out a direction" (e.g. ablating a suspected concept direction $u$ from the residual stream) is exactly $x \mapsto x - uu^\top x = (I-uu^\top)x$ for a unit vector $u$ — the identity map minus the rank-1 projection onto $u$; the corollary above is why this operation exactly zeroes the component along $u$ and leaves the orthogonal complement of $u$ exactly unchanged, rather than merely "reducing" both.

---

## A.7 Singular Value Decomposition

**Theorem (existence).** Every $A \in \mathbb{R}^{m\times n}$ (no symmetry or square-ness required) can be written $A = U\Sigma V^\top$, where $U\in\mathbb{R}^{m\times m}$ and $V\in\mathbb{R}^{n\times n}$ are orthogonal ($U^\top U = UU^\top = I_m$, likewise for $V$), and $\Sigma \in \mathbb{R}^{m\times n}$ is diagonal (in the rectangular sense: $\Sigma_{ij}=0$ for $i\ne j$) with non-negative entries $\sigma_1 \ge \sigma_2 \ge \cdots \ge 0$ (the **singular values**) on the diagonal. (Full proof omitted; it follows from applying the spectral theorem, §A.4.2, to the symmetric matrix $A^\top A$.)

The connection to §A.4.2 is exact and worth stating: $A^\top A = V\Sigma^\top U^\top U \Sigma V^\top = V(\Sigma^\top\Sigma)V^\top$ using $U^\top U = I$, so the columns of $V$ are the eigenvectors of the symmetric matrix $A^\top A$, and $\sigma_i^2$ are its eigenvalues — this is both *why* the decomposition exists at all (§A.4.2 guarantees $A^\top A$, being symmetric, has an orthonormal eigenbasis and real, and here additionally non-negative, since $x^\top A^\top A x = \|Ax\|^2\ge0$, eigenvalues) and the standard way SVD is computed in practice.

**Low-rank approximation (Eckart–Young).** Writing $A = \sum_{i=1}^{\min(m,n)}\sigma_i u_iv_i^\top$ (the sum of rank-1 terms recovered by expanding $U\Sigma V^\top$ column-by-column), the best rank-$r$ approximation to $A$ in Frobenius norm (§A.8) is obtained by truncating this sum to its $r$ largest-$\sigma_i$ terms, $A_r := \sum_{i=1}^r \sigma_i u_iv_i^\top$, with approximation error $\|A-A_r\|_F = \sqrt{\sum_{i>r}\sigma_i^2}$ (stated without proof).

> **MI connection**: this is the formal basis for claims like "this weight matrix is effectively low-rank" or "this circuit is well-approximated by its top-$k$ singular directions" — a rapid decay in $\sigma_i$ is precisely the condition under which a low-rank $A_r$ captures most of $A$'s action, in the exact quantitative sense $\|A-A_r\|_F^2 = \sum_{i>r}\sigma_i^2$ above, not merely an informal sense of "most of the important directions."

---

## A.8 Trace and Frobenius Norm

**Definitions.** $\mathrm{tr}(A) = \sum_i A_{ii}$ (defined for square $A$); $\|A\|_F = \sqrt{\sum_{i,j}A_{ij}^2}$ (defined for any shape).

**Claim (cyclic property).** For conformable $A,B,C$, $\mathrm{tr}(ABC) = \mathrm{tr}(BCA) = \mathrm{tr}(CAB)$ (cyclic permutations only — $\mathrm{tr}(ABC) \ne \mathrm{tr}(ACB)$ in general).

**Proof for two matrices** ($\mathrm{tr}(AB)=\mathrm{tr}(BA)$, the base case the general cyclic property reduces to by grouping): $\mathrm{tr}(AB) = \sum_i(AB)_{ii} = \sum_i\sum_j A_{ij}B_{ji} = \sum_j\sum_i B_{ji}A_{ij} = \sum_j(BA)_{jj} = \mathrm{tr}(BA)$, swapping the order of summation. $\blacksquare$

**Claim.** $\mathrm{tr}(A) = \sum_i \lambda_i$, the sum of $A$'s eigenvalues (with multiplicity), and $\|A\|_F^2 = \mathrm{tr}(A^\top A) = \sum_i \sigma_i^2$, the sum of squared singular values.

**Proof of the second identity.** $\mathrm{tr}(A^\top A) = \sum_i (A^\top A)_{ii} = \sum_i\sum_j (A^\top)_{ij}A_{ji} = \sum_i\sum_j A_{ji}^2 = \sum_{i,j}A_{ij}^2 = \|A\|_F^2$ (reindexing), which establishes $\|A\|_F^2 = \mathrm{tr}(A^\top A)$; that this further equals $\sum_i\sigma_i^2$ follows by substituting the SVD $A=U\Sigma V^\top$ from §A.7, using $A^\top A = V\Sigma^\top\Sigma V^\top$ and the cyclic property to get $\mathrm{tr}(A^\top A) = \mathrm{tr}(\Sigma^\top\Sigma V^\top V) = \mathrm{tr}(\Sigma^\top\Sigma) = \sum_i \sigma_i^2$, using $V^\top V = I$. $\blacksquare$ (The eigenvalue identity for $\mathrm{tr}(A)$ itself follows analogously from expanding $A$ in a basis where it is triangular, e.g. via Schur decomposition; omitted here.)

This gives an exact, basis-independent way to measure a matrix's overall "size" ($\|A\|_F$) purely from its singular values, and connects directly to §A.7: the Eckart–Young error bound $\|A-A_r\|_F=\sqrt{\sum_{i>r}\sigma_i^2}$ is exactly this trace/Frobenius identity applied to the residual $A-A_r$.

---

## A.9 Positive Semi-Definite Matrices

**Definition.** A symmetric matrix $A=A^\top \in \mathbb{R}^{n\times n}$ is **positive semi-definite** (PSD), written $A \succeq 0$, if $x^\top A x \ge 0$ for every $x\in\mathbb{R}^n$. It is (strictly) **positive definite** if $x^\top Ax > 0$ for every $x \ne 0$.

**Claim (equivalent characterizations).** For symmetric $A$, the following are equivalent: (i) $A \succeq 0$; (ii) every eigenvalue of $A$ is $\ge 0$ (well-defined since $A$ is symmetric — §A.4.2 guarantees real eigenvalues); (iii) $A = B^\top B$ for some matrix $B$.

**Proof sketch.** (ii)$\Rightarrow$(iii): by the spectral theorem, $A = Q\Lambda Q^\top$ with $\Lambda = \mathrm{diag}(\lambda_1,\ldots,\lambda_n)$, $\lambda_i\ge0$; set $B = \Lambda^{1/2}Q^\top$ (entrywise square root of the non-negative diagonal), so $B^\top B = Q\Lambda^{1/2}\Lambda^{1/2}Q^\top = Q\Lambda Q^\top = A$. (iii)$\Rightarrow$(i): if $A=B^\top B$, then $x^\top Ax = x^\top B^\top Bx = (Bx)^\top(Bx) = \|Bx\|^2 \ge 0$ for every $x$. (i)$\Rightarrow$(ii): if $Av=\lambda v$ for a (real, by §A.4.2) eigenvalue $\lambda$ with eigenvector $v\ne0$, then $0 \le v^\top Av = v^\top(\lambda v) = \lambda\|v\|^2$, and since $\|v\|^2>0$, $\lambda \ge 0$. $\blacksquare$

Characterization (iii) is why every **Gram matrix** ($A = B^\top B$ for any $B$ — e.g. a covariance matrix, computed as $\frac{1}{n}X^\top X$ for centered data $X$) is automatically PSD without needing to check eigenvalues directly, and is the reason $A^\top A$ was usable inside the SVD existence argument of §A.7 (its eigenvalues, $\sigma_i^2$, being guaranteed non-negative is exactly what makes $\sigma_i := \sqrt{\lambda_i(A^\top A)}$ well-defined as a real, non-negative number).

---

## A.10 Summary Table

| Object | Defining condition | Key fact used elsewhere |
|---|---|---|
| Dot product ($x^\top y$) | $\sum_i x_iy_i$ | $=\|x\|\|y\|\cos\theta$ exactly (§A.2.2) — the geometric basis for "dot product as similarity" |
| Matrix product ($AB$) | $(AB)_{ij}=\sum_p A_{ip}B_{pj}$ | Associative but not commutative; order encodes composition order (§A.3) |
| Eigenpair ($Av=\lambda v$) | $v\ne0$, $\det(A-\lambda I)=0$ | Real for symmetric $A$ (§A.4.2); not guaranteed real/orthogonal-eigenbasis for general (e.g. QK/OV) matrices |
| Orthogonality ($x^\top y=0$) | — | Pairwise-orthogonal nonzero vectors are linearly independent (§A.5) — the exact fact "near-orthogonality" in superposition (App. F §F.5.2) relaxes |
| Projection ($P=UU^\top$) | $U$ orthonormal columns | $P$ symmetric, idempotent, eigenvalues $\in\{0,1\}$; $x-Px$ minimizes distance to $\mathrm{span}(U)$ (§A.6) |
| SVD ($A=U\Sigma V^\top$) | $U,V$ orthogonal, $\Sigma\ge0$ diagonal | Exists for *every* matrix (no symmetry needed); truncation gives the provably-optimal low-rank approximation (§A.7) |
| Trace / Frobenius norm | $\sum_iA_{ii}$ / $\sqrt{\sum_{ij}A_{ij}^2}$ | $\mathrm{tr}(A)=\sum\lambda_i$; $\|A\|_F^2=\mathrm{tr}(A^\top A)=\sum\sigma_i^2$ (§A.8) — links spectral and entrywise views of matrix size |
| PSD ($A\succeq0$) | $x^\top Ax\ge0\ \forall x$ | Equivalent to all eigenvalues $\ge0$, equivalent to $A=B^\top B$ (§A.9) — why Gram/covariance matrices are automatically PSD |

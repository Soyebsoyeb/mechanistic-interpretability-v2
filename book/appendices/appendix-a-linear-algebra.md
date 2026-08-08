# Appendix A — Linear Algebra for Mechanistic Interpretability

## A.1 Introduction

This appendix provides a self-contained review of the linear algebra used throughout this book. Unlike a standard textbook treatment, we emphasize *geometric intuition* and *connections to neural networks*.

If you can answer these questions, you're ready:
- What does a matrix *do* to a vector geometrically?
- Why are eigenvectors called "stable directions"?
- How does the SVD reveal a matrix's "true rank"?
- What does the trace have to do with how information flows through a layer?

### A.1.1 Conventions

Throughout this appendix:
- Vectors are column vectors; $v \in \mathbb{R}^n$ means $v$ is an $n \times 1$ matrix.
- Unless stated otherwise, all matrices have real entries. Where a claim depends on the field (e.g. existence of real eigenvalues), we say so explicitly.
- $\|\cdot\|$ without a subscript denotes the Euclidean ($\ell_2$) norm on vectors.
- "Symmetric" means $A = A^\top$; this is only defined for square matrices.
- Results stated for square $A \in \mathbb{R}^{n \times n}$ do **not** automatically extend to rectangular matrices unless noted (e.g. eigenvalues are defined only for square matrices, while singular values are defined for any $m \times n$ matrix).

---

## A.2 Vectors and Spaces

### A.2.1 Vectors as Points and Directions

A vector $v \in \mathbb{R}^n$ is an ordered list of $n$ numbers. In interpretability, we encounter vectors as:

- **Activations**: The output of a neuron or a residual stream at a given token position
- **Embeddings**: Word or token representations
- **Weight rows/columns**: The "input" or "output" directions of a linear layer

**Geometric view**: A vector is an arrow from the origin to a point in $n$-dimensional space.

> **Key intuition for MI**: When we say a feature is "represented" in a model, we mean there exists a *direction* $d \in \mathbb{R}^n$ in activation space such that projecting activations onto $d$ recovers that feature's strength.

### A.2.2 Dot Product and Similarity

The dot product between two vectors:

$$
x \cdot y = x^\top y = \sum_{i=1}^n x_i y_i
$$

**Geometric meaning**:

$$
x^\top y = \|x\| \|y\| \cos \theta
$$

where $\theta$ is the angle between them. This gives us:

- **Similarity**: When $\cos \theta = 1$ (parallel), vectors are maximally similar
- **Orthogonality**: When $\cos \theta = 0$ ($x^\top y = 0$), vectors are perpendicular and carry independent information

> **MI connection**: The attention mechanism computes dot products between query and key vectors. High dot product $\implies$ high attention $\implies$ information flows between those token positions.

**Example**:

Let

$$
x = \begin{bmatrix} 1 \\ 2 \end{bmatrix}, \quad y = \begin{bmatrix} 3 \\ 4 \end{bmatrix}
$$

Then:

$$
x^\top y = 1(3) + 2(4) = 11
$$

$$
\|x\| = \sqrt{5} \approx 2.236, \quad \|y\| = 5
$$

$$
\cos \theta = \frac{11}{2.236 \times 5} \approx 0.984 \implies \theta \approx 10.3^\circ
$$

The vectors are nearly parallel.

### A.2.3 Norms

The Euclidean norm (length) of a vector:

$$
\|v\|_2 = \sqrt{v^\top v} = \sqrt{\sum_{i=1}^n v_i^2}
$$

> **MI connection**: When we measure the "magnitude" of an activation vector, we're computing its norm. Large norms often correspond to high-confidence predictions or salient features.

---

## A.3 Matrices: Linear Transformations

### A.3.1 Definition and Dimensions

A matrix $A \in \mathbb{R}^{m \times n}$ maps vectors from $\mathbb{R}^n$ (input space) to $\mathbb{R}^m$ (output space):

$$
A: \mathbb{R}^n \to \mathbb{R}^m, \quad v \mapsto Av
$$

For matrix multiplication $AB$ to be defined:

$$
A \in \mathbb{R}^{m \times n}, \quad B \in \mathbb{R}^{n \times k} \implies AB \in \mathbb{R}^{m \times k}
$$

> **MI connection**:
> - A fully-connected layer: $h = Wx + b$ (here $W \in \mathbb{R}^{d_{\text{hidden}} \times d_{\text{input}}}$)
> - An attention head's $QK^\top$ product: dimensions $(n_{\text{queries}} \times d_{\text{head}}) \times (d_{\text{head}} \times n_{\text{keys}}) \to n_{\text{queries}} \times n_{\text{keys}}$

### A.3.2 Column Space and Range

The **column space** (range) of $A$ is all vectors that can be written as $Av$ for some $v$:

$$
\text{Col}(A) = \{Av : v \in \mathbb{R}^n\} \subseteq \mathbb{R}^m
$$

This is the span of the columns of $A$.

> **Key insight**: When you multiply a matrix by a vector, the output is always a linear combination of the columns. The columns define the *possible outputs* of the transformation.

### A.3.3 Null Space (Kernel)

The **null space** of $A$ is all vectors that map to zero:

$$
\text{Null}(A) = \{v \in \mathbb{R}^n : Av = 0\}
$$

> **MI connection**: If two different inputs differ only by a vector in the null space, the layer treats them identically. This is a form of *information loss* or *compression*.

### A.3.4 Matrix Multiplication as Composition

For matrices $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times k}$, the product $AB$ applies $B$ then $A$:

$$
(AB)v = A(Bv)
$$

> **MI connection**: A neural network is a composition of many linear transformations (interleaved with non-linearities). Understanding the composition helps trace information flow.

---

## A.4 Special Matrices and Operations

### A.4.1 Transpose

The transpose $A^\top \in \mathbb{R}^{n \times m}$ swaps rows and columns:

$$
(A^\top)_{ij} = A_{ji}
$$

Properties:

$$
(A^\top)^\top = A
$$

$$
(AB)^\top = B^\top A^\top
$$

$$
(Av) \cdot w = v \cdot (A^\top w)
$$

> **MI connection**: In attention, $QK^\top$ computes pairwise similarities. The transpose is what turns "queries dot keys" into a matrix of all pairwise scores.

### A.4.2 Identity Matrix

$$
I_n = \begin{bmatrix}
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1
\end{bmatrix}
$$

Properties: $AI = A$ and $IA = A$ for compatible dimensions.

### A.4.3 Orthogonal and Orthonormal Matrices

A matrix $U \in \mathbb{R}^{n \times n}$ is **orthogonal** if:

$$
U^\top U = UU^\top = I
$$

This means the columns (and rows) are orthonormal:

$$
u_i^\top u_j = \begin{cases}
1 & i = j \\
0 & i \neq j
\end{cases}
$$

**Geometric meaning**: Orthogonal matrices preserve lengths and angles. They represent rotations and reflections.

$$
\|Ux\| = \|x\|, \quad (Ux)^\top (Uy) = x^\top y
$$

> **MI connection**: Orthogonal weight matrices avoid the "dying neuron" problem and preserve gradient magnitudes during backpropagation.

---

## A.5 Eigenvalues and Eigenvectors

### A.5.1 Definition

For a square matrix $A \in \mathbb{R}^{n \times n}$, a nonzero vector $v$ is an **eigenvector** with corresponding **eigenvalue** $\lambda$ if:

$$
Av = \lambda v
$$

### A.5.2 Geometric Interpretation

The matrix $A$ acts on its eigenvector by *scaling* it by $\lambda$, without changing its direction.

- If $|\lambda| > 1$: The direction is stretched
- If $|\lambda| < 1$: The direction is compressed
- If $\lambda = 0$: The direction is killed (maps to zero)
- If $\lambda$ is complex: The eigenvector has complex entries ($v \in \mathbb{C}^n$) and comes with a conjugate partner $\bar\lambda, \bar v$; the pair spans a real 2-dimensional invariant subspace on which $A$ acts as a rotation combined with a scaling by $|\lambda|$. This cannot occur for symmetric real matrices (§A.5.4), whose eigenvalues are always real.

### A.5.3 The Characteristic Equation

Eigenvalues are found by solving:

$$
\det(A - \lambda I) = 0
$$

This polynomial in $\lambda$ has degree $n$, so $A$ has $n$ eigenvalues (counted with multiplicity).

### A.5.4 Spectral Theorem for Symmetric Matrices

If $A$ is symmetric ($A^\top = A$), then:
- All eigenvalues are real
- Eigenvectors corresponding to distinct eigenvalues are orthogonal
- $A$ has an orthonormal eigenbasis: $A = Q\Lambda Q^\top$ where $Q$ is orthogonal and $\Lambda$ is diagonal

### A.5.5 MI Connection: How Models Represent Features

Eigenvectors reveal the "natural directions" of a linear layer. When we say a model has learned to represent a feature, we often mean that:

1. The feature vector is approximately an eigenvector of some weight matrix
2. The eigenvalue tells us how strongly that feature is preserved or amplified through the layer
3. Large positive eigenvalues correspond to directions the model "cares about" (preserves through layers)

### A.5.6 Example

Let:

$$
A = \begin{bmatrix}
4 & 1 \\
1 & 4
\end{bmatrix}
$$

Characteristic equation:

$$
\det(A - \lambda I) = \det\begin{bmatrix}
4-\lambda & 1 \\
1 & 4-\lambda
\end{bmatrix}
= (4-\lambda)^2 - 1
= \lambda^2 - 8\lambda + 15
= (\lambda - 3)(\lambda - 5)
$$

Eigenvalues: $\lambda = 3, 5$

For $\lambda = 5$:

$$
(A - 5I)v = 0 \implies \begin{bmatrix}-1 & 1 \\ 1 & -1\end{bmatrix}v = 0 \implies v_1 = v_2
$$

so

$$
v = \frac{1}{\sqrt{2}}\begin{bmatrix}1 \\ 1\end{bmatrix}
$$

For $\lambda = 3$:

$$
(A - 3I)v = 0 \implies \begin{bmatrix}1 & 1 \\ 1 & 1\end{bmatrix}v = 0 \implies v_1 = -v_2
$$

so

$$
v = \frac{1}{\sqrt{2}}\begin{bmatrix}1 \\ -1\end{bmatrix}
$$

The matrix stretches the "sum" direction by $5\times$ and the "difference" direction by $3\times$.

---

## A.6 Singular Value Decomposition (SVD)

### A.6.1 Definition

Every matrix $W \in \mathbb{R}^{m \times n}$ can be factorized as:

$$
W = U \Sigma V^\top
$$

where:
- $U \in \mathbb{R}^{m \times m}$: orthogonal matrix of **left singular vectors**
- $\Sigma \in \mathbb{R}^{m \times n}$: diagonal matrix of **singular values** $\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r > 0$
- $V \in \mathbb{R}^{n \times n}$: orthogonal matrix of **right singular vectors**

The rank $r$ is the number of nonzero singular values.

**Existence and uniqueness.** This factorization exists for *every* real matrix $W$, with no assumptions on rank, shape, or invertibility (this is the content of the SVD existence theorem). The singular values $\sigma_1 \ge \cdots \ge \sigma_r$ are always uniquely determined by $W$. The singular vectors are *not* uniquely determined in general: if a singular value has multiplicity $>1$, its corresponding singular vectors can be rotated within the associated subspace, and any singular vector can be replaced by its negation (with a matching sign flip on the paired vector) without changing $U\Sigma V^\top$.

### A.6.2 Geometric Interpretation

The SVD reveals the *action* of $W$ as three steps:

1. Rotate/reflect using $V^\top$ (preserves lengths)
2. Scale each axis by $\sigma_i$ using $\Sigma$ (stretches or compresses)
3. Rotate/reflect using $U$ (preserves lengths)

### A.6.3 Relation to Eigenvalues

For square $W$, the singular values are the square roots of the eigenvalues of $W^\top W$:

$$
\sigma_i = \sqrt{\lambda_i(W^\top W)}
$$

If $W$ is symmetric positive definite, then $U = V$ and $\Sigma = \Lambda$, so the SVD equals the eigenvalue decomposition.

### A.6.4 Low-Rank Approximation (Eckart–Young–Mirsky)

Fix $1 \le k \le r$ and let $U_k \in \mathbb{R}^{m \times k}$, $V_k \in \mathbb{R}^{n \times k}$ denote the first $k$ columns of $U, V$, and $\Sigma_k = \text{diag}(\sigma_1, \ldots, \sigma_k)$. Define the truncated SVD:

$$
W_k = U_k \Sigma_k V_k^\top = \sum_{i=1}^k \sigma_i u_i v_i^\top
$$

**Theorem (Eckart–Young–Mirsky).** Among all matrices of rank at most $k$, $W_k$ minimizes the distance to $W$ in *both* the Frobenius norm and the spectral norm:

$$
W_k = \arg\min_{\text{rank}(B) \le k} \|W - B\|_F = \arg\min_{\text{rank}(B) \le k} \|W - B\|_2
$$

with the resulting errors given exactly by the discarded singular values:

$$
\|W - W_k\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}, \qquad \|W - W_k\|_2 = \sigma_{k+1}
$$

(if $k \ge r$, both errors are $0$). If $\sigma_k = \sigma_{k+1}$, the minimizer $W_k$ is not unique — the theorem guarantees $W_k$ is *a* minimizer, not the only one. The Frobenius-norm case is due to Eckart & Young (1936); Mirsky (1960) extended it to any unitarily invariant norm, which includes the spectral norm as a special case.

### A.6.5 MI Connection: Superposition and the SVD

This is *crucial* for mechanistic interpretability.

1. **Superposition**: When a model has more features than dimensions ($n > d$), the SVD reveals which directions are most "important" (largest singular values)
2. **Pruning**: The low-rank approximation tells us how much information we lose by removing dimensions
3. **Feature detection**: Left singular vectors $U$ correspond to *output features* (what the layer reads), right singular vectors $V$ correspond to *input features* (what the layer writes)
4. **Effective rank**: The number of singular values above noise threshold tells us the model's *true* capacity

### A.6.6 Example

Let:

$$
W = \begin{bmatrix}
3 & 0 \\
0 & 1
\end{bmatrix}
$$

This is already diagonal. SVD gives:

$$
U = I, \quad \Sigma = \begin{bmatrix}
3 & 0 \\
0 & 1
\end{bmatrix}, \quad V = I
$$

The singular values are $\sigma_1 = 3, \sigma_2 = 1$. The matrix stretches the $x$-axis by $3\times$ and the $y$-axis by $1\times$.

---

## A.7 Projections

### A.7.1 Definition

A matrix $P$ is a **projection** if $P^2 = P$ (idempotent).

### A.7.2 Orthogonal Projection onto a Subspace

For a subspace $S$ with orthonormal basis $U \in \mathbb{R}^{n \times k}$, the orthogonal projection onto $S$ is:

$$
P = U U^\top
$$

Properties:
- $P^2 = P$
- $P^\top = P$ (symmetric)
- For any $v$: $Pv \in S$ and $(v - Pv) \perp S$
- Projection onto the orthogonal complement: $I - P$

### A.7.3 Geometric Interpretation

Projection takes a vector and drops its component perpendicular to $S$, keeping only the component in $S$.

### A.7.4 MI Connection: Feature Extraction

> **MI connection**: When we use sparse autoencoders (SAEs) to extract features, we're finding directions $d_i$ and then projecting activations onto them:

$$
\text{feature strength}_i = d_i^\top \text{activation}
$$

> If the directions are orthonormal, this is exactly an orthogonal projection. If they overlap (non-orthogonal), we need to account for the covariance structure.

### A.7.5 Example

Let $S$ be the span of a single vector:

$$
S = \text{span}\left(\begin{bmatrix}1 \\ 0\end{bmatrix}\right), \quad U = \begin{bmatrix}1 \\ 0\end{bmatrix}
$$

Then:

$$
P = UU^\top = \begin{bmatrix}1 \\ 0\end{bmatrix} \begin{bmatrix}1 & 0\end{bmatrix} = \begin{bmatrix}
1 & 0 \\
0 & 0
\end{bmatrix}
$$

For a general vector $v$:

$$
v = \begin{bmatrix}a \\ b\end{bmatrix}
$$

$$
Pv = \begin{bmatrix}1 & 0 \\ 0 & 0\end{bmatrix} \begin{bmatrix}a \\ b\end{bmatrix} = \begin{bmatrix}a \\ 0\end{bmatrix}
$$

We drop the $b$ component.

---

## A.8 Trace

### A.8.1 Definition

The trace of a square matrix $A \in \mathbb{R}^{n \times n}$ is the sum of its diagonal entries:

$$
\text{tr}(A) = \sum_{i=1}^n A_{ii}
$$

### A.8.2 Key Properties

$$
\text{tr}(A + B) = \text{tr}(A) + \text{tr}(B)
$$

$$
\text{tr}(cA) = c\,\text{tr}(A)
$$

$$
\text{tr}(A^\top) = \text{tr}(A)
$$

$$
\text{tr}(AB) = \text{tr}(BA) \quad \text{(cyclic property)}
$$

$$
\text{tr}(ABC) = \text{tr}(BCA) = \text{tr}(CAB)
$$

### A.8.3 Relation to Eigenvalues

The trace equals the sum of eigenvalues:

$$
\text{tr}(A) = \sum_{i=1}^n \lambda_i
$$

The determinant equals the product of eigenvalues:

$$
\det(A) = \prod_{i=1}^n \lambda_i
$$

### A.8.4 MI Connection: Information Flow

1. **Attention self-weight**: Let $A = \text{softmax}(QK^\top / \sqrt{d_k}) \in \mathbb{R}^{n \times n}$ be the (row-stochastic) attention matrix for a sequence of length $n$, so each row sums to 1: $\sum_j A_{ij} = 1$. Its trace,

   $$
   \text{tr}(A) = \sum_{i=1}^n A_{ii},
   $$

   sums the diagonal entries, i.e. the weight each token places on *itself*. Since every entry satisfies $0 \le A_{ij} \le 1$, we have $0 \le \text{tr}(A) \le n$. A trace close to $n$ means the head is close to the identity map (each token mostly attends to itself, so little information is mixed across positions); a trace close to $0$ means the head routes information almost entirely to *other* positions. Note this is a statement about $A$ *after* the softmax — the raw pre-softmax scores $QK^\top$ do not have a comparably clean interpretation via their trace, since they are not row-stochastic.
2. **Weight decay**: For a weight matrix $W$ whose rows/columns are treated as samples, the trace of the associated covariance matrix, $\text{tr}(\text{Cov}(W))$, equals the sum of the per-dimension variances (Bienaymé-type identity) and appears in ridge-style penalties on the second moment of $W$.
3. **Effective capacity**: For a model with parameters $\theta$ and log-likelihood $\ell(\theta)$, the Fisher information matrix is $F(\theta) = \mathbb{E}\!\left[\nabla_\theta \ell(\theta)\,\nabla_\theta \ell(\theta)^\top\right]$, and $\text{tr}(F) = \sum_i \lambda_i(F)$ sums its (non-negative) eigenvalues. A commonly used scalar summary of "effective number of parameters" under an $\ell_2$ prior of strength $\alpha$ is $\text{tr}\!\left(F(F + \alpha I)^{-1}\right)$, which interpolates between $0$ (heavily regularized, most directions unused) and the parameter count (unregularized).
4. **Gradient flow**: $\text{tr}(W^\top W) = \|W\|_F^2$ exactly (§A.9.1), so any regularizer written as $\text{tr}(W^\top W)$ is literally weight decay on $W$; this is an identity, not an analogy.

### A.8.5 Example

For:

$$
A = \begin{bmatrix}2 & 5 \\ 1 & 3\end{bmatrix}
$$

$$
\text{tr}(A) = 2 + 3 = 5
$$

Eigenvalues: $\lambda = 1, 4$. Sum = 5 ✓

---

## A.9 Norms of Matrices

### A.9.1 Frobenius Norm

The Frobenius norm treats the matrix as a long vector:

$$
\|A\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n A_{ij}^2}
$$

Alternative expressions:

$$
\|A\|_F^2 = \text{tr}(A^\top A)
$$

$$
\|A\|_F^2 = \sum_{i=1}^r \sigma_i^2 \quad \text{(SVD)}
$$

$$
\|A\|_F^2 = \sum_{i=1}^n \|A e_i\|_2^2 \quad \text{(sum of column norms)}
$$

### A.9.2 Spectral Norm

The spectral norm (operator norm) is the largest singular value:

$$
\|A\|_2 = \sigma_{\max}(A) = \max_{\|x\|=1} \|Ax\|
$$

This measures the maximum stretching factor of the matrix.

### A.9.3 MI Connection

1. **Stability**: The spectral norm controls how much a layer can amplify inputs. Large spectral norms $\implies$ potential instability
2. **Weight initialization**: Initialization schemes (e.g., Xavier, He) use the Frobenius norm to set scaling
3. **Complexity measures**: The Frobenius norm is used in weight decay to penalize large weights
4. **Superposition**: The ratio $\|W\|_F / \|W\|_2$ tells us how "spread out" the singular values are—this relates to how many features can be in superposition

### A.9.4 Example

For:

$$
A = \begin{bmatrix}3 & 0 \\ 0 & 4\end{bmatrix}
$$

$$
\|A\|_F = \sqrt{3^2 + 0^2 + 0^2 + 4^2} = \sqrt{25} = 5
$$

$$
\|A\|_2 = \max(3, 4) = 4
$$

---

## A.10 Common Identities Reference

For quick reference:

$$
(A^\top)^{-1} = (A^{-1})^\top
$$

$$
(AB)^{-1} = B^{-1}A^{-1} \quad \text{(if invertible)}
$$

$$
\text{tr}(AB) = \text{tr}(BA)
$$

$$
\text{tr}(A^\top B) = \sum_{i,j} A_{ij} B_{ij} \quad \text{(Frobenius inner product)}
$$

$$
\|A\|_F^2 = \text{tr}(A^\top A)
$$

$$
\frac{\partial}{\partial A} \|A\|_F^2 = 2A
$$

$$
\frac{\partial}{\partial A} \text{tr}(BA) = B^\top
$$

$$
\frac{\partial}{\partial A} \text{tr}(A^\top B A) = (B + B^\top)A
$$

---

## A.11 Summary: MI-Relevant Linear Algebra Concepts

| Concept | MI Application |
|---------|----------------|
| Dot product | Attention scores, feature similarity |
| Orthogonality | Independent features, disentanglement |
| Eigenvectors | Stable directions, feature preservation |
| SVD | Superposition, low-rank structure, pruning |
| Projection | Feature extraction, sparse autoencoders |
| Trace | Information flow, capacity measures |
| Frobenius norm | Weight decay, complexity, superposition measure |
| Spectral norm | Stability, Lipschitz constants |

# Linear Algebra for Mechanistic Interpretability

## Introduction

This appendix provides a self-contained review of the linear algebra
used throughout this book. Unlike a standard textbook treatment, we
emphasize *geometric intuition* and *connections to neural networks*.

If you can answer these questions, you’re ready:

  - What does a matrix *do* to a vector geometrically?

  - Why are eigenvectors called “stable directions”?

  - How does the SVD reveal a matrix’s “true rank”?

  - What does the trace have to do with how information flows through a
    layer?

## Vectors and Spaces

### Vectors as Points and Directions

A vector \(v \in \mathbb{R}^n\) is an ordered list of \(n\) numbers. In
interpretability, we encounter vectors as:

  - **Activations**: The output of a neuron or a residual stream at a
    given token position

  - **Embeddings**: Word or token representations

  - **Weight rows/columns**: The “input” or “output” directions of a
    linear layer

**Geometric view**: A vector is an arrow from the origin to a point in
\(n\)-dimensional space.

When we say a feature is “represented” in a model, we mean there exists
a *direction* \(d \in \mathbb{R}^n\) in activation space such that
projecting activations onto \(d\) recovers that feature’s strength.

### Dot Product and Similarity

The dot product between two vectors:

\[x \cdot y = x^\top y = \sum_{i=1}^n x_i y_i\]

**Geometric meaning**:

\[x^\top y = \|x\| \|y\| \cos \theta\]

where \(\theta\) is the angle between them. This gives us:

  - **Similarity**: When \(\cos \theta = 1\) (parallel), vectors are
    maximally similar

  - **Orthogonality**: When \(\cos \theta = 0\) (\(x^\top y = 0\)),
    vectors are perpendicular and carry independent information

The attention mechanism computes dot products between query and key
vectors. High dot product \(\implies\) high attention \(\implies\)
information flows between those token positions.

Let
\[x = \begin{bmatrix}1 \\ 2\end{bmatrix}, \quad y = \begin{bmatrix}3 \\ 4\end{bmatrix}\]
Then: \[\begin{aligned}
x^\top y &= 1(3) + 2(4) = 11 \\
\|x\| &= \sqrt{5} \approx 2.236, \quad \|y\| = 5 \\
\cos \theta &= \frac{11}{2.236 \times 5} \approx 0.984 \implies \theta \approx 10.3^\circ\end{aligned}\]
The vectors are nearly parallel.

### Norms

The Euclidean norm (length) of a vector:

\[\|v\|_2 = \sqrt{v^\top v} = \sqrt{\sum_{i=1}^n v_i^2}\]

**MI connection**: When we measure the “magnitude” of an activation
vector, we’re computing its norm. Large norms often correspond to
high-confidence predictions or salient features.

## Matrices: Linear Transformations

### Definition and Dimensions

A matrix \(A \in \mathbb{R}^{m \times n}\) maps vectors from
\(\mathbb{R}^n\) (input space) to \(\mathbb{R}^m\) (output space):

\[A: \mathbb{R}^n \to \mathbb{R}^m, \quad v \mapsto Av\]

For matrix multiplication \(AB\) to be defined:

\[A \in \mathbb{R}^{m \times n}, \quad B \in \mathbb{R}^{n \times k} \implies AB \in \mathbb{R}^{m \times k}\]

  - A fully-connected layer: \(h = Wx + b\) (here
    \(W \in \mathbb{R}^{d_{\text{hidden}} \times d_{\text{input}}}\))

  - An attention head’s \(QK^\top\) product: dimensions
    \((n_\text{queries} \times d_\text{head}) \times (d_\text{head} \times n_\text{keys}) \to n_\text{queries} \times n_\text{keys}\)

### Column Space and Range

The **column space** (range) of \(A\) is all vectors that can be written
as \(Av\) for some \(v\):

\[\text{Col}(A) = \{Av : v \in \mathbb{R}^n\} \subseteq \mathbb{R}^m\]

This is the span of the columns of \(A\).

When you multiply a matrix by a vector, the output is always a linear
combination of the columns. The columns define the *possible outputs* of
the transformation.

### Null Space (Kernel)

The **null space** of \(A\) is all vectors that map to zero:

\[\text{Null}(A) = \{v \in \mathbb{R}^n : Av = 0\}\]

**MI connection**: If two different inputs differ only by a vector in
the null space, the layer treats them identically. This is a form of
*information loss* or *compression*.

### Matrix Multiplication as Composition

For matrices \(A \in \mathbb{R}^{m \times n}\) and
\(B \in \mathbb{R}^{n \times k}\), the product \(AB\) applies \(B\) then
\(A\):

\[(AB)v = A(Bv)\]

**MI connection**: A neural network is a composition of many linear
transformations (interleaved with non-linearities). Understanding the
composition helps trace information flow.

## Special Matrices and Operations

### Transpose

The transpose \(A^\top \in \mathbb{R}^{n \times m}\) swaps rows and
columns:

\[(A^\top)_{ij} = A_{ji}\]

Properties: \[\begin{aligned}
(A^\top)^\top &= A \\
(AB)^\top &= B^\top A^\top \\
(Av) \cdot w &= v \cdot (A^\top w)\end{aligned}\]

**MI connection**: In attention, \(QK^\top\) computes pairwise
similarities. The transpose is what turns “queries dot keys” into a
matrix of all pairwise scores.

### Identity Matrix

\[I_n = \begin{bmatrix}
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1
\end{bmatrix}\]

Properties: \(AI = A\) and \(IA = A\) for compatible dimensions.

### Orthogonal and Orthonormal Matrices

A matrix \(U \in \mathbb{R}^{n \times n}\) is **orthogonal** if:

\[U^\top U = UU^\top = I\]

This means the columns (and rows) are orthonormal:
\[u_i^\top u_j = \begin{cases}
1 & i = j \\
0 & i \neq j
\end{cases}\]

**Geometric meaning**: Orthogonal matrices preserve lengths and angles.
They represent rotations and reflections.

\[\|Ux\| = \|x\|, \quad (Ux)^\top (Uy) = x^\top y\]

**MI connection**: Orthogonal weight matrices avoid the “dying neuron”
problem and preserve gradient magnitudes during backpropagation.

## Eigenvalues and Eigenvectors

### Definition

For a square matrix \(A \in \mathbb{R}^{n \times n}\), a nonzero vector
\(v\) is an **eigenvector** with corresponding **eigenvalue**
\(\lambda\) if:

\[Av = \lambda v\]

### Geometric Interpretation

The matrix \(A\) acts on its eigenvector by *scaling* it by \(\lambda\),
without changing its direction.

  - If \(|\lambda| > 1\): The direction is stretched

  - If \(|\lambda| < 1\): The direction is compressed

  - If \(\lambda = 0\): The direction is killed (maps to zero)

  - If \(\lambda\) is complex: The direction rotates (not possible for
    symmetric real matrices)

### The Characteristic Equation

Eigenvalues are found by solving:

\[\det(A - \lambda I) = 0\]

This polynomial in \(\lambda\) has degree \(n\), so \(A\) has \(n\)
eigenvalues (counted with multiplicity).

### Spectral Theorem for Symmetric Matrices

If \(A\) is symmetric (\(A^\top = A\)), then:

  - All eigenvalues are real

  - Eigenvectors corresponding to distinct eigenvalues are orthogonal

  - \(A\) has an orthonormal eigenbasis: \(A = Q\Lambda Q^\top\) where
    \(Q\) is orthogonal and \(\Lambda\) is diagonal

### MI Connection: How Models Represent Features

Eigenvectors reveal the “natural directions” of a linear layer. When we
say a model has learned to represent a feature, we often mean that:

1.  The feature vector is approximately an eigenvector of some weight
    matrix

2.  The eigenvalue tells us how strongly that feature is preserved or
    amplified through the layer

3.  Large positive eigenvalues correspond to directions the model “cares
    about” (preserves through layers)

### Example

Let: \[A = \begin{bmatrix}
4 & 1 \\
1 & 4
\end{bmatrix}\]

Characteristic equation: \[\begin{aligned}
\det(A - \lambda I) &= \det\begin{bmatrix}
4-\lambda & 1 \\
1 & 4-\lambda
\end{bmatrix} \\
&= (4-\lambda)^2 - 1 \\
&= \lambda^2 - 8\lambda + 15 \\
&= (\lambda - 3)(\lambda - 5)\end{aligned}\]

Eigenvalues: \(\lambda = 3, 5\)

For \(\lambda = 5\):
\((A - 5I)v = 0 \implies \begin{bmatrix}-1 & 1 \\ 1 & -1\end{bmatrix}v = 0 \implies v_1 = v_2\),
so \(v = \frac{1}{\sqrt{2}}\begin{bmatrix}1 \\ 1\end{bmatrix}\)

For \(\lambda = 3\):
\((A - 3I)v = 0 \implies \begin{bmatrix}1 & 1 \\ 1 & 1\end{bmatrix}v = 0 \implies v_1 = -v_2\),
so \(v = \frac{1}{\sqrt{2}}\begin{bmatrix}1 \\ -1\end{bmatrix}\)

The matrix stretches the “sum” direction by \(5\times\) and the
“difference” direction by \(3\times\).

## Singular Value Decomposition (SVD)

### Definition

Every matrix \(W \in \mathbb{R}^{m \times n}\) can be factorized as:

\[W = U \Sigma V^\top\]

where:

  - \(U \in \mathbb{R}^{m \times m}\): orthogonal matrix of **left
    singular vectors**

  - \(\Sigma \in \mathbb{R}^{m \times n}\): diagonal matrix of
    **singular values**
    \(\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r > 0\)

  - \(V \in \mathbb{R}^{n \times n}\): orthogonal matrix of **right
    singular vectors**

The rank \(r\) is the number of nonzero singular values.

### Geometric Interpretation

The SVD reveals the *action* of \(W\) as three steps:

1.  Rotate/reflect using \(V^\top\) (preserves lengths)

2.  Scale each axis by \(\sigma_i\) using \(\Sigma\) (stretches or
    compresses)

3.  Rotate/reflect using \(U\) (preserves lengths)

### Relation to Eigenvalues

For square \(W\), the singular values are the square roots of the
eigenvalues of \(W^\top W\):

\[\sigma_i = \sqrt{\lambda_i(W^\top W)}\]

If \(W\) is symmetric positive definite, then \(U = V\) and
\(\Sigma = \Lambda\), so the SVD equals the eigenvalue decomposition.

### Low-Rank Approximation (Eckart-Young)

The best rank-\(k\) approximation to \(W\) (in Frobenius norm) is:

\[W_k = U_k \Sigma_k V_k^\top\]

where we keep only the top \(k\) singular values and vectors. The
approximation error is:

\[\|W - W_k\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}\]

### MI Connection: Superposition and the SVD

This is *crucial* for mechanistic interpretability.

1.  **Superposition**: When a model has more features than dimensions
    (\(n > d\)), the SVD reveals which directions are most “important”
    (largest singular values)

2.  **Pruning**: The low-rank approximation tells us how much
    information we lose by removing dimensions

3.  **Feature detection**: Left singular vectors \(U\) correspond to
    *output features* (what the layer reads), right singular vectors
    \(V\) correspond to *input features* (what the layer writes)

4.  **Effective rank**: The number of singular values above noise
    threshold tells us the model’s *true* capacity

### Example

Let: \[W = \begin{bmatrix}
3 & 0 \\
0 & 1
\end{bmatrix}\]

This is already diagonal. SVD gives:
\[U = I, \quad \Sigma = \begin{bmatrix}
3 & 0 \\
0 & 1
\end{bmatrix}, \quad V = I\]

The singular values are \(\sigma_1 = 3, \sigma_2 = 1\). The matrix
stretches the \(x\)-axis by \(3\times\) and the \(y\)-axis by
\(1\times\).

## Projections

### Definition

A matrix \(P\) is a **projection** if \(P^2 = P\) (idempotent).

### Orthogonal Projection onto a Subspace

For a subspace \(S\) with orthonormal basis
\(U \in \mathbb{R}^{n \times k}\), the orthogonal projection onto \(S\)
is:

\[P = U U^\top\]

Properties:

  - \(P^2 = P\)

  - \(P^\top = P\) (symmetric)

  - For any \(v\): \(Pv \in S\) and \((v - Pv) \perp S\)

  - Projection onto the orthogonal complement: \(I - P\)

### Geometric Interpretation

Projection takes a vector and drops its component perpendicular to
\(S\), keeping only the component in \(S\).

### MI Connection: Feature Extraction

When we use sparse autoencoders (SAEs) to extract features, we’re
finding directions \(d_i\) and then projecting activations onto them:

\[\text{feature\_strength}_i = d_i^\top \text{activation}\]

If the directions are orthonormal, this is exactly an orthogonal
projection. If they overlap (non-orthogonal), we need to account for the
covariance structure.

### Example

Let \(S = \text{span}\left(\begin{bmatrix}1 \\ 0\end{bmatrix}\right)\),
so \(U = \begin{bmatrix}1 \\ 0\end{bmatrix}\).

Then:
\[P = UU^\top = \begin{bmatrix}1 \\ 0\end{bmatrix} \begin{bmatrix}1 & 0\end{bmatrix} = \begin{bmatrix}
1 & 0 \\
0 & 0
\end{bmatrix}\]

For \(v = \begin{bmatrix}a \\ b\end{bmatrix}\):
\[Pv = \begin{bmatrix}1 & 0 \\ 0 & 0\end{bmatrix} \begin{bmatrix}a \\ b\end{bmatrix} = \begin{bmatrix}a \\ 0\end{bmatrix}\]

We drop the \(b\) component.

## Trace

### Definition

The trace of a square matrix \(A \in \mathbb{R}^{n \times n}\) is the
sum of its diagonal entries:

\[\text{tr}(A) = \sum_{i=1}^n A_{ii}\]

### Key Properties

\[\begin{aligned}
\text{tr}(A + B) &= \text{tr}(A) + \text{tr}(B) \\
\text{tr}(cA) &= c\,\text{tr}(A) \\
\text{tr}(A^\top) &= \text{tr}(A) \\
\text{tr}(AB) &= \text{tr}(BA) \quad \text{(cyclic property)} \\
\text{tr}(ABC) &= \text{tr}(BCA) = \text{tr}(CAB)\end{aligned}\]

### Relation to Eigenvalues

The trace equals the sum of eigenvalues:

\[\text{tr}(A) = \sum_{i=1}^n \lambda_i\]

The determinant equals the product of eigenvalues:
\[\det(A) = \prod_{i=1}^n \lambda_i\]

### MI Connection: Information Flow

1.  **Attention**: In attention, \(\text{tr}(QK^\top)\) (or more
    precisely, \(\text{tr}(\text{softmax}(QK^\top))\)) relates to how
    much information flows from keys to queries

2.  **Weight decay**: The trace of a weight matrix’s covariance appears
    in regularization terms

3.  **Effective capacity**: The trace of the Fisher information matrix
    relates to how many independent parameters the model can use

4.  **Gradient flow**: \(\text{tr}(W^\top W)\) is the squared Frobenius
    norm, often used as a regularizer

### Example

For \(A = \begin{bmatrix}2 & 5 \\ 1 & 3\end{bmatrix}\):
\[\text{tr}(A) = 2 + 3 = 5\]

Eigenvalues: \(\lambda = 1, 4\). Sum = 5 \(\checkmark\)

## Norms of Matrices

### Frobenius Norm

The Frobenius norm treats the matrix as a long vector:

\[\|A\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n A_{ij}^2}\]

Alternative expressions: \[\begin{aligned}
\|A\|_F^2 &= \text{tr}(A^\top A) \\
\|A\|_F^2 &= \sum_{i=1}^r \sigma_i^2 \quad \text{(SVD)} \\
\|A\|_F^2 &= \sum_{i=1}^n \|A e_i\|_2^2 \quad \text{(sum of column norms)}\end{aligned}\]

### Spectral Norm

The spectral norm (operator norm) is the largest singular value:

\[\|A\|_2 = \sigma_{\max}(A) = \max_{\|x\|=1} \|Ax\|\]

This measures the maximum stretching factor of the matrix.

### MI Connection

1.  **Stability**: The spectral norm controls how much a layer can
    amplify inputs. Large spectral norms \(\implies\) potential
    instability

2.  **Weight initialization**: Initialization schemes (e.g., Xavier, He)
    use the Frobenius norm to set scaling

3.  **Complexity measures**: The Frobenius norm is used in weight decay
    to penalize large weights

4.  **Superposition**: The ratio \(\|W\|_F / \|W\|_2\) tells us how
    “spread out” the singular values are—this relates to how many
    features can be in superposition

### Example

For \(A = \begin{bmatrix}3 & 0 \\ 0 & 4\end{bmatrix}\):
\[\begin{aligned}
\|A\|_F &= \sqrt{3^2 + 0^2 + 0^2 + 4^2} = \sqrt{25} = 5 \\
\|A\|_2 &= \max(3, 4) = 4\end{aligned}\]

## Common Identities Reference

For quick reference:

\[\begin{aligned}
(A^\top)^{-1} &= (A^{-1})^\top \\
(AB)^{-1} &= B^{-1}A^{-1} \quad \text{(if invertible)} \\
\text{tr}(AB) &= \text{tr}(BA) \\
\text{tr}(A^\top B) &= \sum_{i,j} A_{ij} B_{ij} \quad \text{(Frobenius inner product)} \\
\|A\|_F^2 &= \text{tr}(A^\top A) \\
\frac{\partial}{\partial A} \|A\|_F^2 &= 2A \\
\frac{\partial}{\partial A} \text{tr}(BA) &= B^\top \\
\frac{\partial}{\partial A} \text{tr}(A^\top B A) &= (B + B^\top)A\end{aligned}\]

## Summary: MI-Relevant Linear Algebra Concepts

| **Concept**    | **MI Application**                              |
| :------------- | :---------------------------------------------- |
| Dot product    | Attention scores, feature similarity            |
| Orthogonality  | Independent features, disentanglement           |
| Eigenvectors   | Stable directions, feature preservation         |
| SVD            | Superposition, low-rank structure, pruning      |
| Projection     | Feature extraction, sparse autoencoders         |
| Trace          | Information flow, capacity measures             |
| Frobenius norm | Weight decay, complexity, superposition measure |
| Spectral norm  | Stability, Lipschitz constants                  |

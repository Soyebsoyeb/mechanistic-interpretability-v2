\appendix
\chapter{Linear Algebra for Mechanistic Interpretability}

\section{Introduction}

This appendix provides a self-contained review of the linear algebra used throughout this book. Unlike a standard textbook treatment, we emphasize \emph{geometric intuition} and \emph{connections to neural networks}. 

If you can answer these questions, you're ready:
\begin{itemize}
    \item What does a matrix \emph{do} to a vector geometrically?
    \item Why are eigenvectors called ``stable directions''?
    \item How does the SVD reveal a matrix's ``true rank''?
    \item What does the trace have to do with how information flows through a layer?
\end{itemize}

\section{Vectors and Spaces}

\subsection{Vectors as Points and Directions}

A vector $v \in \mathbb{R}^n$ is an ordered list of $n$ numbers. In interpretability, we encounter vectors as:

\begin{itemize}
    \item \textbf{Activations}: The output of a neuron or a residual stream at a given token position
    \item \textbf{Embeddings}: Word or token representations
    \item \textbf{Weight rows/columns}: The ``input'' or ``output'' directions of a linear layer
\end{itemize}

\textbf{Geometric view}: A vector is an arrow from the origin to a point in $n$-dimensional space.

\begin{remark}
When we say a feature is ``represented'' in a model, we mean there exists a \emph{direction} $d \in \mathbb{R}^n$ in activation space such that projecting activations onto $d$ recovers that feature's strength.
\end{remark}

\subsection{Dot Product and Similarity}

The dot product between two vectors:

\begin{equation}
x \cdot y = x^\top y = \sum_{i=1}^n x_i y_i
\end{equation}

\textbf{Geometric meaning}:

\begin{equation}
x^\top y = \|x\| \|y\| \cos \theta
\end{equation}

where $\theta$ is the angle between them. This gives us:

\begin{itemize}
    \item \textbf{Similarity}: When $\cos \theta = 1$ (parallel), vectors are maximally similar
    \item \textbf{Orthogonality}: When $\cos \theta = 0$ ($x^\top y = 0$), vectors are perpendicular and carry independent information
\end{itemize}

\begin{example}[Attention Mechanism]
The attention mechanism computes dot products between query and key vectors. High dot product $\implies$ high attention $\implies$ information flows between those token positions.
\end{example}

\begin{example}[Numerical]
Let 
\begin{equation}
x = \begin{bmatrix}1 \\ 2\end{bmatrix}, \quad y = \begin{bmatrix}3 \\ 4\end{bmatrix}
\end{equation}
Then:
\begin{align}
x^\top y &= 1(3) + 2(4) = 11 \\
\|x\| &= \sqrt{5} \approx 2.236, \quad \|y\| = 5 \\
\cos \theta &= \frac{11}{2.236 \times 5} \approx 0.984 \implies \theta \approx 10.3^\circ
\end{align}
The vectors are nearly parallel.
\end{example}

\subsection{Norms}

The Euclidean norm (length) of a vector:

\begin{equation}
\|v\|_2 = \sqrt{v^\top v} = \sqrt{\sum_{i=1}^n v_i^2}
\end{equation}

\textbf{MI connection}: When we measure the ``magnitude'' of an activation vector, we're computing its norm. Large norms often correspond to high-confidence predictions or salient features.

\section{Matrices: Linear Transformations}

\subsection{Definition and Dimensions}

A matrix $A \in \mathbb{R}^{m \times n}$ maps vectors from $\mathbb{R}^n$ (input space) to $\mathbb{R}^m$ (output space):

\begin{equation}
A: \mathbb{R}^n \to \mathbb{R}^m, \quad v \mapsto Av
\end{equation}

For matrix multiplication $AB$ to be defined:

\begin{equation}
A \in \mathbb{R}^{m \times n}, \quad B \in \mathbb{R}^{n \times k} \implies AB \in \mathbb{R}^{m \times k}
\end{equation}

\begin{example}[Neural Network Layers]
\begin{itemize}
    \item A fully-connected layer: $h = Wx + b$ (here $W \in \mathbb{R}^{d_{\text{hidden}} \times d_{\text{input}}}$)
    \item An attention head's $QK^\top$ product: dimensions $(n_\text{queries} \times d_\text{head}) \times (d_\text{head} \times n_\text{keys}) \to n_\text{queries} \times n_\text{keys}$
\end{itemize}
\end{example}

\subsection{Column Space and Range}

The \textbf{column space} (range) of $A$ is all vectors that can be written as $Av$ for some $v$:

\begin{equation}
\text{Col}(A) = \{Av : v \in \mathbb{R}^n\} \subseteq \mathbb{R}^m
\end{equation}

This is the span of the columns of $A$.

\begin{remark}
When you multiply a matrix by a vector, the output is always a linear combination of the columns. The columns define the \emph{possible outputs} of the transformation.
\end{remark}

\subsection{Null Space (Kernel)}

The \textbf{null space} of $A$ is all vectors that map to zero:

\begin{equation}
\text{Null}(A) = \{v \in \mathbb{R}^n : Av = 0\}
\end{equation}

\textbf{MI connection}: If two different inputs differ only by a vector in the null space, the layer treats them identically. This is a form of \emph{information loss} or \emph{compression}.

\subsection{Matrix Multiplication as Composition}

For matrices $A \in \mathbb{R}^{m \times n}$ and $B \in \mathbb{R}^{n \times k}$, the product $AB$ applies $B$ then $A$:

\begin{equation}
(AB)v = A(Bv)
\end{equation}

\textbf{MI connection}: A neural network is a composition of many linear transformations (interleaved with non-linearities). Understanding the composition helps trace information flow.

\section{Special Matrices and Operations}

\subsection{Transpose}

The transpose $A^\top \in \mathbb{R}^{n \times m}$ swaps rows and columns:

\begin{equation}
(A^\top)_{ij} = A_{ji}
\end{equation}

Properties:
\begin{align}
(A^\top)^\top &= A \\
(AB)^\top &= B^\top A^\top \\
(Av) \cdot w &= v \cdot (A^\top w)
\end{align}

\textbf{MI connection}: In attention, $QK^\top$ computes pairwise similarities. The transpose is what turns ``queries dot keys'' into a matrix of all pairwise scores.

\subsection{Identity Matrix}

\begin{equation}
I_n = \begin{bmatrix}
1 & 0 & \cdots & 0 \\
0 & 1 & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & 1
\end{bmatrix}
\end{equation}

Properties: $AI = A$ and $IA = A$ for compatible dimensions.

\subsection{Orthogonal and Orthonormal Matrices}

A matrix $U \in \mathbb{R}^{n \times n}$ is \textbf{orthogonal} if:

\begin{equation}
U^\top U = UU^\top = I
\end{equation}

This means the columns (and rows) are orthonormal:
\begin{equation}
u_i^\top u_j = \begin{cases}
1 & i = j \\
0 & i \neq j
\end{cases}
\end{equation}

\textbf{Geometric meaning}: Orthogonal matrices preserve lengths and angles. They represent rotations and reflections.

\begin{equation}
\|Ux\| = \|x\|, \quad (Ux)^\top (Uy) = x^\top y
\end{equation}

\textbf{MI connection}: Orthogonal weight matrices avoid the ``dying neuron'' problem and preserve gradient magnitudes during backpropagation.

\section{Eigenvalues and Eigenvectors}

\subsection{Definition}

For a square matrix $A \in \mathbb{R}^{n \times n}$, a nonzero vector $v$ is an \textbf{eigenvector} with corresponding \textbf{eigenvalue} $\lambda$ if:

\begin{equation}
Av = \lambda v
\end{equation}

\subsection{Geometric Interpretation}

The matrix $A$ acts on its eigenvector by \emph{scaling} it by $\lambda$, without changing its direction.

\begin{itemize}
    \item If $|\lambda| > 1$: The direction is stretched
    \item If $|\lambda| < 1$: The direction is compressed
    \item If $\lambda = 0$: The direction is killed (maps to zero)
    \item If $\lambda$ is complex: The direction rotates (not possible for symmetric real matrices)
\end{itemize}

\subsection{The Characteristic Equation}

Eigenvalues are found by solving:

\begin{equation}
\det(A - \lambda I) = 0
\end{equation}

This polynomial in $\lambda$ has degree $n$, so $A$ has $n$ eigenvalues (counted with multiplicity).

\subsection{Spectral Theorem for Symmetric Matrices}

If $A$ is symmetric ($A^\top = A$), then:

\begin{itemize}
    \item All eigenvalues are real
    \item Eigenvectors corresponding to distinct eigenvalues are orthogonal
    \item $A$ has an orthonormal eigenbasis: $A = Q\Lambda Q^\top$ where $Q$ is orthogonal and $\Lambda$ is diagonal
\end{itemize}

\subsection{MI Connection: How Models Represent Features}

Eigenvectors reveal the ``natural directions'' of a linear layer. When we say a model has learned to represent a feature, we often mean that:

\begin{enumerate}
    \item The feature vector is approximately an eigenvector of some weight matrix
    \item The eigenvalue tells us how strongly that feature is preserved or amplified through the layer
    \item Large positive eigenvalues correspond to directions the model ``cares about'' (preserves through layers)
\end{enumerate}

\subsection{Example}

Let:
\begin{equation}
A = \begin{bmatrix}
4 & 1 \\
1 & 4
\end{bmatrix}
\end{equation}

Characteristic equation:
\begin{align}
\det(A - \lambda I) &= \det\begin{bmatrix}
4-\lambda & 1 \\
1 & 4-\lambda
\end{bmatrix} \\
&= (4-\lambda)^2 - 1 \\
&= \lambda^2 - 8\lambda + 15 \\
&= (\lambda - 3)(\lambda - 5)
\end{align}

Eigenvalues: $\lambda = 3, 5$

For $\lambda = 5$: $(A - 5I)v = 0 \implies \begin{bmatrix}-1 & 1 \\ 1 & -1\end{bmatrix}v = 0 \implies v_1 = v_2$, so $v = \frac{1}{\sqrt{2}}\begin{bmatrix}1 \\ 1\end{bmatrix}$

For $\lambda = 3$: $(A - 3I)v = 0 \implies \begin{bmatrix}1 & 1 \\ 1 & 1\end{bmatrix}v = 0 \implies v_1 = -v_2$, so $v = \frac{1}{\sqrt{2}}\begin{bmatrix}1 \\ -1\end{bmatrix}$

The matrix stretches the ``sum'' direction by $5\times$ and the ``difference'' direction by $3\times$.

\section{Singular Value Decomposition (SVD)}

\subsection{Definition}

Every matrix $W \in \mathbb{R}^{m \times n}$ can be factorized as:

\begin{equation}
W = U \Sigma V^\top
\end{equation}

where:
\begin{itemize}
    \item $U \in \mathbb{R}^{m \times m}$: orthogonal matrix of \textbf{left singular vectors}
    \item $\Sigma \in \mathbb{R}^{m \times n}$: diagonal matrix of \textbf{singular values} $\sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r > 0$
    \item $V \in \mathbb{R}^{n \times n}$: orthogonal matrix of \textbf{right singular vectors}
\end{itemize}

The rank $r$ is the number of nonzero singular values.

\subsection{Geometric Interpretation}

The SVD reveals the \emph{action} of $W$ as three steps:

\begin{enumerate}
    \item Rotate/reflect using $V^\top$ (preserves lengths)
    \item Scale each axis by $\sigma_i$ using $\Sigma$ (stretches or compresses)
    \item Rotate/reflect using $U$ (preserves lengths)
\end{enumerate}

\subsection{Relation to Eigenvalues}

For square $W$, the singular values are the square roots of the eigenvalues of $W^\top W$:

\begin{equation}
\sigma_i = \sqrt{\lambda_i(W^\top W)}
\end{equation}

If $W$ is symmetric positive definite, then $U = V$ and $\Sigma = \Lambda$, so the SVD equals the eigenvalue decomposition.

\subsection{Low-Rank Approximation (Eckart-Young)}

The best rank-$k$ approximation to $W$ (in Frobenius norm) is:

\begin{equation}
W_k = U_k \Sigma_k V_k^\top
\end{equation}

where we keep only the top $k$ singular values and vectors. The approximation error is:

\begin{equation}
\|W - W_k\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}
\end{equation}

\subsection{MI Connection: Superposition and the SVD}

This is \emph{crucial} for mechanistic interpretability.

\begin{enumerate}
    \item \textbf{Superposition}: When a model has more features than dimensions ($n > d$), the SVD reveals which directions are most ``important'' (largest singular values)
    \item \textbf{Pruning}: The low-rank approximation tells us how much information we lose by removing dimensions
    \item \textbf{Feature detection}: Left singular vectors $U$ correspond to \emph{output features} (what the layer reads), right singular vectors $V$ correspond to \emph{input features} (what the layer writes)
    \item \textbf{Effective rank}: The number of singular values above noise threshold tells us the model's \emph{true} capacity
\end{enumerate}

\subsection{Example}

Let:
\begin{equation}
W = \begin{bmatrix}
3 & 0 \\
0 & 1
\end{bmatrix}
\end{equation}

This is already diagonal. SVD gives:
\begin{equation}
U = I, \quad \Sigma = \begin{bmatrix}
3 & 0 \\
0 & 1
\end{bmatrix}, \quad V = I
\end{equation}

The singular values are $\sigma_1 = 3, \sigma_2 = 1$. The matrix stretches the $x$-axis by $3\times$ and the $y$-axis by $1\times$.

\section{Projections}

\subsection{Definition}

A matrix $P$ is a \textbf{projection} if $P^2 = P$ (idempotent). 

\subsection{Orthogonal Projection onto a Subspace}

For a subspace $S$ with orthonormal basis $U \in \mathbb{R}^{n \times k}$, the orthogonal projection onto $S$ is:

\begin{equation}
P = U U^\top
\end{equation}

Properties:
\begin{itemize}
    \item $P^2 = P$
    \item $P^\top = P$ (symmetric)
    \item For any $v$: $Pv \in S$ and $(v - Pv) \perp S$
    \item Projection onto the orthogonal complement: $I - P$
\end{itemize}

\subsection{Geometric Interpretation}

Projection takes a vector and drops its component perpendicular to $S$, keeping only the component in $S$.

\subsection{MI Connection: Feature Extraction}

When we use sparse autoencoders (SAEs) to extract features, we're finding directions $d_i$ and then projecting activations onto them:

\begin{equation}
\text{feature\_strength}_i = d_i^\top \text{activation}
\end{equation}

If the directions are orthonormal, this is exactly an orthogonal projection. If they overlap (non-orthogonal), we need to account for the covariance structure.

\subsection{Example}

Let $S = \text{span}\left(\begin{bmatrix}1 \\ 0\end{bmatrix}\right)$, so $U = \begin{bmatrix}1 \\ 0\end{bmatrix}$.

Then:
\begin{equation}
P = UU^\top = \begin{bmatrix}1 \\ 0\end{bmatrix} \begin{bmatrix}1 & 0\end{bmatrix} = \begin{bmatrix}
1 & 0 \\
0 & 0
\end{bmatrix}
\end{equation}

For $v = \begin{bmatrix}a \\ b\end{bmatrix}$:
\begin{equation}
Pv = \begin{bmatrix}1 & 0 \\ 0 & 0\end{bmatrix} \begin{bmatrix}a \\ b\end{bmatrix} = \begin{bmatrix}a \\ 0\end{bmatrix}
\end{equation}

We drop the $b$ component.

\section{Trace}

\subsection{Definition}

The trace of a square matrix $A \in \mathbb{R}^{n \times n}$ is the sum of its diagonal entries:

\begin{equation}
\text{tr}(A) = \sum_{i=1}^n A_{ii}
\end{equation}

\subsection{Key Properties}

\begin{align}
\text{tr}(A + B) &= \text{tr}(A) + \text{tr}(B) \\
\text{tr}(cA) &= c\,\text{tr}(A) \\
\text{tr}(A^\top) &= \text{tr}(A) \\
\text{tr}(AB) &= \text{tr}(BA) \quad \text{(cyclic property)} \\
\text{tr}(ABC) &= \text{tr}(BCA) = \text{tr}(CAB)
\end{align}

\subsection{Relation to Eigenvalues}

The trace equals the sum of eigenvalues:

\begin{equation}
\text{tr}(A) = \sum_{i=1}^n \lambda_i
\end{equation}

The determinant equals the product of eigenvalues:
\begin{equation}
\det(A) = \prod_{i=1}^n \lambda_i
\end{equation}

\subsection{MI Connection: Information Flow}

\begin{enumerate}
    \item \textbf{Attention}: In attention, $\text{tr}(QK^\top)$ (or more precisely, $\text{tr}(\text{softmax}(QK^\top))$) relates to how much information flows from keys to queries
    \item \textbf{Weight decay}: The trace of a weight matrix's covariance appears in regularization terms
    \item \textbf{Effective capacity}: The trace of the Fisher information matrix relates to how many independent parameters the model can use
    \item \textbf{Gradient flow}: $\text{tr}(W^\top W)$ is the squared Frobenius norm, often used as a regularizer
\end{enumerate}

\subsection{Example}

For $A = \begin{bmatrix}2 & 5 \\ 1 & 3\end{bmatrix}$:
\begin{equation}
\text{tr}(A) = 2 + 3 = 5
\end{equation}

Eigenvalues: $\lambda = 1, 4$. Sum = 5 $\checkmark$

\section{Norms of Matrices}

\subsection{Frobenius Norm}

The Frobenius norm treats the matrix as a long vector:

\begin{equation}
\|A\|_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n A_{ij}^2}
\end{equation}

Alternative expressions:
\begin{align}
\|A\|_F^2 &= \text{tr}(A^\top A) \\
\|A\|_F^2 &= \sum_{i=1}^r \sigma_i^2 \quad \text{(SVD)} \\
\|A\|_F^2 &= \sum_{i=1}^n \|A e_i\|_2^2 \quad \text{(sum of column norms)}
\end{align}

\subsection{Spectral Norm}

The spectral norm (operator norm) is the largest singular value:

\begin{equation}
\|A\|_2 = \sigma_{\max}(A) = \max_{\|x\|=1} \|Ax\|
\end{equation}

This measures the maximum stretching factor of the matrix.

\subsection{MI Connection}

\begin{enumerate}
    \item \textbf{Stability}: The spectral norm controls how much a layer can amplify inputs. Large spectral norms $\implies$ potential instability
    \item \textbf{Weight initialization}: Initialization schemes (e.g., Xavier, He) use the Frobenius norm to set scaling
    \item \textbf{Complexity measures}: The Frobenius norm is used in weight decay to penalize large weights
    \item \textbf{Superposition}: The ratio $\|W\|_F / \|W\|_2$ tells us how ``spread out'' the singular values are—this relates to how many features can be in superposition
\end{enumerate}

\subsection{Example}

For $A = \begin{bmatrix}3 & 0 \\ 0 & 4\end{bmatrix}$:
\begin{align}
\|A\|_F &= \sqrt{3^2 + 0^2 + 0^2 + 4^2} = \sqrt{25} = 5 \\
\|A\|_2 &= \max(3, 4) = 4
\end{align}

\section{Common Identities Reference}

For quick reference:

\begin{align}
(A^\top)^{-1} &= (A^{-1})^\top \\
(AB)^{-1} &= B^{-1}A^{-1} \quad \text{(if invertible)} \\
\text{tr}(AB) &= \text{tr}(BA) \\
\text{tr}(A^\top B) &= \sum_{i,j} A_{ij} B_{ij} \quad \text{(Frobenius inner product)} \\
\|A\|_F^2 &= \text{tr}(A^\top A) \\
\frac{\partial}{\partial A} \|A\|_F^2 &= 2A \\
\frac{\partial}{\partial A} \text{tr}(BA) &= B^\top \\
\frac{\partial}{\partial A} \text{tr}(A^\top B A) &= (B + B^\top)A
\end{align}

\section{Summary: MI-Relevant Linear Algebra Concepts}

\begin{table}[h]
\centering
\begin{tabular}{|l|l|}
\hline
\textbf{Concept} & \textbf{MI Application} \\ \hline
Dot product & Attention scores, feature similarity \\ 
Orthogonality & Independent features, disentanglement \\
Eigenvectors & Stable directions, feature preservation \\
SVD & Superposition, low-rank structure, pruning \\
Projection & Feature extraction, sparse autoencoders \\
Trace & Information flow, capacity measures \\
Frobenius norm & Weight decay, complexity, superposition measure \\
Spectral norm & Stability, Lipschitz constants \\ \hline
\end{tabular}
\end{table}

\end{document}

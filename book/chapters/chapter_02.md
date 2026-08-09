# Chapter 2: Neural Networks as Computational Systems

## 2.1 What Is a Neural Network, Really?

Before we can reverse engineer a neural network, we need to know what kind of object we are staring at. At first glance, it is a function approximator: a big black box that eats tensors and spits out tensors. But that is like calling a Swiss watch "a thing that tells time." A neural network is a **computational system**: a directed graph of primitive operations that executes an algorithm, layer by layer, edge by edge.

Mechanistic interpretability is the art of reading that algorithm from the weights.

This chapter gives you the vocabulary and the tools to do that reading. We will:

- Represent networks as **computational graphs** with precise node and edge semantics.
- Treat internal representations as **vectors in high dimensional space** and ask what information they carry.
- Distinguish **local** from **distributed** representations, and see precisely why the claim "neuron 847 fires for cats" is, in general, not a well posed statement about the function the network computes.
- Identify **features** using linear, nonlinear, and subspace methods.
- Understand why representation structure emerges from the interaction of architecture, optimization, data, and loss.

Let's begin.

---

## 2.2 The Computational Graph

### 2.2.1 Definition

A neural network is a **directed acyclic graph (DAG)** $G = (V, E)$ with a finite vertex set $V = \{z_1, \dots, z_n\}$, a finite edge set $E \subseteq V \times V$, and no directed cycle. Because $G$ is a finite DAG, it admits at least one topological ordering, so every node has a well defined computation order. Each node $z_i \in V$ computes a deterministic function of its parents:

$$
z_i = f_i\bigl(z_{\mathrm{parents}(i)}\bigr), \qquad \mathrm{parents}(i) = \{j : (j,i) \in E\}.
$$

The output $y$ is the composition of these functions along every directed path from input $x$ to output:

$$
y = f_L \circ f_{L-1} \circ \cdots \circ f_1(x).
$$

For a standard feedforward network with $L$ layers, this becomes:

$$
\begin{aligned}
h^{(0)} &= x, \\
h^{(\ell)} &= f^{(\ell)}\bigl(W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}\bigr) \quad \text{for } \ell = 1, \dots, L, \\
y &= h^{(L)}.
\end{aligned}
$$

Here $W^{(\ell)} \in \mathbb{R}^{d_{\ell} \times d_{\ell-1}}$ and $b^{(\ell)} \in \mathbb{R}^{d_{\ell}}$ are the trainable parameters, and $f^{(\ell)} : \mathbb{R} \to \mathbb{R}$ is applied elementwise to each coordinate of $W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}$.

> **Key point.** The *full* computational graph contains every scalar multiply add. For a transformer with $L$ layers, model dimension $d$, and sequence length $n$, this is $O(L \cdot n^2 \cdot d^2)$ operations. We cannot interpret that raw graph directly. We need a **simplified graph** $G' = (V', E')$ whose nodes are meaningful units (features, heads, circuits) and whose edges are information flows between those units.

### 2.2.2 The Simplification Problem

Mechanistic interpretability seeks a simplified graph $G'$ such that:

1. **Nodes** $V'$ represent meaningful computational units.
2. **Edges** $E'$ represent information flow between these units.
3. The simplified graph **preserves behavior** $B(x)$ for the inputs $x$ of interest.
4. The simplified graph is **significantly smaller** than the full graph, in the sense $|V'| \ll |V|$ and $|E'| \ll |E|$.

Formally, we want a surjective map $\pi: V \to V'$ such that for every edge $(u, v) \in E$ in the full graph, there is a corresponding directed path in $G'$ from $\pi(u)$ to $\pi(v)$ that preserves the functional dependence: intervening on the value carried at $\pi(u)$ changes the value computed at $\pi(v)$ in the same way that intervening on $u$ changes $v$ in the original graph, restricted to the input distribution under study.

<img src="fig2_1_graph_simplification.svg" alt="A full computational graph with thousands of nodes collapsing into a simplified circuit graph with a handful of interpretable nodes." width="100%">

*Figure 2.1: A full computational graph (left) collapses under $\pi$ into a simplified circuit graph (right). The simplified graph preserves the target behavior while remaining small enough to read.*

### 2.2.3 Example: Simplifying an MLP

Consider a three layer MLP with 512 hidden units per layer. The full graph has $O(512^2)$ edges per layer, roughly 786,000 edges in total across the layer. A mechanistic explanation might read:

> "Layer 1 detects edges, Layer 2 detects textures, Layer 3 detects objects."

This is a **compression ratio of roughly $10^5$**. But the explanation is only valid if every claim in it can be validated with **causal evidence**, meaning a controlled intervention on the graph rather than a mere correlation observed in the data: correlation bounds what a claim could be, intervention certifies that it is true.

---

## 2.3 Internal Representations

### 2.3.1 What Is a Representation?

An **internal representation** is a vector $h \in \mathbb{R}^d$ produced at some layer of the network. The same vector may simultaneously encode many variables (sentiment, syntax, topic, and even the position of a word in a sentence), each encoded in a possibly different geometric structure.

Think of $h$ as a high dimensional message passed from one part of the network to another. The central question is: what does that message *say*, and, more precisely, which parts of it does the rest of the network actually *use*?

### 2.3.2 Linear Decomposition

Suppose $h$ is well approximated by $k$ features $v_1, \dots, v_k \in \mathbb{R}^d$. We can write:

$$
h = \sum_{i=1}^{k} a_i v_i + \varepsilon,
$$

where:

- $v_i$ are **feature directions** (not necessarily orthogonal, and not necessarily linearly independent when $k > d$),
- $a_i \in \mathbb{R}$ are **feature coefficients** (activations),
- $\varepsilon$ is **residual noise**, the component of $h$ not modeled by $\mathrm{span}\{v_1, \dots, v_k\}$.

If the $v_i$ are orthonormal and $k \le d$, the $a_i$ are given exactly by the orthogonal projection $a_i = v_i^\top h$ and $\varepsilon$ is the component of $h$ orthogonal to $\mathrm{span}\{v_i\}$. If the $v_i$ are not orthogonal, or $k > d$ (an **overcomplete** dictionary), the coefficients must instead be obtained by solving a least squares or sparse coding problem, and the decomposition is no longer unique without an additional regularizer, typically an $\ell_0$ or $\ell_1$ sparsity penalty on $a$. The representation is called **distributed** when $k \gg 1$ and no single $v_i$ carries most of the variance of $h$, and **local** when a single coordinate or direction dominates.

### 2.3.3 The Information Theoretic View

Let $h$ be a representation and $Y$ a target variable, jointly distributed according to $p(h, y)$. The **mutual information** is:

$$
I(h; Y) = \mathbb{E}_{p(h, y)}\left[\log \frac{p(h, y)}{p(h)\,p(y)}\right] = D_{\mathrm{KL}}\bigl(p(h,y) \,\|\, p(h)p(y)\bigr).
$$

Two structural facts matter for interpretability work:

- $I(h; Y) \ge 0$, with equality if and only if $h$ and $Y$ are independent.
- **Data processing inequality.** If $Y \to h \to \hat{Y}$ forms a Markov chain, that is $\hat Y$ is computed from $h$ alone, then $I(h; Y) \ge I(\hat{Y}; Y)$. No downstream processing of $h$ can manufacture information about $Y$ that was not already present in $h$.

Neither fact, however, tells us about causal structure. In particular:

- $I(h; Y) > 0$ does **not** imply $h$ causes $Y$.
- $I(h; Y) > 0$ does **not** imply $Y$ causes $h$.
- $I(h; Y) > 0$ is consistent with both being effects of a common, unobserved cause $Z$, i.e. a confounder.

**Causal relevance requires intervention, not just information.** This is the central lesson of Chapter 1, and it bears repeating here in sharper form: mutual information is a ceiling on how much a representation could matter mechanistically, not a certificate that it does.

<img src="fig2_2_information_overlap.svg" alt="Two overlapping circles representing h and Y, with the overlap labeled as mutual information; the direction of causation is left undetermined." width="90%">

*Figure 2.2: Representation $h$ and target $Y$ share mutual information, the overlapping region. The overlap alone does not fix the direction of causation, nor whether $h$ is actually used to compute $Y$ downstream.*

---

## 2.4 The Zoo of Representations

A concept need not live in a single neuron. It may be encoded in any of the following structures.

| Structure | Mathematical Form | Interpretability Implication |
|:----------|:------------------|:-----------------------------|
| **Single neuron** | $f(h) = h_i$ | Easy to identify, but often **polysemantic**: one neuron can encode several unrelated concepts. |
| **Direction** | $f(h) = v^\top h$ | Requires finding $v$, but is more robust to basis choice than a single neuron. |
| **Subspace** | $f(h) = \|P_V h\|$ | Requires an orthonormal basis $V$; captures multi dimensional features. |
| **Sparse combination** | $f(h) = \sum_{i \in S} a_i (v_i^\top h)$ | Requires sparse autoencoders (SAEs); assumes features are sparse and possibly overcomplete. |
| **Nonlinear manifold** | $f(h) = g(\phi(h))$ | Requires nonlinear probes; captures curved feature boundaries. |
| **Circuit** | Distributed across layers | Requires graph level analysis; the feature is not localized to a single layer. |

> **The neuron is a coordinate; the feature is a functional pattern.** This distinction is fundamental. A neuron level claim ("neuron 847 fires for cats") is a claim about a *coordinate axis*, which is a modeling choice. A feature level claim ("the cat direction $v_{\text{cat}}$ is decodable from layer 5") is a claim about a *functional pattern* in the representation itself. As Exercise 2.1 makes precise, coordinates can be rotated by any orthogonal transformation without changing the function the network computes, while a genuine feature, defined as a direction with a demonstrated causal role, cannot be rotated away.

<img src="fig2_3_representation_zoo.svg" alt="Four panels showing a single neuron as a coordinate axis, a direction as a line through the origin, a subspace as a shaded plane, and a nonlinear manifold as a curved surface." width="100%">

*Figure 2.3: Four representational structures in a 2D slice of high dimensional space. A single neuron is a coordinate axis. A direction is a line through the origin. A subspace is spanned by a basis $V$. A nonlinear manifold is a curved surface parameterized by $\phi$.*

---

## 2.5 Why Representations Look the Way They Do

Representation structure does not fall from the sky. It emerges from the interaction of four forces:

1. **Architecture.** The inductive biases of the network (attention, convolutions, residual connections) constrain the space of representations that are even reachable by gradient descent.
2. **Optimization.** Gradient descent finds parameter configurations that minimize the loss on the training distribution. Nothing in the optimization objective rewards interpretability directly.
3. **Data.** The data distribution $p(x)$ determines which features are statistically useful to encode, and how correlated those features are with one another.
4. **Objective.** The loss function determines which of the useful features are actually *selected for*, among many that would fit the data equally well.

These four forces jointly push representations into a high dimensional geometry that is statistically efficient for the task but not necessarily aligned with human concepts. Our job is to reverse engineer that geometry.

---

## 2.6 Implementation: Extracting and Analyzing Representations

### 2.6.1 Extracting a Hidden Representation

```python
import torch
import torch.nn as nn
from typing import Dict

def extract_representation(
    model: nn.Module,
    layer_name: str,
    inputs: torch.Tensor
) -> torch.Tensor:
    """Extract hidden representation from a specific layer.

    Uses a forward hook to capture the activation tensor at the
    exact computational node specified by `layer_name`.

    Args:
        model: The neural network under investigation.
        layer_name: Dot-path to the layer, e.g. "encoder.layer2.attn".
        inputs: Input tensor of shape [batch, ...].

    Returns:
        Activation tensor of shape [batch, ..., d_model].
    """
    representations: Dict[str, torch.Tensor] = {}

    def hook_fn(module, input, output):
        # Transformer blocks often return tuples (output, attn_weights).
        # We capture only the primary activation tensor.
        if isinstance(output, tuple):
            representations["output"] = output[0].detach().clone()
        else:
            representations["output"] = output.detach().clone()

    # Navigate the model using dot notation: "blocks.5.attn"
    target = model
    for part in layer_name.split("."):
        target = getattr(target, part)

    handle = target.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        _ = model(inputs)

    handle.remove()  # Critical: always remove hooks to avoid leaks.
    return representations["output"]
```

### 2.6.2 Decomposing Along Feature Directions

```python
from typing import Tuple

def decompose_representation(
    h: torch.Tensor,
    feature_directions: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Decompose representation along given feature directions.

    Computes the projection coefficients and the reconstruction.
    Assumes feature_directions rows are orthonormal; if they are not,
    coefficients are the least squares solution only when d >= k and
    feature_directions has full row rank.

    Args:
        h: Representation tensor of shape [..., d_model].
        feature_directions: Unit vectors of shape [k, d_model].

    Returns:
        coefficients: Projection coefficients of shape [..., k].
        reconstruction: Reconstructed representation of shape [..., d_model].
    """
    # h: [..., d]
    # directions: [k, d]
    coefficients = h @ feature_directions.T  # [..., k]
    reconstruction = coefficients @ feature_directions  # [..., d]
    return coefficients, reconstruction
```

### 2.6.3 Measuring Representation Geometry

```python
def compute_representation_geometry(
    representations: torch.Tensor
) -> Dict[str, float]:
    """Compute geometric properties of a set of representations.

    Args:
        representations: Tensor of shape [N, d].

    Returns:
        Dictionary with norm statistics and pairwise cosine similarity stats.
    """
    norms = representations.norm(dim=1)  # [N]
    normalized = representations / (norms.unsqueeze(1) + 1e-8)
    cosine_matrix = normalized @ normalized.T  # [N, N]

    # Exclude the diagonal (self-similarity = 1)
    mask = ~torch.eye(len(representations), dtype=torch.bool)
    off_diagonal = cosine_matrix[mask]

    return {
        "mean_norm": norms.mean().item(),
        "std_norm": norms.std().item(),
        "mean_cosine": off_diagonal.mean().item(),
        "std_cosine": off_diagonal.std().item(),
        "max_cosine": off_diagonal.max().item(),
        "min_cosine": off_diagonal.min().item(),
    }
```

---

## 2.7 Measurement: What Does a Representation Contain?

Given a representation $h$ and a target variable $Y$, here are four ways to ask what $h$ knows about $Y$, in increasing order of evidential strength.

### 2.7.1 Linear Predictability

Train a linear probe $\hat{Y} = W h + b$ by minimizing squared error:

$$
\hat{W}, \hat{b} = \arg\min_{W, b} \; \mathbb{E}\bigl[\|Y - (W h + b)\|^2\bigr].
$$

Report $R^2$ (regression) or accuracy (classification) on held out data. If $R^2$ is high, $Y$ is **linearly decodable** from $h$. This is necessary but not sufficient for causal relevance: a linear probe can succeed purely because $h$ and $Y$ share a confound.

### 2.7.2 Nonlinear Predictability

Train an MLP probe $\hat Y = g_\theta(h)$ for some nonlinear $g_\theta$. If the nonlinear probe attains materially higher $R^2$ or accuracy than the best linear probe, $Y$ is encoded **nonlinearly** in $h$, meaning it lies on a curved decision boundary rather than a hyperplane.

### 2.7.3 Mutual Information Estimation

Estimate $I(h; Y)$ using one of:

- **Binning.** Discretize $h$ and $Y$ into bins and compute the discrete mutual information directly from empirical frequencies.
- **k nearest neighbor estimators.** The Kraskov Stögbauer Grassberger (KSG) estimator, which avoids binning by using local neighbor distances.
- **Neural estimators.** MINE (Mutual Information Neural Estimation) or InfoNCE, which lower bound $I(h;Y)$ via a learned critic function.

### 2.7.4 Causal Effect (The Gold Standard)

Intervene on $h$, using Pearl's $do$ operator to denote a forced assignment that overrides the value $h$ would otherwise take, and measure the resulting change in $Y$:

$$
\Delta_Y = \mathbb{E}\bigl[Y \mid do(h = h_{\text{intervened}})\bigr] - \mathbb{E}\bigl[Y \mid do(h = h_{\text{natural}})\bigr].
$$

Because $do(\cdot)$ severs the incoming edges to $h$ before fixing its value, $\Delta_Y \ne 0$ is evidence that $h$ lies on a causal path to $Y$ within the model, not merely that the two are correlated. This requires careful experimental design; see Chapter 1 and Appendix E.

<img src="fig2_4_measurement_pyramid.svg" alt="A pyramid with four levels: linear probe at the base, then nonlinear probe, then mutual information, then causal intervention at the apex." width="80%">

*Figure 2.4: The four levels of measurement, from linear probe (weakest) to causal intervention (strongest). Each step up costs more compute and licenses a stronger claim about mechanism.*

---

## 2.8 Intervention: Modifying Representations

Measurement tells us what a representation *contains*. Intervention tells us what a representation *does*.

### 2.8.1 Steering Along a Direction

Add a vector along a specific direction in representation space:

```python
def intervene_on_direction(
    model: nn.Module,
    inputs: torch.Tensor,
    layer_name: str,
    direction: torch.Tensor,
    scale: float
) -> torch.Tensor:
    """Steer the representation by adding a vector along a specific direction.

    This implements do(h = h + scale * v), where v is a unit vector.

    Args:
        direction: torch.Tensor [d_model], the steering direction.
        scale: float, magnitude of the intervention.

    Returns:
        Model output after the intervention.
    """
    direction = direction / (direction.norm() + 1e-8)

    def hook_fn(module, input, output):
        # output: [batch, seq, d_model] or [batch, d_model]
        intervention = scale * direction
        # Broadcast to match output shape
        while intervention.dim() < output.dim():
            intervention = intervention.unsqueeze(0)
        return output + intervention.to(output.device)

    target = model
    for part in layer_name.split("."):
        target = getattr(target, part)

    handle = target.register_forward_hook(hook_fn)
    output = model(inputs)
    handle.remove()
    return output
```

### 2.8.2 Ablating a Subspace

Remove the component of $h$ lying in a subspace spanned by orthonormal basis vectors $V = [v_1, \dots, v_k]$:

```python
def ablate_subspace(
    model: nn.Module,
    inputs: torch.Tensor,
    layer_name: str,
    basis_vectors: torch.Tensor
) -> torch.Tensor:
    """Ablate the component of the representation lying in a subspace.

    Computes the orthogonal projection onto span(basis_vectors) and subtracts it.

    Args:
        basis_vectors: torch.Tensor [k, d_model], orthonormal basis.

    Returns:
        Model output with the subspace ablated.
    """
    def hook_fn(module, input, output):
        # Project onto subspace: coeffs = output @ V^T, then projection = coeffs @ V
        coeffs = output @ basis_vectors.T  # [..., k]
        projection = coeffs @ basis_vectors  # [..., d]
        return output - projection

    target = model
    for part in layer_name.split("."):
        target = getattr(target, part)

    handle = target.register_forward_hook(hook_fn)
    output = model(inputs)
    handle.remove()
    return output
```

Mathematically, ablation computes:

$$
h_{\text{ablated}} = h - P_V h = (I - V V^\top) h,
$$

where $P_V = V V^\top$ is the orthogonal projector onto $\mathrm{span}(V)$, and $I - P_V$ is itself an orthogonal projector onto the orthogonal complement $\mathrm{span}(V)^{\perp}$, since $P_V^2 = P_V$ and $P_V^\top = P_V$.

<img src="fig2_5_steering_vs_ablation.svg" alt="Left panel shows steering as a point displaced along a direction vector. Right panel shows ablation as a point projected onto a subspace, flattening one coordinate to zero." width="100%">

*Figure 2.5: Steering versus ablation. Steering (left) displaces the representation along a chosen direction. Ablation (right) projects out an entire subspace, flattening the representation along that subspace to zero.*

---

## 2.9 Falsification

A representation hypothesis is **falsified** if any of the following hold:

1. **Poor generalization.** The feature direction does not predict the target variable on held out data.
2. **No causal effect.** Ablating the direction does not change the target behavior, i.e. $\Delta_Y \approx 0$.
3. **Better alternative.** Another direction explains more variance in the behavior under the same measurement protocol.
4. **Confounding.** The representation encodes the target variable only through a confounder, so the apparent dependence disappears once the confounder is controlled for or held fixed.

Each of these is a separate, testable claim. A mechanistic hypothesis is only considered established once it survives all four.

---

## 2.10 Reproduction Checklist

To make your representation analysis reproducible, record:

1. **Model architecture** and checkpoint hash.
2. **Layer names** and exact extraction points.
3. **Feature directions**, or the method used to compute them (PCA, SAE, etc.), including hyperparameters.
4. **Input distribution** and preprocessing pipeline.
5. **Random seeds** and software versions.
6. **Raw representations** and analysis code.

Without this record, your result is a story, not a scientific finding.

---

## 2.11 Alternative Explanations

Before you declare victory, consider these five traps.

| Trap | What It Means | How to Test |
|:-----|:------------|:------------|
| **Multiplexing** | $h$ encodes $Y$ and other variables simultaneously, along overlapping or non orthogonal directions | Check whether ablating the $Y$ direction also disrupts unrelated behaviors. |
| **Spurious decoding** | The linear probe exploits a correlation in the training distribution rather than a causal structure the network relies on | Test on counterfactual inputs where the correlation is broken by construction. |
| **Basis ambiguity** | The recovered feature direction is one of many directions related by a rotation of the true underlying basis | Check robustness of the finding under orthogonal transformations of the representation space (see Exercise 2.1). |
| **Nonlinear encoding** | $Y$ is a nonlinear function of $h$ that a linear probe cannot detect at all | Compare linear versus nonlinear probe performance directly. |
| **Downstream epiphenomenon** | $h$ correlates with $Y$ but does not lie on any causal path from $h$ to $Y$ inside the network | Perform the ablation test: does removing the feature direction actually change $Y$? |

---

## 2.12 Exercises

### Mathematical

**Exercise 2.1, Basis Ambiguity.**
Let $\theta = \{W^{(\ell)}, b^{(\ell)}\}_{\ell=1}^L$ be the parameters of the feedforward network in Section 2.2.1, with elementwise nonlinearity $f^{(\ell)}$ commuting suitably under permutation of coordinates. Show that for any orthogonal matrix $Q^{(\ell)}$ satisfying $f^{(\ell)}(Q^{(\ell)} u) = Q^{(\ell)} f^{(\ell)}(u)$ for all $u$ (as holds, for instance, for ReLU composed with a signed permutation matrix), replacing all weight matrices as

$$
W^{(\ell)} \mapsto Q^{(\ell)} W^{(\ell)} \bigl(Q^{(\ell-1)}\bigr)^\top, \qquad b^{(\ell)} \mapsto Q^{(\ell)} b^{(\ell)}
$$

preserves the network function $f_\theta(x)$ for every input $x$, while changing every neuron level interpretation of layer $\ell$. What does this imply for claims of the form "neuron $i$ in layer $\ell$ encodes concept $C$", and what additional property must a *direction* $v$ have to survive this transformation as a well posed claim?

**Exercise 2.2, Representation Manifold.**
Prove that the set $\mathcal{M} = \{h(x) : x \in \mathcal{X}\}$ forms a (possibly singular) manifold in $\mathbb{R}^d$ under mild smoothness assumptions on $h(\cdot)$. State a formal condition on the Jacobian $J_h(x) = \partial h / \partial x$, in terms of its rank as a function of $x$, under which $\mathcal{M}$ is a smooth submanifold locally, and a further condition under which it is well approximated by an affine subspace over a neighborhood of $x$.

### Implementation

**Exercise 2.3, Graph Tracer.**
Implement a computational graph tracer for a simple MLP that returns:

- The DAG as an adjacency list.
- A topological ordering of the nodes.
- The in-degree and out-degree of each node.

Test it on a three layer MLP and verify computationally that the resulting graph is acyclic.

**Exercise 2.4, Principal Angles.**
Write a function `principal_angles(U, V)` that computes the principal angles between two subspaces $U, V \subseteq \mathbb{R}^d$ using the SVD of $U^\top V$, where the columns of $U$ and $V$ are orthonormal bases of the respective subspaces. The principal angles $\theta_1 \le \cdots \le \theta_k$ satisfy

$$
\cos \theta_i = \sigma_i(U^\top V),
$$

where $\sigma_i$ is the $i$-th singular value of $U^\top V$ in decreasing order. Verify numerically that $\theta_i = 0$ for all $i$ when $U$ and $V$ span the same subspace.

### Experimental

**Exercise 2.5, Latent Variable Alignment.**
Train a three layer MLP on a synthetic task with five known latent variables $z_1, \dots, z_5$. Apply PCA to the hidden layer activations and measure how well each principal component $PC_i$ aligns with each latent variable $z_j$, using $R^2$ from a linear regression of $z_j$ on $PC_i$. Report the alignment matrix $A_{ij} = R^2(PC_i, z_j)$, and discuss what a near diagonal $A$ would imply versus a dense $A$.

### Research

**Exercise 2.6, Mutual Information versus Causation.**
Prove or disprove: if $I(h; Y) > 0$, there exists an intervention $do(h = h')$ that changes the distribution of $Y$. If the statement is false, provide an explicit counterexample. As a hint, consider a collider structure in which $h$ and $Y$ are both effects of a common cause $Z$ but neither causes the other; construct $p(h, Y, Z)$ explicitly and verify $I(h;Y) > 0$ while $do(h=h')$ leaves the marginal of $Y$ unchanged.

**Exercise 2.7, Layer Wise Interpretability.**
Investigate whether distributed representations in early layers of transformers are more or less interpretable, by the measurement pyramid of Section 2.7, than distributed representations in late layers. Design an experiment using both linear probes and causal ablations across matched layers. State your findings as falsifiable claims in the style of Section 2.9.

---

## References

- Hinton, G. E. (1986). "Learning Distributed Representations." *Technical Report, Carnegie-Mellon University*.
- Bengio, Y., Courville, A., & Vincent, P. (2013). "Representation Learning: A Review and New Perspectives." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 35(8), 1798-1828.
- Elhage, N., et al. (2022). "Superposition, Memorization, and Double Descent." *Anthropic*.
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.

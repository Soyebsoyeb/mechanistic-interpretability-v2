# Chapter 2: Neural Networks as Computational Systems

## 2.1 What Is a Neural Network, Really?

Before we try to reverse engineer a neural network, we need to know exactly what kind of object it is. At first glance, it looks like a function approximator: a box that takes in tensors and returns tensors. But that description is too shallow. It is like calling a Swiss watch "a thing that tells time." A neural network is a **computational system**: a directed graph of simple operations that runs an algorithm, step by step, edge by edge.

Mechanistic interpretability is the practice of reading that algorithm back out of the weights.

This chapter builds the tools for that reading. We will:

- Define networks as **computational graphs**, with a precise meaning for each node and edge.
- Treat internal representations as **vectors in a high dimensional space**, and ask exactly what information each vector carries.
- Separate **local** from **distributed** representations, and prove why the sentence "neuron 847 fires for cats" is, in general, not even a well defined claim about the function the network computes.
- Learn how to find **features** using linear, nonlinear, and subspace methods.
- See why the shape of a representation is not an accident. It comes from four forces acting together: architecture, optimization, data, and loss.

Let's begin.

---

## 2.2 The Computational Graph

### 2.2.1 Definition

**Definition 2.1 (Computational graph).** A neural network is a **directed acyclic graph (DAG)** $G = (V, E)$, where:

- $V = \{z_1, \dots, z_n\}$ is a finite set of nodes,
- $E \subseteq V \times V$ is a finite set of directed edges,
- there is no directed cycle: no sequence of edges leads from a node back to itself.

**Claim.** Every finite DAG has at least one *topological order*, that is, a way to list the nodes $z_{\sigma(1)}, \dots, z_{\sigma(n)}$ so that every edge points from an earlier node to a later one.

*Proof sketch.* Because $G$ is finite and acyclic, at least one node has no incoming edges (otherwise, following edges backward forever would have to repeat a node, giving a cycle). Remove that node and its outgoing edges; the remaining graph is still a finite DAG, so the same argument applies. Repeating this until no nodes remain produces the order. $\blacksquare$

This order matters because it tells us in which order the nodes can actually be computed: every node's parents already have values by the time we reach it.

Each node $z_i \in V$ computes a fixed function of its parents:

$$
z_i = f_i\bigl(z_{\mathrm{parents}(i)}\bigr), \qquad \mathrm{parents}(i) = \{j : (j,i) \in E\}.
$$

The network's output $y$ is the composition of these functions along every directed path from the input $x$ to the output:

$$
y = f_L \circ f_{L-1} \circ \cdots \circ f_1(x).
$$

For an ordinary feedforward network with $L$ layers, this reads:

$$
\begin{aligned}
h^{(0)} &= x, \\
h^{(\ell)} &= f^{(\ell)}\bigl(W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}\bigr) \quad \text{for } \ell = 1, \dots, L, \\
y &= h^{(L)}.
\end{aligned}
$$

Here $W^{(\ell)} \in \mathbb{R}^{d_{\ell} \times d_{\ell-1}}$ and $b^{(\ell)} \in \mathbb{R}^{d_{\ell}}$ are the trainable weights and biases, and $f^{(\ell)} : \mathbb{R} \to \mathbb{R}$ is applied to each entry of $W^{(\ell)} h^{(\ell-1)} + b^{(\ell)}$ separately.

> **Key point.** The *full* computational graph includes every single multiply and add. For a transformer with $L$ layers, model width $d$, and sequence length $n$, that is $O(L \cdot n^2 \cdot d^2)$ operations. No person can read that graph directly. We need a **simplified graph** $G' = (V', E')$ whose nodes stand for meaningful units, such as features, attention heads, or circuits, and whose edges show how information moves between them.

### 2.2.2 The Simplification Problem

We want to build a simplified graph $G'$ with four properties.

1. **Nodes** $V'$ stand for meaningful computational units.
2. **Edges** $E'$ stand for real information flow between those units.
3. The simplified graph **preserves the behavior** $B(x)$ we care about, for the inputs $x$ we care about.
4. The simplified graph is **much smaller** than the full graph: $|V'| \ll |V|$ and $|E'| \ll |E|$.

To state this precisely: we want a surjective map $\pi: V \to V'$, sending each node of the full graph to a node of the simplified graph, such that for every edge $(u, v) \in E$ there is a directed path in $G'$ from $\pi(u)$ to $\pi(v)$. That path must preserve the actual dependence: if we intervene on the value at $\pi(u)$, the value at $\pi(v)$ must change in the same way that intervening on $u$ changes $v$ in the original graph, at least for the inputs $x$ we are studying. This is a strong requirement. A grouping that only *looks* right on a few examples, without this intervention property, is not yet a valid simplification.

<img src="fig2_1_graph_simplification.svg" alt="A full computational graph with thousands of nodes collapsing into a simplified circuit graph with a handful of interpretable nodes." width="100%">

*Figure 2.1: A full computational graph (left) collapses under $\pi$ into a simplified circuit graph (right). The simplified graph keeps the target behavior while becoming small enough for a person to read.*

### 2.2.3 Example: Simplifying an MLP

Take a three layer MLP with 512 hidden units in each layer. The full graph has $O(512^2)$ edges per layer, about 786,000 edges in total across that layer. A mechanistic story might say:

> "Layer 1 detects edges, Layer 2 detects textures, Layer 3 detects objects."

That is a **compression ratio of roughly $10^5$**. This story is only worth trusting if every part of it can be checked with **causal evidence**: a controlled intervention on the graph, not just a pattern noticed in the data. Correlation tells us what a claim *might* be. Intervention tells us whether it is *actually* true.

---

## 2.3 Internal Representations

### 2.3.1 What Is a Representation?

An **internal representation** is a vector $h \in \mathbb{R}^d$ produced at some layer of the network. The same vector can carry several variables at once: sentiment, grammar, topic, even the position of a word in its sentence. Each of these variables might live in a different geometric shape inside that one vector.

Think of $h$ as a long message passed from one part of the network to another. The real question is not just "what does this message say," but "which parts of it does the rest of the network actually read and use."

### 2.3.2 Linear Decomposition

Suppose $h$ can be approximated using $k$ feature directions $v_1, \dots, v_k \in \mathbb{R}^d$:

$$
h = \sum_{i=1}^{k} a_i v_i + \varepsilon,
$$

where:

- $v_i$ are **feature directions** (they need not be perpendicular to each other, and need not be linearly independent when $k > d$),
- $a_i \in \mathbb{R}$ are **feature coefficients**, also called activations,
- $\varepsilon$ is **residual noise**: the part of $h$ that $\mathrm{span}\{v_1, \dots, v_k\}$ cannot explain.

**Case 1: orthonormal directions, $k \le d$.** Then the coefficients are given exactly by orthogonal projection, $a_i = v_i^\top h$, and $\varepsilon$ is exactly the part of $h$ perpendicular to $\mathrm{span}\{v_i\}$. This case is simple and unique.

**Case 2: non-orthogonal or overcomplete directions, $k > d$.** Now there is no single correct answer for $a$. We must solve a least squares or sparse coding problem, and we need an extra assumption, usually an $\ell_0$ or $\ell_1$ penalty that favors few nonzero coefficients, to pick out one solution among many.

We call the representation **distributed** when $k \gg 1$ and no single $v_i$ carries most of the variance of $h$. We call it **local** when one coordinate or direction does almost all the work.

### 2.3.3 The Information Theoretic View

Let $h$ be a representation and $Y$ a target variable, with a joint distribution $p(h, y)$. The **mutual information** between them is:

$$
I(h; Y) = \mathbb{E}_{p(h, y)}\left[\log \frac{p(h, y)}{p(h)\,p(y)}\right] = D_{\mathrm{KL}}\bigl(p(h,y) \,\|\, p(h)p(y)\bigr).
$$

Two facts about $I$ matter for our purposes.

**Fact 1 (non-negativity).** $I(h; Y) \ge 0$, with equality exactly when $h$ and $Y$ are independent. This follows from Jensen's inequality applied to the concave function $\log$, since $I(h;Y) = -\mathbb{E}\left[\log \frac{p(h)p(y)}{p(h,y)}\right] \ge -\log \mathbb{E}\left[\frac{p(h)p(y)}{p(h,y)}\right] = -\log 1 = 0$.

**Fact 2 (data processing inequality).** If $Y \to h \to \hat{Y}$ is a Markov chain, meaning $\hat Y$ is computed from $h$ alone and gets no other information about $Y$, then $I(h; Y) \ge I(\hat{Y}; Y)$. In plain terms: no amount of downstream processing of $h$ can create information about $Y$ that was not already inside $h$.

Neither fact tells us anything about causal direction. In particular:

- $I(h; Y) > 0$ does **not** mean $h$ causes $Y$.
- $I(h; Y) > 0$ does **not** mean $Y$ causes $h$.
- $I(h; Y) > 0$ is fully consistent with $h$ and $Y$ both being effects of some hidden common cause $Z$, called a confounder, with no direct link between $h$ and $Y$ at all.

**Causal relevance needs intervention, not just information.** This is the central lesson of Chapter 1, worth repeating here in sharper form: mutual information is a ceiling on how much a representation *could* matter mechanistically. It is not proof that the representation *does* matter.

<img src="fig2_2_information_overlap.svg" alt="Two overlapping circles representing h and Y, with the overlap labeled as mutual information; the direction of causation is left undetermined." width="90%">

*Figure 2.2: The representation $h$ and the target $Y$ share mutual information, shown as the overlapping region. The overlap alone does not tell us which way the causation runs, or whether $h$ is actually used by the network to compute $Y$.*

---

## 2.4 The Zoo of Representations

A concept does not have to live inside one neuron. It can be encoded in any of the structures below.

| Structure | Mathematical Form | Interpretability Implication |
|:----------|:------------------|:-----------------------------|
| **Single neuron** | $f(h) = h_i$ | Easy to find, but often **polysemantic**: one neuron may fire for several unrelated concepts. |
| **Direction** | $f(h) = v^\top h$ | Harder to find than a single neuron, but far more robust to how we chose our coordinates. |
| **Subspace** | $f(h) = \|P_V h\|$ | Needs an orthonormal basis $V$; captures features that need more than one dimension. |
| **Sparse combination** | $f(h) = \sum_{i \in S} a_i (v_i^\top h)$ | Needs a sparse autoencoder (SAE); assumes features are sparse and possibly overcomplete. |
| **Nonlinear manifold** | $f(h) = g(\phi(h))$ | Needs a nonlinear probe; captures curved feature boundaries. |
| **Circuit** | Distributed across layers | Needs graph level analysis; the feature is not tied to a single layer. |

> **A neuron is a coordinate. A feature is a functional pattern.** This is the key distinction of the whole chapter. A neuron level claim, "neuron 847 fires for cats," is a claim about one *coordinate axis*, and a coordinate axis is only a modeling choice, not a fact about the world. A feature level claim, "the cat direction $v_{\text{cat}}$ can be decoded from layer 5," is a claim about a *functional pattern* built into the representation itself. As Exercise 2.1 proves formally, we can rotate the coordinate axes by any orthogonal transformation without changing the function the network computes at all. A genuine feature, meaning a direction with a demonstrated causal role, survives that rotation. A single neuron, in general, does not.

<img src="fig2_3_representation_zoo.svg" alt="Four panels showing a single neuron as a coordinate axis, a direction as a line through the origin, a subspace as a shaded plane, and a nonlinear manifold as a curved surface." width="100%">

*Figure 2.3: Four representational structures in a 2D slice of a high dimensional space. A single neuron is a coordinate axis. A direction is a line through the origin. A subspace is spanned by a basis $V$. A nonlinear manifold is a curved surface parameterized by $\phi$.*

---

## 2.5 Why Representations Look the Way They Do

The shape of a representation is not random. It comes from four forces acting together.

1. **Architecture.** The built-in structure of the network, such as attention, convolutions, or residual connections, limits which representations gradient descent can ever reach in the first place.
2. **Optimization.** Gradient descent finds parameters that minimize the loss on the training data. Nothing in that process rewards a representation for being easy for humans to read.
3. **Data.** The distribution $p(x)$ of the training data decides which features are useful to encode, and how correlated those features tend to be with each other.
4. **Objective.** The loss function decides which of the useful features actually get **selected for**, out of many that would fit the data equally well.

Together, these four forces push representations toward a geometry that is efficient for the task, but not necessarily aligned with the concepts humans use. Our job in this book is to reverse engineer that geometry.

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

Given a representation $h$ and a target variable $Y$, here are four ways to ask what $h$ knows about $Y$, ordered from weakest to strongest evidence.

### 2.7.1 Linear Predictability

Train a linear probe $\hat{Y} = W h + b$ by minimizing squared error:

$$
\hat{W}, \hat{b} = \arg\min_{W, b} \; \mathbb{E}\bigl[\|Y - (W h + b)\|^2\bigr].
$$

Report $R^2$ for regression, or accuracy for classification, always on held out data the probe never trained on. A high $R^2$ means $Y$ is **linearly decodable** from $h$. This is necessary but not sufficient for causal relevance: a linear probe can succeed for the wrong reason, purely because $h$ and $Y$ happen to share a confounder.

### 2.7.2 Nonlinear Predictability

Train an MLP probe $\hat Y = g_\theta(h)$, where $g_\theta$ is nonlinear. If this probe does clearly better than the best possible linear probe, then $Y$ is encoded **nonlinearly** in $h$: it sits on a curved boundary that a straight hyperplane cannot capture.

### 2.7.3 Mutual Information Estimation

Estimate $I(h; Y)$ using one of these methods:

- **Binning.** Sort $h$ and $Y$ into bins, then compute mutual information directly from the resulting frequencies.
- **k nearest neighbor estimators.** The Kraskov Stögbauer Grassberger (KSG) estimator, which avoids binning by measuring distances to nearby points instead.
- **Neural estimators.** MINE (Mutual Information Neural Estimation) or InfoNCE, which produce a lower bound on $I(h;Y)$ using a trained critic network.

### 2.7.4 Causal Effect (The Gold Standard)

Force $h$ to take a chosen value, using Pearl's $do$ operator to mean "override $h$, ignoring whatever would normally produce it," and measure how much $Y$ changes as a result:

$$
\Delta_Y = \mathbb{E}\bigl[Y \mid do(h = h_{\text{intervened}})\bigr] - \mathbb{E}\bigl[Y \mid do(h = h_{\text{natural}})\bigr].
$$

Because $do(\cdot)$ cuts every incoming edge to $h$ before fixing its value, a nonzero $\Delta_Y$ is real evidence that $h$ sits on a causal path to $Y$ inside the model. It is not just a coincidence in the data. This kind of experiment needs careful design; see Chapter 1 and Appendix E.

<img src="fig2_4_measurement_pyramid.svg" alt="A pyramid with four levels: linear probe at the base, then nonlinear probe, then mutual information, then causal intervention at the apex." width="80%">

*Figure 2.4: The four levels of measurement, from linear probe (weakest) to causal intervention (strongest). Each level up costs more compute, and buys a stronger claim about mechanism.*

---

## 2.8 Intervention: Modifying Representations

Measurement tells us what a representation *contains*. Intervention tells us what a representation *does*.

### 2.8.1 Steering Along a Direction

Add a vector along one chosen direction in representation space:

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

Remove the part of $h$ that lies inside a subspace spanned by orthonormal basis vectors $V = [v_1, \dots, v_k]$:

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

In matrix form, ablation computes:

$$
h_{\text{ablated}} = h - P_V h = (I - V V^\top) h,
$$

where $P_V = V V^\top$ is the orthogonal projector onto $\mathrm{span}(V)$.

**Claim.** $I - P_V$ is itself an orthogonal projector, onto the orthogonal complement $\mathrm{span}(V)^{\perp}$.

*Proof.* Since the columns of $V$ are orthonormal, $V^\top V = I_k$, so $P_V^2 = V V^\top V V^\top = V (V^\top V) V^\top = V V^\top = P_V$: applying the projection twice does nothing new. Also $P_V^\top = (VV^\top)^\top = VV^\top = P_V$, so $P_V$ is symmetric. A symmetric matrix satisfying $P_V^2 = P_V$ is exactly the definition of an orthogonal projector, and one checks directly that $(I-P_V)^2 = I - 2P_V + P_V^2 = I - P_V$ and $(I-P_V)^\top = I - P_V$, so $I - P_V$ is an orthogonal projector too. Its image is $\{h - P_V h : h \in \mathbb{R}^d\}$, which is exactly the set of vectors orthogonal to every column of $V$, i.e. $\mathrm{span}(V)^{\perp}$. $\blacksquare$

<img src="fig2_5_steering_vs_ablation.svg" alt="Left panel shows steering as a point displaced along a direction vector. Right panel shows ablation as a point projected onto a subspace, flattening one coordinate to zero." width="100%">

*Figure 2.5: Steering versus ablation. Steering (left) moves the representation along a chosen direction. Ablation (right) removes an entire subspace, flattening the representation to zero along that subspace.*

---

## 2.9 Falsification

A representation hypothesis is **falsified** if any of the following are true.

1. **Poor generalization.** The feature direction fails to predict the target variable on held out data.
2. **No causal effect.** Ablating the direction does not change the target behavior, that is, $\Delta_Y \approx 0$.
3. **Better alternative.** A different direction explains more of the behavior under the same measurement protocol.
4. **Confounding.** The apparent dependence on the target variable disappears once we control for, or hold fixed, some confounder that explains both.

Each of these four points is a separate, testable claim. We only call a mechanistic hypothesis established once it has survived all four tests.

---

## 2.10 Reproduction Checklist

To make a representation analysis reproducible, record all of the following.

1. **Model architecture** and checkpoint hash.
2. **Layer names** and the exact points where activations were extracted.
3. **Feature directions**, or the exact method used to find them (PCA, SAE, etc.), including every hyperparameter.
4. **Input distribution** and preprocessing pipeline.
5. **Random seeds** and software versions.
6. **Raw representations** and the analysis code itself.

Without this record, a result is a story, not a scientific finding.

---

## 2.11 Alternative Explanations

Before declaring victory, check for these five traps.

| Trap | What It Means | How to Test |
|:-----|:------------|:------------|
| **Multiplexing** | $h$ encodes $Y$ and other variables at the same time, along directions that overlap or are not orthogonal | Check whether ablating the $Y$ direction also disrupts unrelated behaviors. |
| **Spurious decoding** | The linear probe is exploiting a coincidence in the training data, not a real causal structure the network relies on | Test on counterfactual inputs where that coincidence is broken on purpose. |
| **Basis ambiguity** | The direction we found is only one of many, related to each other by a rotation of the true underlying basis | Check whether the finding survives an orthogonal transformation of the representation space (see Exercise 2.1). |
| **Nonlinear encoding** | $Y$ depends on $h$ nonlinearly, in a way a linear probe cannot detect at all | Compare linear against nonlinear probe performance directly. |
| **Downstream epiphenomenon** | $h$ correlates with $Y$, but sits nowhere on the network's actual causal path from $h$ to $Y$ | Run the ablation test: does removing the feature direction actually change $Y$? |

---

## 2.12 Exercises

### Mathematical

**Exercise 2.1, Basis Ambiguity.**
Let $\theta = \{W^{(\ell)}, b^{(\ell)}\}_{\ell=1}^L$ be the parameters of the feedforward network from Section 2.2.1, and suppose the elementwise nonlinearity $f^{(\ell)}$ commutes appropriately with coordinate permutations. Prove that for any orthogonal matrix $Q^{(\ell)}$ satisfying $f^{(\ell)}(Q^{(\ell)} u) = Q^{(\ell)} f^{(\ell)}(u)$ for every $u$ (this holds, for example, when $f^{(\ell)}$ is ReLU and $Q^{(\ell)}$ is a signed permutation matrix), replacing every weight as

$$
W^{(\ell)} \mapsto Q^{(\ell)} W^{(\ell)} \bigl(Q^{(\ell-1)}\bigr)^\top, \qquad b^{(\ell)} \mapsto Q^{(\ell)} b^{(\ell)}
$$

leaves the network function $f_\theta(x)$ exactly the same for every input $x$, while changing every neuron level interpretation at layer $\ell$. What does this mean for a claim like "neuron $i$ in layer $\ell$ encodes concept $C$"? What extra property must a *direction* $v$ have in order to survive this transformation as a meaningful, well defined claim?

**Exercise 2.2, Representation Manifold.**
Prove that the set $\mathcal{M} = \{h(x) : x \in \mathcal{X}\}$ forms a (possibly singular) manifold in $\mathbb{R}^d$, under mild smoothness assumptions on $h(\cdot)$. State a precise condition on the rank of the Jacobian $J_h(x) = \partial h / \partial x$, as a function of $x$, under which $\mathcal{M}$ is a smooth submanifold near that point. State a further condition under which $\mathcal{M}$ is well approximated by a flat, affine subspace in a neighborhood of $x$.

### Implementation

**Exercise 2.3, Graph Tracer.**
Implement a computational graph tracer for a simple MLP that returns:

- the DAG as an adjacency list,
- a topological ordering of the nodes,
- the in-degree and out-degree of each node.

Test it on a three layer MLP, and verify by computation that the resulting graph really is acyclic.

**Exercise 2.4, Principal Angles.**
Write a function `principal_angles(U, V)` that computes the principal angles between two subspaces $U, V \subseteq \mathbb{R}^d$, using the SVD of $U^\top V$, where the columns of $U$ and $V$ are orthonormal bases of the two subspaces. The principal angles $\theta_1 \le \cdots \le \theta_k$ satisfy

$$
\cos \theta_i = \sigma_i(U^\top V),
$$

where $\sigma_i$ is the $i$-th singular value of $U^\top V$, listed in decreasing order. Confirm numerically that $\theta_i = 0$ for every $i$ when $U$ and $V$ span the same subspace.

### Experimental

**Exercise 2.5, Latent Variable Alignment.**
Train a three layer MLP on a synthetic task built from five known latent variables $z_1, \dots, z_5$. Run PCA on the hidden layer activations, and measure how well each principal component $PC_i$ lines up with each latent variable $z_j$, using the $R^2$ of a linear regression of $z_j$ on $PC_i$. Report the alignment matrix $A_{ij} = R^2(PC_i, z_j)$. Explain what a near-diagonal $A$ would tell us, and what a dense $A$ would tell us instead.

### Research

**Exercise 2.6, Mutual Information versus Causation.**
Prove or disprove: if $I(h; Y) > 0$, there must exist some intervention $do(h = h')$ that changes the distribution of $Y$. If the claim is false, build an explicit counterexample. Hint: consider a collider structure, where $h$ and $Y$ are both effects of a shared cause $Z$, but neither one causes the other. Write down $p(h, Y, Z)$ explicitly, confirm that $I(h;Y) > 0$, and show that $do(h=h')$ leaves the marginal distribution of $Y$ completely unchanged.

**Exercise 2.7, Layer Wise Interpretability.**
Investigate whether distributed representations in early transformer layers are more or less interpretable, by the measurement pyramid of Section 2.7, than those in late layers. Design an experiment that uses both linear probes and causal ablations across matched layers. Write your findings as falsifiable claims, in the style of Section 2.9.

---

## References

- Hinton, G. E. (1986). "Learning Distributed Representations." *Technical Report, Carnegie-Mellon University*.
- Bengio, Y., Courville, A., & Vincent, P. (2013). "Representation Learning: A Review and New Perspectives." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 35(8), 1798-1828.
- Elhage, N., et al. (2022). "Superposition, Memorization, and Double Descent." *Anthropic*.
- Pearl, J. (2009). *Causality: Models, Reasoning, and Inference* (2nd ed.). Cambridge University Press.

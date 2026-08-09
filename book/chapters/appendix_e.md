# Appendix E — Causal Inference

## E.1 Introduction

Appendix F §F.1 refers forward to "path patching from Appendix E" as an interpretability tool, and Appendix I §I.2.4–I.2.5 repeatedly distinguishes correlational from interventional evidence without formally defining what the difference *is*. This appendix supplies that formal content: what a structural causal model actually asserts (§E.2), what an intervention changes about it and why that's a different mathematical object from conditioning (§E.3–E.4), the graphical criterion (d-separation, §E.5) and adjustment procedure (back-door criterion, §E.6) that let a causal effect be computed from purely observational data under stated assumptions, and the potential-outcomes formalism (§E.7) that expresses the same content in a form closer to how causal effects are usually reported. §E.8 makes explicit the connection to path patching that motivated including this appendix at all.

---

## E.2 Structural Causal Models

**Definition.** A structural causal model (SCM) consists of a set of endogenous variables $X_1,\ldots,X_n$, exogenous (noise) variables $U_1,\ldots,U_n$ with a joint distribution $P(U)$, and structural equations
$$
X_i = f_i(\mathrm{Pa}_i, U_i),
$$
where $\mathrm{Pa}_i \subseteq \{X_1,\ldots,X_n\}\setminus\{X_i\}$ is the set of **parents** of $X_i$ — the other endogenous variables that directly determine $X_i$'s value — and $U_i$ is a variable-specific exogenous noise term, with $U_1,\ldots,U_n$ assumed **jointly independent** unless stated otherwise (dependence among the $U_i$ corresponds to unmodeled common causes, i.e. **latent confounding**, and is handled separately when present).

**Requirement (acyclicity).** The parent relation, read as a directed graph $G$ with an edge $X_j\to X_i$ whenever $X_j\in\mathrm{Pa}_i$, is required to be a **directed acyclic graph (DAG)** — no variable can be its own ancestor. This is not a simplifying convenience: without acyclicity, $X_i=f_i(\mathrm{Pa}_i,U_i)$ need not have a well-defined solution at all (a cyclic system of equations, like $X=Y+1,\,Y=X+1$, may have no solution or infinitely many), so acyclicity is exactly the condition under which the structural equations, together with a draw of $U$, determine a unique value for every $X_i$.

**Induced distribution.** Given $P(U)$ and the structural equations, sampling $U\sim P(U)$ and evaluating the $f_i$ in an order consistent with $G$ (guaranteed to exist by acyclicity — a topological sort) determines a unique value for every $X_i$, and hence induces a joint distribution $P(X_1,\ldots,X_n)$ over the endogenous variables — this is the **observational distribution**, the one estimated by passively collected data with no manipulation of the system.

**Claim (Markov factorization).** The induced observational distribution factorizes according to $G$:
$$
P(x_1,\ldots,x_n) = \prod_{i=1}^n P(x_i\mid \mathrm{pa}_i).
$$

**Proof sketch.** Because $X_i$ is a deterministic function of $\mathrm{Pa}_i$ and $U_i$ alone, and the $U_i$ are mutually independent, $X_i$ is conditionally independent of every non-descendant of $X_i$ given $\mathrm{Pa}_i$ (a non-descendant's value depends only on noise terms $U_j$ that are independent of $U_i$, propagated through the DAG structure); ordering the variables topologically and applying the chain rule of probability, $P(x_1,\ldots,x_n) = \prod_iP(x_i\mid x_1,\ldots,x_{i-1})$ collapses to $\prod_iP(x_i\mid\mathrm{pa}_i)$ once each conditional is reduced using this conditional-independence property. $\blacksquare$

This factorization is what makes the SCM a computationally tractable object (the joint distribution is specified by $n$ low-dimensional conditionals rather than one $n$-variable joint) and is the foundation of d-separation's graphical reading of conditional independence (§E.5).

---

## E.3 Intervention

**Definition.** The intervention $do(X=x)$ modifies the SCM by **replacing** the structural equation for $X$, $X=f_X(\mathrm{Pa}_X,U_X)$, with the constant assignment $X:=x$, leaving every *other* structural equation $f_j$, $j\ne X$, and the distribution $P(U)$, **unchanged**. Graphically, this deletes every edge *into* $X$ in $G$ (since $X$ no longer depends on its former parents or noise at all) while leaving all edges *out of* $X$ intact — the resulting graph is denoted $G_{\overline X}$, and the resulting **interventional** (post-intervention, mutilated-model) distribution is denoted $P(\,\cdot \mid do(X=x))$.

**Why this is a different object from conditioning.** Conditioning, $P(\,\cdot\mid X=x)$, restricts attention to the subpopulation of the *original, unmodified* system in which $X$ happened to equal $x$ — every other variable's value is whatever the *actual* structural equations (including $X$'s former parents' influence, propagated through any path) produced in that subpopulation. Intervening, $P(\,\cdot\mid do(X=x))$, instead asks what the system *would produce* if $X$ were forcibly set to $x$, with the rest of the mechanism running as usual but $X$'s own former causes no longer having any bearing on it. These coincide only when $X$ has no parents that also influence the variable of interest through another route (formally, §E.6's back-door criterion, with $Z=\emptyset$ sufficient) — in general $P(Y\mid do(X=x)) \ne P(Y\mid X=x)$, and the entire identification problem of causal inference (§E.4) is the problem of computing the left-hand side, which is often unobservable directly, from data that only ever samples the right-hand side (plus, sometimes, additional variables).

**Claim (truncated factorization / g-formula).** 
$$
P(x_1,\ldots,x_n\mid do(X_k=x_k)) = \prod_{i\ne k} P(x_i\mid\mathrm{pa}_i)\Big|_{X_k=x_k}, \qquad P(X_k=x_k\mid do(X_k=x_k))=1.
$$

**Proof.** Under $do(X_k=x_k)$, the SCM's structural equations are identical to the original for every $i\ne k$, and $X_k$'s equation is replaced by the constant $x_k$. Applying the Markov factorization (§E.2) to this *mutilated* SCM gives the same product as the original factorization, except the factor $P(x_k\mid\mathrm{pa}_k)$ — which represented $X_k$'s dependence on its former parents — is simply absent (replaced by the deterministic fact $X_k=x_k$), while every other factor $P(x_i\mid\mathrm{pa}_i)$, $i\ne k$, is unchanged because neither $f_i$ nor $P(U_i)$ changed for $i\ne k$; the parent values $\mathrm{pa}_i$ appearing in these unchanged factors are simply evaluated with $X_k$ held fixed at $x_k$ wherever $X_k\in\mathrm{Pa}_i$. $\blacksquare$

This single identity is the computational core of causal inference from a fully specified SCM: it reduces "what happens under intervention" to "the same observational factorization, with one factor deleted and one variable clamped" — the entire content of "correlation is not causation" is the fact that the deleted factor, $P(x_k\mid\mathrm{pa}_k)$, is in general *not* something that conditioning can also make vanish, because conditioning on $X_k=x_k$ leaves that factor's *implications for $\mathrm{pa}_k$* intact (conditioning can make you infer things about $\mathrm{pa}_k$ from observing $X_k=x_k$; intervening cannot, since $X_k$'s value no longer depends on $\mathrm{pa}_k$ at all).

---

## E.4 Causal Effect and the Identification Problem

Given an SCM, $P(Y\mid do(X=x))$ is *in principle* always computable via the g-formula (§E.3) if the full SCM (all structural equations and $P(U)$) is known. In practice, the SCM is not known — only samples from the observational distribution $P(X,Y,\ldots)$ are available (possibly along with the causal graph $G$, or an assumed graph, but not the functional forms $f_i$ or noise distributions). The **identification problem** is: given only $G$ (or weaker structural assumptions) and the observational distribution, can $P(Y\mid do(X=x))$ be expressed as a functional of the observational distribution alone — with no unmeasured quantity appearing in the expression? When it can, the effect is called **identifiable**; §E.6 gives one general sufficient condition (the back-door criterion) under which it is, with an explicit formula.

This is worth stating precisely because it is the exact sense in which "no causal intervention" (Appendix I §I.3) is a defect: a paper offering only $P(Y\mid X=x)$ (observational) has not thereby also reported $P(Y\mid do(X=x))$, and the two are equal only under a specific, checkable graphical condition (§E.6) that must be argued for, not assumed — reporting the former while writing prose that reads as a claim about the latter is exactly the confusion this apparatus exists to prevent.

---

## E.5 d-Separation

**Definition (path blocking).** A path (a sequence of edges, of any direction, connecting two nodes in $G$ — not necessarily a directed path) between $X$ and $Y$ is **blocked** by a set $Z$ if it contains at least one node $W$ such that either:

(a) $W$ is a **non-collider** on the path (i.e. the path does not have both edges pointing into $W$ — so $W$ is a "chain" node, $\cdots\to W\to\cdots$, or a "fork" node, $\cdots\leftarrow W\to\cdots$) and $W\in Z$; or

(b) $W$ is a **collider** on the path (both edges point into $W$: $\cdots\to W\leftarrow\cdots$) and neither $W$ nor any descendant of $W$ is in $Z$.

**Definition (d-separation).** $X$ and $Y$ are **d-separated** by $Z$ if *every* path between $X$ and $Y$ is blocked by $Z$.

The asymmetry between (a) and (b) is the crux of the definition and is worth making concrete: conditioning on a chain or fork node *blocks* the flow of association through it (this matches the intuitive "controlling for a confounder/mediator removes its influence"), but conditioning on a **collider** *opens* a path that was otherwise blocked — a fact with no analogue in simple correlational reasoning, and the source of **collider bias** (conditioning on a common effect of two otherwise-unrelated causes induces a spurious association between them, since ruling out some values of the effect constrains the causes to compensate for each other).

**Claim (global Markov property).** If the SCM's causal Markov condition holds (§E.2's factorization) and, additionally, **faithfulness** holds (no conditional independencies in $P$ beyond those implied by $G$'s structure — an assumption, not a consequence of the SCM definition, since it rules out exact cancellations between different causal paths that happen to produce an independence not read off the graph), then $X\perp\!\!\!\perp Y\mid Z$ (probabilistic conditional independence) **iff** $X$ and $Y$ are d-separated by $Z$ in $G$. *(Proof beyond this appendix's scope; this equivalence is what makes d-separation — a purely graphical, syntactic criterion — a correct proxy for conditional independence — a probabilistic, semantic property — given the graph.)*

---

## E.6 The Back-Door Criterion

**Definition.** A **back-door path** from $X$ to $Y$ is a path that begins with an edge *into* $X$ (i.e. $X\leftarrow\cdots$) — informally, a path along which association could flow due to a common cause of $X$ and $Y$, rather than along $X$'s own causal influence on $Y$.

**Definition.** $Z$ satisfies the back-door criterion relative to $(X,Y)$ if: (1) $Z$ blocks (in the d-separation sense, §E.5) every back-door path from $X$ to $Y$; and (2) no node in $Z$ is a descendant of $X$.

Condition (2) is essential and easy to violate inadvertently: including a descendant of $X$ in $Z$ risks conditioning on a variable that is itself partly a *consequence* of $X$ (e.g. a mediator on the very causal path from $X$ to $Y$ being estimated), which — by blocking a *front*-door (genuinely causal) path, or by opening a collider path if the descendant is a collider — can bias the adjustment away from the true causal effect rather than toward it.

**Claim (back-door adjustment formula).** If $Z$ satisfies the back-door criterion relative to $(X,Y)$, then
$$
P(Y=y\mid do(X=x)) = \sum_z P(Y=y\mid X=x, Z=z)\,P(Z=z)
$$
(sum replaced by an integral for continuous $Z$) — the causal effect is computable from purely observational quantities: a conditional distribution and a marginal, both estimable from data with no intervention required.

**Proof sketch.** By condition (1), $Z$ blocks every back-door path, so the only association remaining between $X$ and $Y$ conditional on $Z$ flows along directed paths *out of* $X$ — exactly the causal influence the do-operator is meant to isolate. Condition (2) guarantees conditioning on $Z$ does not itself remove or distort part of that causal influence (since $Z$ contains no mediator of it) and does not open a spurious collider path back through a descendant of $X$. Formally, this is proved via **do-calculus rule 2** (not derived here): under exactly these two conditions, $P(y\mid do(x),z) = P(y\mid x,z)$ — intervening on $X$ and conditioning on $Z$ gives the same result as merely conditioning on both, precisely because $Z$ has already screened off the confounding paths. Combining this with the law of total probability (Appendix B §B.7) applied to the post-intervention distribution, $P(y\mid do(x)) = \sum_zP(y\mid do(x),z)P(z\mid do(x))$, and using $P(z\mid do(x))=P(z)$ (which holds because condition (2) — $Z$ not a descendant of $X$ — implies intervening on $X$ cannot causally affect $Z$'s distribution, by the graph-surgery definition of §E.3, which only removes edges *into* $X$), the claim follows. $\blacksquare$

This is the formula practically meant when a paper says "we adjust for confounders $Z$": it is valid only under the two stated graphical conditions, both of which require knowing (or credibly assuming) enough about the causal graph $G$ to check them — "we controlled for $Z$" is an unsupported causal claim unless $Z$'s back-door adequacy is itself argued, not merely asserted, exactly the assumption-transparency point of Appendix I §I.2.4.

---

## E.7 Potential Outcomes

**Definition.** $Y_x$ denotes the value $Y$ *would* take if $X$ were set to $x$ — formally, in SCM terms, $Y_x$ is $Y$'s value in the mutilated model under $do(X=x)$ (§E.3), evaluated at the *same* draw of exogenous noise $U$ that actually occurred. This equivalence, $Y_x \equiv Y\big|_{do(X=x)}$ evaluated at the realized $U$, is what identifies the potential-outcomes framework as a restatement of the SCM framework at the level of individual units, rather than a distinct theory: a "potential outcome" is exactly a do-intervention's result, indexed by the specific (otherwise unobserved) noise realization of a specific unit.

**Fundamental problem of causal inference.** For a given unit, only one potential outcome is ever observed — $Y = Y_X$ (the outcome under whichever treatment $X$ the unit actually received) — while $Y_{x'}$ for any $x'\ne X$ is, for that unit, permanently counterfactual (Appendix E cannot observe both a person's outcome under treatment and under no treatment). This is a structural, not merely practical, limitation: no amount of additional data on the *same* unit under the *same* conditions resolves it, since it is a statement about a single realization of $U$.

**Definition.** The average treatment effect is $\mathrm{ATE} := E[Y_1-Y_0]$ (for binary $X\in\{0,1\}$), the average, over the population (i.e. over the distribution of $U$), of each unit's individual — and individually unobservable — treatment effect $Y_1-Y_0$. By linearity of expectation (Appendix B §B.4), $\mathrm{ATE}=E[Y_1]-E[Y_0]$, so estimating the ATE requires only the two *marginal* means $E[Y_1],E[Y_0]$, not the joint distribution of $(Y_0,Y_1)$ or any individual difference.

**Identification under unconfoundedness.** The **consistency** assumption, $Y = Y_X$ (the observed outcome equals the potential outcome under the treatment actually received — ruling out interference between units' treatments and outcomes, sometimes separated out as the **SUTVA**, stable unit treatment value assumption), together with **conditional ignorability**, $(Y_0,Y_1)\perp\!\!\!\perp X\mid Z$ (treatment assignment is as-good-as-random once $Z$ is accounted for — the potential-outcomes-framework restatement of $Z$ satisfying the back-door criterion, §E.6), gives
$$
E[Y_x] = E_Z\big[E[Y\mid X=x,Z]\big] = \sum_zP(Y\mid X=x,Z=z)P(z),
$$
identical in form to the back-door adjustment formula of §E.6 — the two frameworks, potential outcomes and SCMs/do-calculus, are not different theories with different formulas; conditional ignorability *is* the back-door criterion restated in potential-outcomes notation, and the resulting adjustment formula is the same object derived twice from equivalent starting assumptions.

---

## E.8 MI Connection: Path Patching as Intervention

Path patching (introduced informally in Appendix F §F.1 and referenced in Appendix I §I.2.4's mechanism-vs-correlation discussion) is a direct instance of the $do$-operator (§E.3) applied to the computational graph of a neural network: treat each attention head's or MLP's output as a node $X_i$ in a DAG whose edges are the residual-stream read/write dependencies of Appendix F §F.5.1, run the network once on a "clean" input to obtain one set of activations and once on a "corrupted" input to obtain another, and then compute $P(\text{output}\mid do(X_i = x_i^{\text{corrupted}}))$ — i.e., force the single node $X_i$ to the value it took on the corrupted run, while leaving every other node's structural equation (the rest of the forward pass) unchanged, exactly the graph-surgery definition of §E.3, not a conditioning operation. This is precisely why patching is described as a *causal* intervention rather than a correlational probe (the observational-vs-interventional distinction of Appendix I §I.3): reading out $X_i$'s value under different inputs and correlating it with the output (a "visualization" or "no held-out testing" style of evidence) only ever samples $P(\text{output}\mid X_i=x_i)$, whereas patching directly constructs the mutilated computation graph and evaluates $P(\text{output}\mid do(X_i=x_i))$ — the gap between these two quantities is exactly the gap the back-door criterion (§E.6) is designed to close in general causal settings, and is closed *automatically and exactly* in the patching setting because the network's forward pass is a fully known SCM (every $f_i$ and every edge is given by the architecture), not one whose structure must be inferred or assumed.

---

## E.9 Summary Table

| Concept | Definition | Key fact |
|---|---|---|
| SCM | $X_i=f_i(\mathrm{Pa}_i,U_i)$, $U_i$ jointly independent, $G$ acyclic | Induces $P(x_1,\ldots,x_n)=\prod_iP(x_i\mid\mathrm{pa}_i)$ (§E.2) |
| $do(X=x)$ | Replace $X$'s equation with $X:=x$; delete edges into $X$ | $P(\cdot\mid do(x)) \ne P(\cdot\mid x)$ in general; equal iff no confounding back-door (§E.3) |
| g-formula | — | $P(\cdot\mid do(X_k{=}x_k)) = \prod_{i\ne k}P(x_i\mid\mathrm{pa}_i)\big\vert_{X_k=x_k}$ (§E.3), proved from graph surgery |
| Identifiability | — | $P(Y\mid do(x))$ expressible from observational data alone; not guaranteed, needs a condition like back-door (§E.4) |
| d-separation | Path blocked by non-collider $\in Z$ or collider $\notin Z$ (and no descendant $\in Z$) | Colliders *open* paths when conditioned on — opposite of chains/forks (§E.5) |
| Back-door criterion | $Z$ blocks all back-door paths; $Z\cap\mathrm{Desc}(X)=\emptyset$ | $P(y\mid do(x)) = \sum_zP(y\mid x,z)P(z)$ (§E.6), via do-calculus rule 2 |
| Potential outcome $Y_x$ | $Y$ under $do(X=x)$, same noise draw | Only one $Y_x$ per unit ever observed (fundamental problem, §E.7) |
| ATE | $E[Y_1]-E[Y_0]$ | Identified by the same formula as back-door adjustment, under conditional ignorability $\equiv$ back-door (§E.7) |
| Path patching | do-intervention on one activation node | Exact instance of §E.3's g-formula; architecture gives the SCM directly, no identifiability assumption needed (§E.8) |

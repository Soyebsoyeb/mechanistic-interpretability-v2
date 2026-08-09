# Chapter 1: What Is Mechanistic Interpretability?

## Motivation

A neural network computes a function $f_\theta$ from an input space $\mathcal X$ to an output space $\mathcal Y$. Suppose we know $f_\theta(x)$ for every $x$ in some domain. This does not tell us how the network computes that output. This is not just a vague warning. It is a precise claim, and it can be proved. Since this claim motivates the whole field, we should prove it rather than simply assert it.

**Claim.** Take any finite domain $\mathcal X$. There exist two networks $f_{\theta_1}$ and $f_{\theta_2}$ that are truly different as computation graphs (different weights, and not related by a simple symmetry such as swapping the order of two hidden units) such that $f_{\theta_1}(x) = f_{\theta_2}(x)$ for every $x$ in $\mathcal X$.

**Proof by construction.** Let $\mathcal X = \{0,1\}^2$ and let the target function be XOR, so $y = x_1 \oplus x_2$. We build two networks that agree on every point of $\mathcal X$ but compute the answer in completely different ways.

The first network, $N_1$, has one hidden layer with two ReLU units. Define
$$
h_1 = \mathrm{ReLU}(x_1+x_2), \qquad h_2 = \mathrm{ReLU}(x_1+x_2-1), \qquad \text{output} = h_1 - 2h_2.
$$
Check all four inputs. At $(0,0)$: $h_1=0$, $h_2=0$, output $0$. At $(1,0)$: $h_1=1$, $h_2=0$, output $1$. At $(0,1)$: output $1$ by symmetry. At $(1,1)$: $h_1=2$, $h_2=1$, output $2-2=0$. So $N_1$ gets XOR right on all four points. Notice that the hidden feature in $N_1$ is a simple linear function of $x_1+x_2$.

The second network, $N_2$, has one hidden layer with four ReLU units, one for each of the four possible inputs. For a large constant $M$, unit $i$ fires only when the input matches pattern $i$:
$$
h_i = \mathrm{ReLU}\big(\mathbf 1[\text{input matches pattern } i]\cdot M\big), \qquad \text{output} = \sum_i \frac{c_i h_i}{M},
$$
where $c_i$ is the correct label for pattern $i$. This network also computes XOR correctly on all four points of $\mathcal X$. But its hidden units behave nothing like $N_1$'s. Each unit in $N_2$ is a switch for one single input pattern. None of them varies smoothly with $x_1+x_2$.

Both networks agree perfectly on $\mathcal X$. Yet they tell two different stories about what is happening inside. $N_1$'s story is: the network computes a simple function of $x_1+x_2$ and applies a threshold. $N_2$'s story is: the network memorizes each input pattern on its own, with no shared structure between them. No amount of input and output data from $\mathcal X$ can tell these two stories apart. $\blacksquare$

![Two different networks, N1 and N2, that agree on every point of the XOR domain but compute the answer through completely different internal structure.](../images/fig1_xor_networks.png)

*(Exercise 1 asks you to prove the general version of this fact, for any finite domain, not just XOR.)*

Here is the general reason this works. Matching a fixed function $g$ on a domain $\mathcal X$ places exactly $|\mathcal X|$ times the output dimension many constraints on the network's parameters. A typical network has far more free parameters than that. So the set of parameter values consistent with the observed behavior is usually a large, high dimensional set, not a single point. The internal mechanism of a trained network is a choice of where in that large set the training process happened to land. Input and output data alone cannot see that choice.

![A schematic of parameter space. The whole region shows every setting of theta that matches the observed behavior on X. The two marked points, theta 1 and theta 2, both sit inside that region, meaning both reproduce the same behavior, yet they correspond to networks with completely different internal mechanisms, matching N1 and N2 above.](images/fig4_solution_set.png)

So mechanistic interpretability asks a harder question than "what does the model do." It asks: what internal computation produces the observed behavior? This is an inference problem, and as the proof above shows, it is genuinely underdetermined by behavior alone. It needs extra evidence, especially evidence from intervention (see Appendix E), rather than being something we can simply read off from input and output pairs.

## Learning Objectives

By the end of this chapter, you will be able to:
- Tell apart black box analysis and mechanistic analysis, and say exactly what extra evidence the second one requires (see "Formal Definition")
- Name six levels of explanation for a neural network, and say exactly what each level does and does not prove (see "Levels of Explanation")
- Apply seven scientific tests to check a mechanistic hypothesis, stated as conditions you can actually check (see "Scientific Criteria")
- Design an intervention that tests a mechanistic claim, in the precise sense of the do operator from Appendix E
- Recognize the limits of correlation based interpretation, in the exact sense proved above

## Intuition

Think about a pocket calculator. If we only look at its behavior, we can ask "what is $2+2$?" and get the answer $4$. This tells us about $f(2,2)$ for one pair of inputs. By the proof above, this rules out mechanisms that disagree with this one answer. But among all the mechanisms that agree with it, and even with every other input pair we might check, it does not tell us which one the calculator actually uses inside. If we open the calculator and trace the electrical signals through its logic gates, we get a different kind of evidence entirely. This is closer to intervening on, or directly observing, the internal state, in the sense that Appendix E separates $P(\cdot \mid X = x)$ from $P(\cdot \mid do(X = x))$. It is not just "more of the same" evidence we already had.

Neural networks are far more complex than a calculator, but the goal is the same. We want to reverse engineer the internal algorithm from the trained weights, knowing from the proof above that behavior on its own cannot fully determine that algorithm.

## Formal Definition

**Mechanistic interpretability** is the scientific field that tries to reverse engineer what a trained neural network computes, by finding and testing causal relationships, in the technical sense of Appendix E, between internal parts of the network and the behaviors we observe. A causal relationship is one that survives an intervention. A merely correlational relationship is not enough.

### Levels of Explanation

| Level | Description | Example | What is actually proved |
|-------|-------------|---------|------|
| Behavioral | Input to output mapping | "The model predicts positive sentiment" | A claim about $f_\theta$ on the inputs we tested, and nothing about anything inside the network |
| Statistical | Correlation patterns | "The output correlates with sentiment words" | A claim that $P(\text{output}\mid\text{feature})$ differs from $P(\text{output})$, in the sense of mutual information from Appendix C. No claim about $P(\text{output}\mid do(\text{feature}))$ |
| Feature | An identified representation | "A direction in layer 5 encodes sentiment polarity" | A claim that some internal direction can be read out, or decoded, as correlating with a property. As Appendix E notes, being decodable is not the same as being used |
| Component | One specific unit | "Neuron 847 activates on negative phrases" | A statistical or behavioral claim about one unit on its own. Still not a causal claim about the network's output |
| Circuit | A subgraph of components | "Heads 3.1 and 4.2 form a sentiment circuit" | A causal claim, but only if it is backed by intervention evidence, such as ablation or patching (Appendix E), on exactly that subgraph. Co-activation alone is not enough |
| Algorithmic | The full procedure | "Negation detection, then polarity accumulation, then threshold comparison" | A causal claim about every single edge in a specific computational graph, each one tested on its own. This is the conjunction of many circuit level claims, chained together as in "The Causal Chain" below |

![The six levels of explanation drawn as a staircase, from Behavioral at the bottom to Algorithmic at the top, with each step needing a stronger kind of evidence than the one below it.](images/fig2_levels_of_explanation.png)

This table is really a strict order of evidence strength, not just a list. A claim only moves up a level if it also brings the kind of evidence the level above required. An "Algorithmic" claim is not simply a longer "Statistical" claim. It needs interventional evidence at every single step. It is common to state a claim using language from a high level, such as "the model computes sentiment through negation detection," while the actual evidence for it only reaches a low level, such as correlation for each individual step. This gap between how a claim is phrased and what its evidence actually shows is the single most common mistake in this field, and this book returns to it often. See Appendix I §I.2.1 on claim scope, and §I.3 on the same pattern shown as a red flag.

**Claim.** The sentence "the model predicts positive sentiment because its output correlates with sentiment" is not a mechanistic explanation. The word "because" claims causation. But the evidence given, that the output correlates with a feature (formally, $I(\text{output}; \text{sentiment word feature}) > 0$), is only Statistical level evidence. It is not Circuit level or Algorithmic level evidence. This is exactly the mismatch described above.

### The Causal Chain

$$
\text{Input} \to \text{Feature Detection} \to \text{Feature Representation} \to \text{Information Routing} \to \text{Decision Computation} \to \text{Output}
$$

![The causal chain drawn as six boxes connected by arrows, with a reminder above each arrow that it stands for a do-query to be tested, not an assumption.](images/fig3_causal_chain.png)

In the formal language of Appendix E, this chain is a claim about a structural causal model. Each box is a variable. Each arrow is an edge that is supposed to carry a real dependence. To explain the chain, we must show, for every edge $U \to V$, that $P(V \mid do(U=u))$ actually changes as $u$ changes. This uses the identification tools of Appendix E §E.3 through §E.6. It is not enough to see that $U$ and $V$ vary together across different inputs, since that is only the identification problem of Appendix E §E.4, restated here for a single network instead of a general causal question. So the warning "missing any link leaves the explanation incomplete" is a precise mathematical point, not just good advice. A chain of causal claims is only as strong as its weakest untested link. If one edge is never tested, it might in fact carry no causal weight at all, meaning $P(V \mid do(U=u))$ does not depend on $u$. In that case, everything downstream of that edge would be happening despite the earlier stages, not because of them, no matter how convincing the story sounds.

## Scientific Criteria for Mechanistic Hypotheses

Suppose we have a hypothesis $H$ that says some mechanism $M$ produces some behavior $B$. Here are seven conditions we can actually check.

1. **Localization.** $M$ must be named as a specific node, or a specific set of nodes, in the network's computation graph (see Appendix E §E.2 for this graph, applied to one particular model). That means a layer index, a component type, a head or neuron index, and the exact tensor coordinates. Saying "somewhere in the middle layers" does not name a node. Without a named node, the intervention $do(M=m)$ is not even well defined.

2. **Specification.** $M$ must be given as an explicit function, $M: \mathbb R^d \to \mathbb R^k$, written as a formula, not described in words. The causal claims in points 3 and 4 are claims about this exact function's role. A different function sitting in the same location would be a different hypothesis.

3. **Causal relevance.** This is exactly the identification problem from Appendix E, applied to one network. Does $P(B \mid do(M=m))$ differ from $P(B \mid do(M=m'))$ for some pair of values $m \ne m'$ that we can actually produce, using ablation, patching, or steering? Correlation alone is never enough, because $P(B \mid M=m)$ and $P(B \mid do(M=m))$ are different mathematical objects (Appendix E §E.3). They only agree under specific conditions, such as no confounding path, and those conditions must be shown, not assumed.

4. **Sufficiency.** If we intervene only on $M$, holding everything else at its normal computed value, this should reproduce the full change in $B$, with no unexplained extra components involved. Formally, using only $M$ and the known downstream structure, the g formula computation from Appendix E §E.3 should match the observed result of the intervention. If the explanation needs to add "and then some other stuff happens," that phrase is really admitting that some unnamed node is doing real causal work the hypothesis never accounted for.

5. **Generalization.** Showing causal relevance and sufficiency on one set of inputs does not guarantee the same result on another set of inputs. This is a separate empirical question, because a $do$ intervention's effect can genuinely change if $M$ interacts with other features that only appear in some inputs. So we must test held out data, different token identities, different sentence structures, and different topics, each as its own separate test of points 3 and 4. We cannot assume success on one case means success everywhere.

6. **Falsifiability.** Before running the experiment, state a specific result $R$ that would force us to reject $H$. For example, "the intervention in point 3 would show that $P(B \mid do(M=m))$ is approximately equal to $P(B \mid do(M=m'))$." If every possible outcome could somehow be explained as still consistent with $H$, perhaps by saying "the circuit is doing something subtler," then $H$ has not really been given a form that can be tested, no matter how technical it sounds.

7. **Independence.** Is the causal contribution of $M$, from point 3, separable from the rest of the network? In other words, does ablating $M$ produce an effect we can attribute to $M$ alone, or does the size of that effect depend on the state of other components too, so that "the effect of $M$" is not even well defined without also fixing those other components? This concern is the same one behind redundant or backup circuits, discussed again below, but stated here as a basic requirement for the causal claim to make sense at all.

## The Eight Questions Framework

| Question | Chapter 1 Example | Formal status |
|----------|-------------------|------|
| What is being claimed? | "Attention head 2 in layer 3 copies information from position $i$ to position $j$" | The hypothesis $H$, matching points 1 and 2 above |
| What mathematical object? | $W_{OV}^{(3,2)} = W_V^{(3,2)}W_O^{(3,2)} \in \mathbb{R}^{d\times d}$ | The explicit function required by point 2, here the OV circuit from Appendix F §F.7.1 |
| Where in the model? | Layer 3, head 2, in the residual stream right after attention | The node named by point 1 |
| How measured? | Attention weight $A_{j,i}$ and the projection of the output onto a target direction | Statistical level evidence from the levels table above. Necessary, but not enough on its own |
| How intervened? | Zero out the head's output, or patch the OV circuit from a clean run into a corrupted run | The $do(M=m)$ operation from point 3, carried out in practice through path patching, as in Appendix E §E.8 |
| What falsifies it? | Ablating the head does not change the copying behavior on held out data | The specific result $R$ required by point 6, stated in advance |
| Reproducible? | Fixed seed 42, checkpoint `gpt2`, prompt template `[A][B]...[A]` | Meets at least the first two levels of the reproduction criterion in Appendix I §I.2.6 |
| What are the alternatives? | The head may be redundant, another head may compensate, or the copying may be a side effect with no real role | The check for alternative explanations from Appendix I §I.2.5, worked out below |

## Implementation: Testing a Hypothesis

```python
import torch
import torch.nn as nn
from typing import Callable, Dict, Any

def evaluate_hypothesis(
    model: nn.Module,
    mechanism_location: str,
    intervention_fn: Callable,
    test_inputs: torch.Tensor,
    behavior_metric: Callable,
    n_repetitions: int = 5
) -> Dict[str, Any]:
    """Evaluate a mechanistic hypothesis by intervention.

    This function directly implements point 3, causal relevance.
    `clean_mean` estimates E[behavior_metric | do(M = the value the network
    actually computes)], which is simply the ordinary forward pass.
    `intervened_mean` estimates E[behavior_metric | do(M =
    intervention_fn(actual value))]. These are two genuinely different
    do-queries, in the sense of Appendix E §E.3. They are not a query and a
    plain conditioning of the same quantity. The forward hook below performs
    graph surgery, replacing the value at one node, exactly as Appendix E
    §E.3 defines it. It is not simply a filter applied to the final output.

    Args:
        model: The neural network under investigation
        mechanism_location: Dot-path to component, e.g. "blocks.5.attn.heads.2"
        intervention_fn: Callable that modifies the mechanism activation
        test_inputs: Input tensor [batch, ...]
        behavior_metric: Callable(output) -> float measuring target behavior
        n_repetitions: Number of independent runs for uncertainty estimation

    Returns:
        A dictionary with the clean score, the intervened score, their
        difference, an effect size (not just a yes or no about
        significance; see Appendix I §I.2.2 on why the size of an effect
        matters, not only whether it is detectable), and a boolean flag for
        whether the hypothesis is supported.

    Caveat: `n_repetitions` only changes the random seed. It keeps
    `test_inputs` fixed. So it measures variance coming from randomness
    inside the forward pass, such as dropout if it is active. It does NOT
    measure variance across different inputs, which is what point 5,
    generalization, requires. A hypothesis that passes this test has not
    yet been checked on held out inputs.
    """
    model.eval()
    clean_scores = []
    intervened_scores = []

    for seed in range(n_repetitions):
        torch.manual_seed(seed)

        with torch.no_grad():
            clean_output = model(test_inputs)
            clean_scores.append(behavior_metric(clean_output).item())

        # Intervened forward pass with hook
        intervened_score = _run_with_intervention(
            model, test_inputs, mechanism_location, 
            intervention_fn, behavior_metric
        )
        intervened_scores.append(intervened_score.item())

    clean_mean = sum(clean_scores) / len(clean_scores)
    inter_mean = sum(intervened_scores) / len(intervened_scores)
    clean_std = (sum((s - clean_mean)**2 for s in clean_scores) / len(clean_scores)) ** 0.5

    return {
        "clean_mean": clean_mean,
        "clean_std": clean_std,
        "intervened_mean": inter_mean,
        "difference": clean_mean - inter_mean,
        "effect_size": abs(clean_mean - inter_mean) / (clean_std + 1e-8),
        "hypothesis_supported": abs(inter_mean - clean_mean) > 2 * clean_std,
        "n_repetitions": n_repetitions
    }


def _run_with_intervention(model, inputs, location, intervention_fn, metric_fn):
    """Internal helper: run the model with an intervention hook applied.

    A forward hook that returns a new value in place of `out` (see
    Appendix G §G.4: any non-None value returned by a forward hook replaces
    the module's output for everything downstream) is exactly graph
    surgery in the sense of Appendix E §E.3. Every later computation now
    reads the intervened value. We remove the hook right after use (see
    Appendix G §G.4 on the risk of a leaked hook), so later, unrelated
    calls to `model` are not affected.
    """
    parts = location.split(".")
    target = model
    for part in parts:
        target = getattr(target, part)

    handle = target.register_forward_hook(
        lambda m, inp, out: intervention_fn(out)
    )

    with torch.no_grad():
        output = model(inputs)
        score = metric_fn(output)

    handle.remove()
    return score
```

## Alternative Explanations

Each item below describes a different causal structure, stated using the graph language of Appendix E, that would produce the exact same correlational evidence as our preferred hypothesis, but a different answer once we actually intervene, as in point 3. Checking these is not just extra caution. It follows directly from the identification problem in Appendix E §E.4.

**Downstream effect.** Here $M$ is actually a descendant of $B$, or of whatever truly causes $B$, rather than a cause of $B$. $M$ activates because the behavior already happened earlier in the computation. So $do(M=m)$ would not change $B$, even though $M$ and $B$ are strongly correlated. What looked like an arrow $M \to B$ was really $B \to M$.

**Redundancy.** Here a second component $M'$ is a separate, sufficient cause of $B$ on its own. If we ablate only $M$, the path through $M'$ still produces $B$, so point 3's single node test shows no effect, even though $M$ really does take part in computing $B$ when nothing is ablated. To tell this apart from "no causal role at all," we need a joint intervention on both $M$ and $M'$ together, not another single node test.

**Epiphenomenon.** Here $M$ has no edge into $B$'s computation at all in the true graph. Any correlation we see is either a small sample coincidence, or comes from a path that is not actually used on the inputs we tested.

**Correlated confounder.** Here some third node $C$ is a common ancestor of both $M$ and $B$. This is a back door path, as described in Appendix E §E.6. $M$ and $B$ correlate, but $do(M=m)$ cuts $M$ off from $C$ entirely, since intervention deletes incoming edges (Appendix E §E.3). If $C$'s influence on $B$ does not pass through $M$, then intervening on $M$ leaves $B$ untouched by $C$'s continuing influence along its other paths.

**Common cause.** This is the extreme version of the confounder case. Here $M$ and $B$ are both direct effects of one upstream cause $C$, with no edge from $M$ to $B$ at all. Intervening on $M$ cannot change $B$, no matter how tightly they are correlated in observation, because $do(M=m)$ does not propagate back to $C$ or to any of $C$'s other effects (Appendix E §E.3).

## Limitations

Mechanistic interpretability does not promise any of the following.

**Complete understanding of every model behavior.** The problem may simply be too hard to fully solve. The non identifiability result we proved above shows that the space of hypotheses consistent with observed behavior is usually large. Nothing guarantees this space can be narrowed down to one single mechanism using a realistic amount of intervention evidence.

**Human readable descriptions of every computation.** Some computations may not have any short description at all. A valid causal account, one that satisfies points 1 through 7 above, does not have to be a short one.

**Safety guarantees without further checking.** Even a mechanism that satisfies every criterion above is still only a description of what the network currently computes on the inputs we actually tested. Point 5, generalization, sets a real limit here. Nothing in this chapter allows us to assume safety properties carry over to inputs we have not tested.

**Transfer across model sizes.** A causal graph and its edge strengths, in the sense of Appendix E, are properties of one specific trained network. Nothing in the structural causal model framework implies that two networks trained differently, or at different sizes, share the same graph, even if they were trained on the same task.

## Exercises

### Mathematical
1. The Motivation section proves non identifiability for one specific case, XOR, where $|\mathcal X| = 4$. Now prove the general case. For an arbitrary finite domain $\mathcal X$ and an arbitrary function $g: \mathcal X \to \mathcal Y$, show that whenever a network architecture has more parameters than $|\mathcal X|$ times the output dimension of $\mathcal Y$ (this product is the number of real valued constraints imposed by matching $g$ on $\mathcal X$), the set of parameters $\theta$ with $f_\theta$ equal to $g$ on $\mathcal X$ generically contains more than one point, and these points are not all related by a simple weight space symmetry such as permuting hidden units. State clearly what the word "generically" is doing in this claim.
2. Turn the six levels of explanation from the table above into a strict hierarchy of evidence requirements. For each level, define the smallest set of do queries (Appendix E §E.3), or observational statistics (Appendix C §C.4), that a claim must have to count at that level. Then prove that each level's requirement set fully contains the requirement set of the level below it. In other words, show the table is a genuine refinement order, not just an informal list.

### Implementation
3. Write a function `extract_computational_graph(model)` that returns a directed acyclic graph, as an adjacency list, representing one forward pass, in the precise sense of Appendix E §E.2. Nodes are tensors or activations. Edges are direct functional dependencies inside $f_i(\mathrm{Pa}_i, U_i)$. Test the function on a simple MLP and on a transformer block, and check that the graph you extract really is acyclic in both cases, as Appendix E §E.2 requires.
4. Write a test suite that checks `evaluate_hypothesis` returns the same `clean_mean` and `clean_std` every time it is run with the same seeds (see Appendix G §G.7's determinism checklist, which applies directly here). Also check that the intervention hook is fully removed after each call, so no leftover hook affects a later, unrelated call (see Appendix G §G.4).

### Experimental
5. Train a small MLP with two hidden layers of 128 neurons each on MNIST. Find a neuron that correlates with the digit "7" using maximum activating examples. This is only a Statistical level claim, per the levels table. Now ablate that neuron, which is a point 3 intervention, and measure how accuracy changes on "7" compared to other digits. Report whether the neuron is causal, meaning point 3 is satisfied, or merely correlated, meaning we only have Statistical level evidence. Before concluding, check at least the redundancy and confounder alternatives from "Alternative Explanations."
6. Write a mechanistic hypothesis, addressing all seven criteria from "Scientific Criteria," for how a transformer predicts the next number in a simple arithmetic sequence, such as "2, 4, 6, 8, ...". Design three separate interventions, as in point 3, to test it. For each one, state the null hypothesis, the result you expect, and, following point 6, the specific result that would prove the hypothesis wrong, all stated before you run the experiment.
7. Investigate whether mechanistic explanations found in small models, under one million parameters, still hold for larger models trained on the same task. This is the generalization test from point 5, applied across models instead of across inputs. What breaks? What still holds?

## References

- Olah, C., Mordvintsev, A., & Schubert, L. (2017). "Feature Visualization." *Distill*.
- Olah, C., et al. (2020). "An Overview of Early Vision in InceptionV1." *Distill*.
- Elhage, N., et al. (2021). "A Mathematical Framework for Transformer Circuits." *Anthropic*.
- Nanda, N. (2022). "A Comprehensive Mechanistic Interpretability Explainer."
- Meng, K., et al. (2022). "Locating and Editing Factual Associations in GPT." *NeurIPS*.

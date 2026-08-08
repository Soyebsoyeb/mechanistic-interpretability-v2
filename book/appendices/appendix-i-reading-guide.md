# Appendix I — How to Read a Research Paper

## I.1 Introduction

The seven-item extraction below is easy to do superficially — restate the abstract under "Claim," restate the results section under "Evidence," and move on. Done that way it produces a summary, not an evaluation. The version here makes each category an operational question with a specific failure mode attached, and treats the "Red Flags" list not as a blacklist but as a set of conditions under which a *specific* inferential step is licensed or is not — because several of the "red flags" are only flaws in combination with a particular claim, not in isolation (§I.3 makes this precise).

---

## I.2 The Seven-Item Extraction

### I.2.1 Claim

Restating the abstract's claim in the paper's own words is not extraction — it is transcription. The operational version separates three things the abstract usually blurs together:

1. **The literal empirical claim**: what specific, checkable statement about a specific system did the experiments produce? (E.g., "ablating head 5.1 in this 12-layer, 117M-parameter model increases loss on the induction-pattern subset of this dataset by $X$.")
2. **The claimed scope**: does the paper assert this generalizes — across model scale, architecture family, dataset, task — or is generalization only implied by placement, framing, and title, without a corresponding experiment?
3. **The mechanism being claimed, if any** (kept separate here but overlapping with §I.2.3): "head 5.1 does induction" is a mechanistic claim; "ablating head 5.1 increases loss" is not, and papers frequently state the second while writing prose that reads like the first.

The failure mode this guards against: a title or abstract asserting a general mechanism ("Transformers Learn X"), a body of text presenting one behavioral measurement on one model, and a reader who absorbs the title's generality without noticing the gap.

### I.2.2 Evidence

"What experiment supports it" should be answered with enough specificity to say what the experiment *rules out*, not just what it shows. At minimum, record:

- **Design**: correlational (measure two things, note they covary) vs. interventional (manipulate one thing, measure the other) — see §I.3's discussion of why this distinction is often the whole ballgame.
- **N and units**: how many models / seeds / examples, and is the unit of replication the thing the claim is actually about? (A claim about "transformers" supported by one trained model, one seed, has $N=1$ at the level the claim is stated, even if the test set has a million examples.)
- **Effect size, not just significance**: a reported $p<0.05$ or a qualitative "clear effect in the plot" says nothing about magnitude. A statistically detectable effect and a *practically or mechanistically meaningful* one are different claims; conflating them is one of the most common overclaims in empirical ML papers.
- **Comparison baseline**: relative to what? An ablation without a matched random-ablation control, or an intervention without a matched-magnitude control intervention, cannot distinguish "this specific component matters" from "any sufficiently large perturbation matters."

### I.2.3 Mechanism

The proposed *internal process* is the part most likely to be underdetermined by the evidence (§I.2.2), because mechanism claims are usually about latent structure (representations, circuits, computations) that the experiment observes only indirectly. A rigorous extraction states the mechanism as a specific causal diagram or algorithm ("component A computes feature $f$; component B reads $f$ and writes output $g$"), not as a redescription of the result ("the model learned to do the task"), and separately notes: does the paper's evidence constrain the *specific* proposed mechanism, or only the *existence of some* mechanism producing the observed behavior? Behavioral evidence alone (§I.3) typically supports only the latter.

### I.2.4 Assumptions

Every interpretive step from evidence to claim relies on background assumptions that are usually not stated as assumptions — they're stated as facts, or not stated at all. Look specifically for:

- **Faithfulness assumptions**: that the tool measuring the internal process (a probe, an attention pattern, an activation visualization) actually reflects what the model computes, rather than what a downstream classifier can be trained to extract regardless of whether the model "uses" that information. A linear probe achieving high accuracy demonstrates the information is *linearly decodable*, not that the model *reads* it on the computational path that produces the output — these can come apart (Appendix E's discussion of path patching is precisely a tool for closing this gap).
- **Stationarity/generalization assumptions**: that a pattern found at one scale, checkpoint, or dataset slice extrapolates to the ones not tested.
- **Independence assumptions** silently required by any statistical test used (i.i.d. samples, no shared random seed across "independent" runs, no data leakage between the set a pattern was found on and the set used to "confirm" it).

### I.2.5 Alternative Explanations

For each candidate, ask specifically whether the paper's own evidence can distinguish it from the preferred explanation — an alternative that the reported experiment cannot rule out is a real gap, not a hypothetical nitpick. Standing candidates worth checking against *every* mechanistic or behavioral claim:

- **Confounded feature**: the "mechanism" tracks a correlated but distinct feature of the input (e.g., a claimed semantic feature that is actually a proxy for sequence length, token frequency, or surface form).
- **Shortcut/spurious correlation**: the effect holds on the tested distribution because of a dataset artifact, not because the proposed general mechanism exists.
- **Selection effect**: the reported cases were found by searching for cases that show the effect (§I.3), so their existence is not evidence the mechanism is common or central.
- **Simpler sufficient mechanism**: a lower-level or more mundane explanation (numerical artifact, initialization statistics, an unintended asymmetry in the ablation itself) produces the same measurement without the proposed higher-level mechanism being real.

### I.2.6 Reproduction

"Can the experiment be reproduced" decomposes into at least three independently-failable questions, and a paper can satisfy some without satisfying others:

1. **Code+data available**: can the exact pipeline be re-run at all?
2. **Result replicates**: re-running it (same or re-implemented code) on the same data produces the same qualitative — ideally quantitative — result.
3. **Result generalizes under re-implementation**: an independent re-implementation from the paper's *description alone* (not its code) produces the same result — this is the strongest form, and the one most papers are never subjected to, since it is the only one of the three that also tests whether the paper's own description of its method is complete and correct.

A result that reproduces under (1)–(2) but has never been attempted under (3) should be flagged as such, not treated as fully validated.

### I.2.7 Extension

The strongest extension experiment is usually the one that would most cleanly separate the paper's preferred mechanism (§I.2.3) from the strongest alternative explanation identified in §I.2.5 — not simply "more of the same" (larger model, more examples), which typically increases confidence in the *existence* of the effect without addressing *which explanation* produces it. A useful test: state the alternative explanation as a specific prediction that differs from the paper's mechanism's prediction, and design the experiment as the one that adjudicates between the two predictions, not one that is merely consistent with the preferred one.

---

## I.3 Red Flags, Made Precise

The five items below are not independent, absolute defects — each is a specific way that a specific class of evidence fails to support a specific class of claim. Restated with the condition under which each is actually disqualifying:

- **Claims based solely on visualization.** A visualization (an attention-pattern heatmap, a PCA plot of activations, a UMAP embedding) is an observational, non-interventional form of evidence: it shows a correlation between an internal quantity and something interpretable to a human, but by construction involves no manipulation of the system, so it cannot on its own distinguish "this internal quantity is functionally responsible for the behavior" from "this internal quantity happens to be a byproduct that correlates with it" (§I.2.4's faithfulness point, §I.2.5's confounded-feature and simpler-sufficient-mechanism alternatives). This is a red flag specifically for **causal/mechanistic** claims; it is not a defect if the paper's claim is explicitly correlational/descriptive ("here is a pattern we observed") and is presented as such. The flag is the mismatch between an observational method and a causal claim, not the method in isolation.

- **No causal intervention.** A stronger, more general version of the above: if the paper's central claim is that component/feature X *causes* behavior Y, and no experiment manipulates X (ablation, activation patching, steering, causal scrubbing) while holding other factors fixed and measuring Y, the causal claim is unsupported regardless of how much correlational evidence is presented — correlational evidence, however abundant, does not accumulate into causal evidence (this is not a matter of needing "more" observational data; it is a difference in evidence *type*, addressed in §I.2.2's design distinction).

- **No held-out testing.** A pattern or rule identified by inspecting a dataset and then "confirmed" by checking it on the same dataset (or hand-picked examples drawn from the same search that found the pattern) has not been tested against the possibility that it was fit to noise or to that specific sample — this is the same failure as evaluating a model on its training set. The disqualifying condition specifically is: the confirmation set and the discovery set overlap, or the confirmation set was selected using knowledge of what would confirm the pattern.

- **No discussion of limitations.** The absence of a limitations section is weak evidence on its own (space constraints, venue norms, and author habits vary); the stronger and more specific version of this flag is the absence of any acknowledgment of the specific gaps identifiable from §I.2.4–I.2.5 applied to *this particular paper* — i.e., whether the paper's own claims outrun what its own assumptions and alternative explanations would allow, independent of whether a section is literally labeled "Limitations."

- **Cherry-picked examples only.** Qualitative examples ("here are three cases where the model does X") are a legitimate way to *illustrate* a claim already supported by systematic evidence, but are not themselves evidence of prevalence, frequency, or generality — three supporting examples say nothing about whether the effect holds on 3% or 97% of cases, and a paper that presents only examples (no systematic count, rate, or held-out evaluation, §I.2.2's "N and units") has provided existence evidence, not the frequency or generality evidence its framing may be claiming. This is disqualifying specifically when the surrounding prose asserts prevalence or generality; it is not disqualifying for a claim that is explicitly scoped as "this is possible" rather than "this is typical."

> **General pattern across all five**: each red flag identifies a form of evidence that supports a narrower claim than the one being made — existence rather than prevalence, correlation rather than causation, in-sample fit rather than generalization. None of the five is a reason to dismiss a paper's *narrower*, correctly-scoped claim; all five are reasons to distrust the *broader* claim when the paper's framing quietly upgrades to it.

---

## I.4 Applying the Framework: A Worked Skeleton

For a hypothetical paper claiming "head $h$ in layer $\ell$ implements induction, which explains the model's in-context learning ability":

| Item | What to actually check |
|---|---|
| Claim | Is "implements induction" a claim about this one trained model, or about induction heads generally? Is "explains in-context learning" backed by an experiment connecting head $h$ to ICL performance, or is it asserted by proximity? |
| Evidence | Is the induction-head identification behavioral (attention pattern matches the induction template) or interventional (ablating $h$ measurably degrades induction-pattern completions, ideally relative to a random-head-ablation baseline)? |
| Mechanism | Does the paper specify the QK/OV circuit (Appendix F §F.7.1) implementing "attend to the token after the last occurrence of the current token," or only show an attention-pattern picture consistent with that story? |
| Assumptions | Does the attention-pattern-based identification assume the visualized pattern is what the head's OV circuit actually acts on (§I.2.4's faithfulness assumption)? |
| Alternatives | Could $h$'s ablation effect be explained by removing attention capacity generally (simpler sufficient mechanism), rather than removing an induction-specific computation? Was a magnitude-matched random ablation run? |
| Reproduction | Is $h$'s induction behavior reported on one model/seed, or checked across re-trained seeds and, ideally, an independent re-implementation? |
| Extension | An activation-patching experiment that swaps only $h$'s output between two prompts differing solely in the token to be induction-completed, predicting the completion should follow the patched value — this discriminates "$h$ causally carries the induction signal" from "$h$'s attention pattern merely correlates with it." |

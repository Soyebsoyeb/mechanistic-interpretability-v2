# Appendix E — Causal Inference

## Structural Causal Models
$X_i = f_i(Pa_i, U_i)$ where $Pa_i$ are parents and $U_i$ are exogenous noise.

## Intervention
$do(X=x)$ replaces the structural equation for $X$ with $X=x$.

## Causal Effect
$P(Y|do(X=x))$ vs. observational $P(Y|X=x)$.

## Back-Door Criterion
A set $Z$ satisfies the back-door criterion if:
1. $Z$ blocks all back-door paths from $X$ to $Y$
2. No node in $Z$ is a descendant of $X$

## d-Separation
$X$ and $Y$ are d-separated by $Z$ if all paths between them are blocked by $Z$.

## Potential Outcomes
$Y_x$: outcome under treatment $X=x$.
$\text{ATE} = E[Y_1 - Y_0]$.

# Appendix B — Probability

## Conditional Probability
$P(A|B) = P(A \cap B)/P(B)$ for $P(B) > 0$.

## Bayes' Theorem
$P(A|B) = P(B|A)P(A)/P(B)$.

## Expectation
$E[X] = \sum_x x p(x)$ for discrete; $E[X] = \int x p(x) dx$ for continuous.

## Variance
$\text{Var}(X) = E[(X-E[X])^2] = E[X^2] - E[X]^2$.

## Covariance
$\text{Cov}(X,Y) = E[(X-E[X])(Y-E[Y])]$.

## Law of Total Probability
$P(A) = \sum_i P(A|B_i)P(B_i)$ for partition $\{B_i\}$.

## Central Limit Theorem
$\frac{1}{\sqrt{n}}\sum_{i=1}^n (X_i - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$.

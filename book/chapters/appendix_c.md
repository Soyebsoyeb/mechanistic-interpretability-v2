# Appendix C — Information Theory

## Entropy
$H(X) = -\sum_x p(x) \log p(x)$.

## Joint Entropy
$H(X,Y) = -\sum_{x,y} p(x,y) \log p(x,y)$.

## Conditional Entropy
$H(X|Y) = -\sum_{x,y} p(x,y) \log p(x|y)$.

## Mutual Information
$I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X) = H(X) + H(Y) - H(X,Y)$.

## KL Divergence
$D_{KL}(p||q) = \sum_x p(x) \log \frac{p(x)}{q(x)}$.

## Cross-Entropy
$H(p,q) = -\sum_x p(x) \log q(x) = H(p) + D_{KL}(p||q)$.

## Data Processing Inequality
$I(X;Y) \geq I(X;g(Y))$ for any function $g$.

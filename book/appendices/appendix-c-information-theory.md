# Appendix C — Information Theory

- Entropy: $H(X) = -\sum p(x) \log p(x)$
- Conditional entropy: $H(X|Y) = -\sum p(x,y) \log p(x|y)$
- Mutual information: $I(X;Y) = H(X) - H(X|Y)$
- KL divergence: $D_{KL}(p\|q) = \sum p(x) \log(p(x)/q(x))$
- Cross-entropy: $H(p,q) = -\sum p(x) \log q(x)$
- Chain rule: $I(X;Y,Z) = I(X;Y) + I(X;Z|Y)$

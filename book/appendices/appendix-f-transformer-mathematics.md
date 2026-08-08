# Appendix F — Transformer Mathematics

For one head:
$Q = XW_Q, K = XW_K, V = XW_V$
$A = \text{softmax}(QK^\top/\sqrt{d_k})$
$Z = AV, O = ZW_O$

MLP: $M(X) = W_2\sigma(W_1X+b_1)+b_2$
Residual: $X_{\ell+1} = X_\ell + A_\ell + M_\ell$
LayerNorm: $\text{LN}(x) = \gamma \odot \frac{x - \mu}{\sigma} + \beta$

Multi-head: $\text{Concat}(Z^{(1)},\ldots,Z^{(h)})W_O^{\text{multi}}$

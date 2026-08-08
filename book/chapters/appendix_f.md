# Appendix F — Transformer Mathematics

## Single Head
$Q = XW_Q$, $K = XW_K$, $V = XW_V$
$A = \text{softmax}(QK^\top/\sqrt{d_k})$
$Z = AV$, $O = ZW_O$

## Multi-Head Attention
$\text{MultiHead}(X) = \text{Concat}(Z^{(1)},...,Z^{(h)})W_O^{\text{multi}}$

## MLP
$M(X) = W_2\sigma(W_1X+b_1)+b_2$

## Residual Update
$X_{\ell+1} = X_\ell + A_\ell + M_\ell$

## Layer Normalization
$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$

## Position Embeddings
$PE_{(pos, 2i)} = \sin(pos/10000^{2i/d})$
$PE_{(pos, 2i+1)} = \cos(pos/10000^{2i/d})$

## Rotary Position Embedding (RoPE)
$R_{\Theta}^d x = \begin{pmatrix} x_1 \ x_2 \ x_3 \ x_4 \\ \vdots \\ \end{pmatrix} \otimes \begin{pmatrix} \cos m\theta_1 \\ \cos m\theta_1 \\ \cos m\theta_2 \\ \cos m\theta_2 \\ \vdots \\ \end{pmatrix} + \begin{pmatrix} -x_2 \ x_1 \ -x_4 \ x_3 \\ \vdots \\ \end{pmatrix} \otimes \begin{pmatrix} \sin m\theta_1 \\ \sin m\theta_1 \\ \sin m\theta_2 \\ \sin m\theta_2 \\ \vdots \\ \end{pmatrix}$

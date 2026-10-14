# Chain rule in deep learning

Backpropagation is the chain rule applied backward. This note builds from a
scalar case up to a small two-layer network, using the vector-Jacobian product
(VJP) form that PyTorch's autograd (and AOTAutograd's backward graphs) actually
compute.

It is the math behind the `tangents_1` and `expand` pieces discussed in
`tasks/02/joint_graph_explained.md`.

## 1. The basic rule

If $y = f(u)$ and $u = g(x)$, then:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

The derivative of the outside function, evaluated at the inside value, times
the derivative of the inside function.

## 2. Simple scalar example

$$y = (3x + 2)^2$$

Inner: $u = 3x + 2$, so $\dfrac{du}{dx} = 3$.

Outer: $y = u^2$, so $\dfrac{dy}{du} = 2u = 2(3x + 2)$.

$$\frac{dy}{dx} = 2(3x + 2) \cdot 3 = 6(3x + 2)$$

At $x = 1$:

$$u = 5, \qquad \frac{dy}{dx} = 30$$

Numeric check: $y(1) = 25$ and $y(1.001) = (3.003 + 2)^2 = 25.030\ldots$, so
the slope is about $\dfrac{0.030}{0.001} = 30$.

## 3. One neuron with a nonlinearity

$$z = wx + b, \qquad a = \operatorname{relu}(z), \qquad L = a$$

The chain:

$$\frac{dL}{dw} = \frac{dL}{da}\cdot\frac{da}{dz}\cdot\frac{dz}{dw}$$

$$\frac{dL}{db} = \frac{dL}{da}\cdot\frac{da}{dz}\cdot\frac{dz}{db}$$

$$\frac{dL}{dx} = \frac{dL}{da}\cdot\frac{da}{dz}\cdot\frac{dz}{dx}$$

Concrete numbers: $x = 1$, $w = 2$, $b = -1$.

$$z = 2\cdot 1 - 1 = 1, \qquad a = \operatorname{relu}(1) = 1, \qquad
\operatorname{relu}'(1) = 1$$

$$\frac{dL}{dw} = 1 \cdot 1 \cdot x = 1$$

$$\frac{dL}{db} = 1 \cdot 1 \cdot 1 = 1$$

$$\frac{dL}{dx} = 1 \cdot 1 \cdot w = 2$$

## 4. Vector intermediate and the VJP

Now $z$ and $a$ are vectors:

$$z = Wx + b, \qquad a = \operatorname{relu}(z), \qquad L = \sum a$$

This is where the `sum -> broadcast` pattern from the joint-graph notes appears.

- $\dfrac{dL}{da} = \mathbf{1}$, the scalar loss broadcast to the output shape.
  This is exactly the `expand` we saw in the backward graph.
- $\dfrac{da}{dz} = \operatorname{relu}'(z)$, elementwise.
- $\dfrac{dL}{dz} = \dfrac{dL}{da} \odot \operatorname{relu}'(z)$, the
  elementwise product.

Then the gradients:

$$\frac{dL}{dx} = W^\top \frac{dL}{dz}$$

$$\frac{dL}{dW} = \frac{dL}{dz}\, x^\top$$

$$\frac{dL}{db} = \frac{dL}{dz}$$

The key idea: reverse mode never builds the full Jacobian $W$. It only computes
the product "upstream gradient $\times$ local Jacobian", which is the VJP. The
upstream gradient is what AOTAutograd calls the tangent (`tangents_1`).

## 5. Two-layer network

$$h = \operatorname{relu}(W_1 x + b_1), \qquad y = W_2 h + b_2, \qquad L = \sum y$$

Backward, in reverse order:

1. $\dfrac{dL}{dy} = \mathbf{1}$
2. $\dfrac{dL}{db_2} = \dfrac{dL}{dy}$
3. $\dfrac{dL}{dW_2} = \dfrac{dL}{dy}\, h^\top$
4. $\dfrac{dL}{dh} = W_2^\top \dfrac{dL}{dy}$
5. $\dfrac{dL}{dz_1} = \dfrac{dL}{dh} \odot \operatorname{relu}'(W_1 x + b_1)$
6. $\dfrac{dL}{db_1} = \dfrac{dL}{dz_1}$
7. $\dfrac{dL}{dW_1} = \dfrac{dL}{dz_1}\, x^\top$
8. $\dfrac{dL}{dx} = W_1^\top \dfrac{dL}{dz_1}$

Every layer applies the same rule: multiply the incoming (upstream) gradient by
that layer's local Jacobian. The only new piece in a deep network is that the
result of one layer becomes the upstream gradient for the next.

## 6. Concrete vector walkthrough

Use the identity weights so the numbers stay simple:

$$x = \begin{bmatrix}1 \\ 2\end{bmatrix}, \qquad
W = \begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix}, \qquad
b = \begin{bmatrix}0 \\ 0\end{bmatrix}$$

Forward:

$$z = Wx + b = \begin{bmatrix}1 \\ 2\end{bmatrix}, \qquad
a = \operatorname{relu}(z) = \begin{bmatrix}1 \\ 2\end{bmatrix}, \qquad
L = \sum a = 3$$

Backward:

$$\frac{dL}{da} = \begin{bmatrix}1 \\ 1\end{bmatrix}
\quad \text{(broadcast of the scalar loss gradient, the `expand`)}$$

$$\operatorname{relu}'(z) = \begin{bmatrix}1 \\ 1\end{bmatrix}, \qquad
\frac{dL}{dz} = \begin{bmatrix}1 \\ 1\end{bmatrix}$$

$$\frac{dL}{dx} = W^\top \begin{bmatrix}1 \\ 1\end{bmatrix}
= \begin{bmatrix}1 \\ 1\end{bmatrix}$$

$$\frac{dL}{dW} = \begin{bmatrix}1 \\ 1\end{bmatrix}
\begin{bmatrix}1 & 2\end{bmatrix}
= \begin{bmatrix}1 & 2 \\ 1 & 2\end{bmatrix}$$

$$\frac{dL}{db} = \begin{bmatrix}1 \\ 1\end{bmatrix}$$

Sanity check on $L$: increasing $x_1$ by $dx$ increases $z_1$ by $dx$, $a_1$ by
$dx$, and $L$ by $dx$, so $\dfrac{dL}{dx_1} = 1$. The same holds for $x_2$, so
$\dfrac{dL}{dx} = [1, 1]^\top$.

## 7. Vector and matrix handling formulas

The one rule that covers most of deep learning is the **vector-Jacobian
product**. If $L$ is a scalar and $z = z(x)$ is a vector:

$$\nabla_x L = \left(\frac{\partial z}{\partial x}\right)^\top \nabla_z L$$

Here $\dfrac{\partial z}{\partial x}$ is the Jacobian and $\nabla_z L$ is the
upstream gradient. Read it right to left: upstream gradient, then multiply by
the transposed Jacobian.

### Linear layer: $z = Wx + b$

Shapes: $x\in\mathbb{R}^n$, $W\in\mathbb{R}^{m\times n}$, $b,z\in\mathbb{R}^m$.

| Derivative | Formula | Shape |
| --- | --- | --- |
| $\dfrac{\partial z}{\partial x}$ | $W$ | $m\times n$ |
| $\nabla_x L$ | $W^\top \nabla_z L$ | $n$ |
| $\nabla_W L$ | $(\nabla_z L)\, x^\top$ | $m\times n$ |
| $\nabla_b L$ | $\nabla_z L$ | $m$ |

### Elementwise nonlinearity: $a = \operatorname{relu}(z)$

$$\nabla_z L = \nabla_a L \odot \operatorname{relu}'(z)$$

The $\odot$ is the elementwise product. This is the `where`/`le` gate in the
backward graph.

### Common matrix identities

For a scalar $L$:

- $\dfrac{\partial}{\partial x}\left(a^\top x\right) = a$
- $\dfrac{\partial}{\partial x}\left(x^\top A x\right) = \left(A + A^\top\right) x$,
  which is $2Ax$ when $A$ is symmetric
- $\dfrac{\partial}{\partial W}\left\|Wx - y\right\|^2 = 2\,(Wx - y)\, x^\top$
- $\dfrac{\partial}{\partial W}\operatorname{tr}\left(W^\top W\right) = 2W$
- $\dfrac{\partial}{\partial X}\log\left|\det X\right| = \left(X^{-1}\right)^\top$

### Shape check

A scalar derivative has the same shape as its variable: $\nabla_x L$ is shaped
like $x$, and $\nabla_W L$ is shaped like $W$. Verify every step by dimensions:

$$(n,) = (m\times n)^\top \cdot (m,)$$

$$(m\times n) = (m,) \cdot (n,)^\top$$

## Why reverse mode is the chain rule backward

Forward mode computes $\dfrac{dy}{dx}$ by carrying a Jacobian forward and costs
$O(n)$ passes for $n$ inputs. Reverse mode instead carries the upstream gradient
backward and computes every input gradient in one pass, at roughly the cost of
one forward plus one backward pass.

The backward graphs printed by AOTAutograd are literally these VJP steps:
`expand` is the broadcast of a scalar loss gradient, `mm`/`permute` are the
$W^\top$ parts, and the elementwise `where`/`le` are the $\operatorname{relu}'$
gate.

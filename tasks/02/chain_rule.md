# Chain rule: how gradients flow through ops

Backpropagation is the chain rule applied backward. Each operation receives an
**upstream gradient** (the tangent, `tangents_1`) and returns gradients for its
own inputs. That single step is the vector-Jacobian product (VJP), and the rest
is just composing those steps.

It is the math behind the `tangents_1` and `expand` pieces in
`tasks/02/joint_graph_explained.md`.

## Notation

- Plain letters are scalars ($x$, $z$), bold lowercase are vectors
  ($\mathbf{x}$, $\mathbf{y}$), and bold uppercase are matrices
  ($\mathbf{X}$, $\mathbf{Y}$, $\mathbf{Z}$).
- $\odot$ — element-wise product
- $\mathbf{x}^\top \mathbf{y}$ — dot product
- $\mathbf{x}\mathbf{y}^\top$ — outer product
- $\lVert \mathbf{X} \rVert_F$ — Frobenius norm
- $\langle \mathbf{A}, \mathbf{B} \rangle$ — Frobenius inner product,
  $\sum_{i,j} A_{ij} B_{ij} = \operatorname{tr}(\mathbf{A}^\top \mathbf{B})$

## The master rule

For $z = f(x)$:

$$\nabla_x L = \left(\frac{\partial z}{\partial x}\right)^\top \nabla_z L$$

Read it right to left: upstream gradient, times the transposed Jacobian. Reverse
mode never materializes the full Jacobian; it only computes this product.

**Note:** $\nabla_z L$ has the same shape as $z$, the output, and $\nabla_x L$
has the same shape as $x$, the input.

### Examples

Scalar to scalar:

$$z = f(x), \qquad \nabla_x L = f'(x)\,\nabla_z L$$

Scalar to vector, vector to vector:

$$\mathbf{z} = x\,\mathbf{y}, \qquad \nabla_x L = \mathbf{y}^\top \nabla_{\mathbf{z}} L, \qquad \nabla_{\mathbf{y}} L = x\,\nabla_{\mathbf{z}} L$$

Scalar to matrix:

$$\mathbf{Z} = x\,\mathbf{Y}, \qquad \nabla_x L = \langle \mathbf{Y}, \nabla_{\mathbf{Z}} L \rangle, \qquad \nabla_{\mathbf{Y}} L = x\,\nabla_{\mathbf{Z}} L$$

Vector to scalar:

$$z = \mathbf{x}^\top \mathbf{y}, \qquad \nabla_{\mathbf{x}} L = \mathbf{y}\,\nabla_z L, \qquad \nabla_{\mathbf{y}} L = \mathbf{x}\,\nabla_z L$$

Vector to matrix:

$$\mathbf{Z} = \mathbf{x}\mathbf{y}^\top, \qquad \nabla_{\mathbf{x}} L = \nabla_{\mathbf{Z}} L\,\mathbf{y}, \qquad \nabla_{\mathbf{y}} L = \nabla_{\mathbf{Z}}^\top L\,\mathbf{x}$$

Matrix to scalar:

$$z = \lVert \mathbf{X} \rVert_F^2 = \sum_{i,j} X_{ij}^2, \qquad \nabla_{\mathbf{X}} L = 2\,\mathbf{X}\,\nabla_z L$$

Matrix to vector:

$$\mathbf{z} = \mathbf{X}\mathbf{y}, \qquad \nabla_{\mathbf{X}} L = \nabla_{\mathbf{z}} L\,\mathbf{y}^\top, \qquad \nabla_{\mathbf{y}} L = \mathbf{X}^\top \nabla_{\mathbf{z}} L$$

Matrix to matrix:

$$\mathbf{Z} = \mathbf{X}\mathbf{Y}, \qquad \nabla_{\mathbf{X}} L = \nabla_{\mathbf{Z}} L\,\mathbf{Y}^\top, \qquad \nabla_{\mathbf{Y}} L = \mathbf{X}^\top \nabla_{\mathbf{Z}} L$$

Transpose:

$$\mathbf{Z} = \mathbf{X}^\top, \qquad \nabla_{\mathbf{X}} L = \left(\nabla_{\mathbf{Z}} L\right)^\top$$

View (reshape):

$$\mathbf{z} = \operatorname{view}\left(\mathbf{x},\ \text{new shape}\right), \qquad \nabla_{\mathbf{x}} L = \operatorname{view}\left(\nabla_{\mathbf{z}} L,\ \text{shape of } \mathbf{x}\right)$$

Element-wise add:

$$\mathbf{z} = \mathbf{x} + \mathbf{y}, \qquad \nabla_{\mathbf{x}} L = \nabla_{\mathbf{z}} L, \qquad \nabla_{\mathbf{y}} L = \nabla_{\mathbf{z}} L$$

Element-wise multiply:

$$\mathbf{z} = \mathbf{x} \odot \mathbf{y}, \qquad \nabla_{\mathbf{x}} L = \nabla_{\mathbf{z}} L \odot \mathbf{y}, \qquad \nabla_{\mathbf{y}} L = \nabla_{\mathbf{z}} L \odot \mathbf{x}$$

Reduce sum:

$$z = \sum \mathbf{x}, \qquad \nabla_{\mathbf{x}} L = \operatorname{expand}\left(\nabla_z L,\ \text{shape of } \mathbf{x}\right)$$

Expand (broadcast):

$$\mathbf{z} = \operatorname{expand}\left(\mathbf{x},\ \text{target shape}\right), \qquad \nabla_{\mathbf{x}} L = \sum_{\text{broadcast dims}} \nabla_{\mathbf{z}} L$$

ReLU:

$$\mathbf{z} = \operatorname{relu}(\mathbf{x}), \qquad \nabla_{\mathbf{x}} L = \nabla_{\mathbf{z}} L \odot \mathbf{1}[\mathbf{x} > 0]$$

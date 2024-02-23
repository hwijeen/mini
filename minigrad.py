from collections import defaultdict
from typing import Optional, Tuple

import numpy as np
import torch


class Tensor:
    def __init__(self, data, grad_fn=None, inputs=[]):
        self.data = data
        self.inputs = inputs
        self.grad_fn = grad_fn
        self.grad = None

    def __repr__(self):
        return f"Tensor({self.data})"

    def __add__(self, other):
        return Add()(self, other)

    def __matmul__(self, other):
        return MatMul()(self, other)

    def relu(self):
        return Relu()(self)

    def backward(self):
        out_grad = np.ones_like(self.data)
        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for n in node.inputs:
                    build_topo(n)
                topo.append(node)

        build_topo(self)

        node_to_grad = defaultdict(list)  # gradient could come from multiple
        node_to_grad[self].append(out_grad)
        for node in reversed(topo):
            node.grad = node_to_grad[node]
            for idx, n in enumerate(node.inputs):
                node.grad = sum(node_to_grad[node])
                in_grad = node.grad_fn.backward(out_grad)
                node_to_grad[n].append(in_grad[idx])


class Function:
    def __call__(self, *args) -> Tensor:
        """Ensures that the inputs and outputs of the function are tensors"""
        self.inputs = args
        out = self._compute(*[x.data for x in args])
        return Tensor(out, self, args)

    def backward(self, grad_out: Tensor) -> np.ndarray:
        """Ensures that the input and outputs are tensors"""
        grad_inputs = self._gradient(grad_out.data)
        return grad_inputs

    def _compute(self):
        """Calculates forward using numpy"""
        raise NotImplementedError

    def _gradient(self):
        """Calculates backward using numpy"""
        raise NotImplementedError


class Add(Function):
    def _compute(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x + y

    def _gradient(self, out_grad: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return out_grad, out_grad


class MatMul(Function):
    def _compute(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return x @ y

    def _gradient(self, out_grad: np.ndarray) -> Tuple[np.ndarray]:
        x, y = self.inputs
        return out_grad @ y.data.T, x.data.T @ out_grad


class Relu(Function):
    def _compute(self, x: np.ndarray) -> np.ndarray:
        return x * (x > 0)

    def _gradient(self, out_grad: np.ndarray) -> Tuple[np.ndarray]:
        x = self.inputs[0]
        return (out_grad * (x.data > 0),)


if __name__ == "__main__":
    x = Tensor(np.array([[1.0, 2.0]]))
    y = Tensor(np.array([[-5.0], [6.0]]))
    z = x @ y
    q = z.relu()
    q.backward()

    x_pt = torch.tensor(np.array([[1.0, 2.0]]), requires_grad=True)
    y_pt = torch.tensor(np.array([[-5.0], [6.0]]), requires_grad=True)
    z_pt = x_pt @ y_pt
    q_pt = z_pt.relu()
    q_pt.backward()

    assert np.allclose(q.data, q_pt.data.numpy())
    assert np.allclose(x.grad, x_pt.grad.numpy())
    assert np.allclose(y.grad, y_pt.grad.numpy())

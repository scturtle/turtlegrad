from typing import Optional, Tuple

import numpy as np

from . import op


class Tensor:
    def __init__(
        self, data: np.ndarray, op: Optional[op.Op] = None, requires_grad=True
    ):
        if not isinstance(data, np.ndarray):
            data = np.array([data], dtype=np.float32)
        self.data: np.ndarray = data
        self.grad = None
        self.op = op
        self.requires_grad = requires_grad

    def __str__(self):
        return f"Tensor(data={self.data}, grad={self.grad}, op={self.op})"

    @property
    def shape(self):
        return self.data.shape

    @staticmethod
    def _ensure_tensor(val):
        return val if isinstance(val, Tensor) else Tensor(val)

    def __add__(self, other):
        other = self._ensure_tensor(other)
        return op.Add(self, other).tensor()

    def __mul__(self, other):
        other = self._ensure_tensor(other)
        return op.Mul(self, other).tensor()

    def __pow__(self, other):
        other = self._ensure_tensor(other)
        return op.Pow(self, other).tensor()

    def __neg__(self):  # -self
        return op.Neg(self).tensor()

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return other + (-self)

    def __rmul__(self, other):  # other * self
        return self * other

    def __truediv__(self, other):  # self / other
        return self * (other**-1)

    def __rtruediv__(self, other):  # other / self
        return other * (self**-1)

    def __getitem__(self, slices):
        return op.Slice(self, slices=slices).tensor()

    def transpose(self) -> "Tensor":
        return op.Transpose(self).tensor()

    def reshape(self, shape: Tuple[int]) -> "Tensor":
        return op.Reshape(self, shape=shape).tensor()

    def max(self, axis=None, keepdims=False) -> "Tensor":
        return op.Max(self, axis=axis, keepdims=keepdims).tensor()

    def min(self, axis=None, keepdims=False) -> "Tensor":
        return op.Min(self, axis=axis, keepdims=keepdims).tensor()

    def exp(self) -> "Tensor":
        return op.Exp(self).tensor()

    def sigmoid(self) -> "Tensor":
        return op.Sigmoid(self).tensor()

    def tanh(self) -> "Tensor":
        return op.Tanh(self).tensor()

    def relu(self) -> "Tensor":
        return op.Relu(self).tensor()

    def matmul(self, other) -> "Tensor":
        return op.Matmul(self, other).tensor()

    def __matmul__(self, other):
        return self.matmul(other)

    def log_softmax(self) -> "Tensor":
        m = self.max(axis=1, keepdims=True)
        return self - m - (self - m).exp().sum(axis=1, keepdims=True).log()

    def log(self) -> "Tensor":
        return op.Log(self).tensor()

    def sum(self, axis=None, keepdims=False) -> "Tensor":
        return op.Sum(self, axis=axis, keepdims=keepdims).tensor()

    def conv2d(self, weight, stride=1) -> "Tensor":
        return op.Conv2d(self, weight, stride=stride).tensor()

    def pad2d(self, pad=1):
        return op.Pad2d(self, pad=pad).tensor()

    def backprop(self):
        def walk(v, visited, topo):
            if v in visited:
                return
            visited.add(v)
            if v.op and v.requires_grad:
                for prev in v.op.inputs:
                    walk(prev, visited, topo)
                topo.append(v)

        topo = []
        walk(self, set(), topo)

        for v in topo:
            for prev in v.op.inputs:
                prev.grad = np.zeros(prev.shape).astype(np.float32)
            v.grad = None
        assert self.data.shape == (1,), self
        self.grad = np.array([1.0], dtype=np.float32)

        for v in reversed(topo):
            grads = v.op.backward()
            grads = (grads,) if not isinstance(grads, tuple) else grads
            for prev, grad in zip(v.op.inputs, grads):
                prev.grad += grad

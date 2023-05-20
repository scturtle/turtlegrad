from typing import TYPE_CHECKING, Dict, List, Tuple, Union

import numpy as np

from .utils import unbroadcast

if TYPE_CHECKING:
    from .tensor import Tensor


class Op:
    def __init__(self, *inputs, **args):
        self.inputs: List["Tensor"] = inputs
        self.args: Dict = args
        self.output: "Tensor"

    def forward(self) -> np.ndarray:
        raise NotImplementedError()

    def backward(self) -> Union[np.ndarray, Tuple[np.ndarray]]:
        raise NotImplementedError()

    def tensor(self) -> "Tensor":
        from .tensor import Tensor

        out = Tensor(data=self.forward(), op=self)
        out.requires_grad = any(i.requires_grad for i in self.inputs)
        self.output = out
        return out


class Add(Op):
    def forward(self):
        return self.inputs[0].data + self.inputs[1].data

    def backward(self):
        a, b = self.inputs
        a_grad = unbroadcast(self.output.grad, a.shape)
        b_grad = unbroadcast(self.output.grad, b.shape)
        return a_grad, b_grad


class Mul(Op):
    def forward(self):
        return self.inputs[0].data * self.inputs[1].data

    def backward(self):
        a, b = self.inputs
        a_grad = unbroadcast(b.data * self.output.grad, a.shape)
        b_grad = unbroadcast(a.data * self.output.grad, b.shape)
        return a_grad, b_grad


class Neg(Op):
    def forward(self):
        return -self.inputs[0].data

    def backward(self):
        return -self.output.grad


class Pow(Op):
    def forward(self):
        from .tensor import Tensor

        assert isinstance(self.inputs[1], Tensor)
        return self.inputs[0].data ** self.inputs[1].data

    def backward(self):
        a, b = self.inputs[0], self.inputs[1]
        a_grad = unbroadcast(
            b.data * a.data ** (b.data - 1) * self.output.grad, a.shape
        )
        b_grad = unbroadcast(
            a.data**b.data * np.log(a.data) * self.output.grad, b.shape
        )
        return a_grad, b_grad


class Log(Op):
    def forward(self):
        return np.log(self.inputs[0].data)

    def backward(self):
        return self.output.grad / self.inputs[0].data


class Exp(Op):
    def forward(self) -> np.ndarray:
        return np.exp(self.inputs[0].data)

    def backward(self):
        return np.exp(self.inputs[0].data) * self.output.grad


class Transpose(Op):
    def forward(self):
        return self.inputs[0].T

    def backward(self):
        return self.output.grad.T


class Reshape(Op):
    def forward(self):
        shape = self.args["shape"]
        return self.inputs[0].data.reshape(shape)

    def backward(self):
        shape = self.inputs[0].shape
        return self.output.grad.reshape(shape)


class Relu(Op):
    def forward(self):
        return np.maximum(self.inputs[0].data, 0)

    def backward(self):
        return (self.output.data > 0) * self.output.grad


class Matmul(Op):
    def forward(self):
        return self.inputs[0].data @ self.inputs[1].data

    def backward(self):
        a, b = self.inputs
        a_grad = self.output.grad @ b.data.T
        b_grad = a.data.T @ self.output.grad
        return a_grad, b_grad


class Sigmoid(Op):
    def forward(self):
        return self.sigmoid(self.inputs[0].data)

    def backward(self):
        (x,) = self.inputs
        return self.sigmoid(x.data) * (1 - self.sigmoid(x.data)) * self.output.grad

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))


class Tanh(Op):
    def forward(self):
        return np.tanh(self.inputs[0].data)

    def backward(self):
        (x,) = self.inputs
        return (1 - np.tanh(x.data) ** 2) * self.output.grad


class Sum(Op):
    def forward(self):
        axis = self.args["axis"]
        keepdims = self.args["keepdims"]
        x = self.inputs[0].data
        if axis is None:
            return np.array([x.sum()])
        axis = (axis,) if isinstance(axis, int) else tuple(axis)
        return x.sum(axis=axis, keepdims=keepdims)

    def backward(self):
        axis = self.args["axis"]
        if axis is None:
            output_shape = [1]
        else:
            output_shape = list(self.inputs[0].shape)
            axis = (axis,) if isinstance(axis, int) else tuple(axis)
            for i in axis:
                output_shape[i] = 1
        grad = self.output.grad.reshape(output_shape)
        return grad + np.zeros_like(self.inputs[0].data)


class Max(Op):
    def forward(self):
        axis = self.args["axis"]
        keepdims = self.args["keepdims"]
        x = self.inputs[0].data
        if axis is None:
            return np.array([x.sum()])
        axis = (axis,) if isinstance(axis, int) else tuple(axis)
        return x.max(axis=axis, keepdims=keepdims)

    def backward(self):
        axis = self.args["axis"]
        if axis is None:
            shape = [1]
        else:
            shape = list(self.inputs[0].shape)
            axis = (axis,) if isinstance(axis, int) else tuple(axis)
            for i in axis:
                shape[i] = 1
        res = self.output.data.reshape(shape)
        selected = self.inputs[0].data == res
        div = selected.sum(axis, keepdims=True)
        grad = selected * self.output.grad.reshape(shape) / div
        return grad + np.zeros_like(self.inputs[0].data)


class Min(Max):
    def forward(self):
        axis = self.args["axis"]
        keepdims = self.args["keepdims"]
        x = self.inputs[0].data
        if axis is None:
            return np.array([x.sum()])
        axis = (axis,) if isinstance(axis, int) else tuple(axis)
        return x.min(axis=axis, keepdims=keepdims)


class Slice(Op):
    def forward(self):
        slices = self.args["slices"]
        return self.inputs[0].data[slices]

    def backward(self):
        slices = self.args["slices"]
        grad = np.zeros_like(self.inputs[0].data)
        grad[slices] = self.output.grad
        return grad


class Pad2d(Op):
    def forward(self):
        x = self.inputs[0].data
        pad = self.args["pad"]
        return np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), "constant")

    def backward(self):
        pad = self.args["pad"]
        return self.output.grad[:, :, pad:-pad, pad:-pad]


class Conv2d(Op):
    @staticmethod
    def get_output_shape(x, kernel, stride):
        (_, _, ih, iw), (kh, kw) = x.shape, kernel
        oh = (ih - kh) // stride + 1
        ow = (iw - kw) // stride + 1
        return oh, ow

    def forward(self):
        x = self.inputs[0].data
        weight = self.inputs[1].data
        n, ic, _, _ = x.shape
        _, ic2, kh, kw = weight.shape
        assert ic == ic2
        stride = self.args["stride"]
        oh, ow = self.get_output_shape(x, (kh, kw), stride)
        ns, ics, ihs, iws = x.strides
        x_col = np.lib.stride_tricks.as_strided(
            x,
            shape=(ic, kh, kw, n, oh, ow),
            strides=(ics, ihs, iws, ns, stride * ihs, stride * iws),
            writeable=False,
        )
        self.saved = x_col
        r = np.tensordot(weight, x_col, ((1, 2, 3), (0, 1, 2)))
        r = r.transpose(1, 0, 2, 3)
        return r

    def backward(self):
        x = self.inputs[0].data
        weight = self.inputs[1].data
        _, _, kh, kw = weight.shape
        stride = self.args["stride"]
        grad = self.output.grad
        oh, ow = self.output.shape[2:]
        x_col = self.saved
        grad_w = np.tensordot(grad, x_col, ((0, 2, 3), (3, 4, 5)))
        grad_x = np.zeros_like(x)
        for i in range(oh):
            for j in range(ow):
                ii, jj = i * stride, j * stride
                grad_x[:, :, ii : ii + kh, jj : jj + kw] += np.tensordot(
                    grad[:, :, i, j], weight, ((1,), (0,))
                )
        return (grad_x, grad_w)

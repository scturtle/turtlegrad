import numpy as np

from .tensor import Tensor
from .utils import init_weights


class Module:
    def __init__(self):
        self.modules = []
        self.trainable = []
        self.is_train = True

    def trainable_tensors(self):
        for submodule in self.modules:
            yield from submodule.trainable_tensors()
        for tensor in self.trainable:
            yield tensor

    def train(self):
        pass

    def evail(self):
        pass

    def __call__(self, *args):
        return self.forward(*args)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def train(self):
        self.is_train = True
        for submodule in self.modules:
            submodule.train()

    def eval(self):
        self.is_train = False
        for submodule in self.modules:
            submodule.eval()

    def forward(self, x: Tensor) -> Tensor:
        for module in self.modules:
            x = module(x)
        return x


class Linear(Module):
    def __init__(self, ic: int, oc: int, with_bias: bool = True):
        super().__init__()
        self.ic = ic
        self.oc = oc
        self.with_bias = with_bias
        self.weight = init_weights((ic, oc))
        self.trainable.append(self.weight)
        if with_bias:
            self.bias = Tensor(np.zeros(oc).astype(np.float32))
            self.trainable.append(self.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = x @ self.weight
        if self.with_bias:
            x = x + self.bias
        return x


class Conv(Module):
    def __init__(self, ic: int, oc: int, kernel: int, stride: int, pad: int):
        super().__init__()
        self.weight = init_weights((oc, ic, kernel, kernel))
        self.trainable.append(self.weight)
        self.bias = Tensor(np.zeros((1, oc, 1, 1)).astype(np.float32))
        self.trainable.append(self.bias)
        self.stride = stride
        self.pad = pad

    def forward(self, x: Tensor) -> Tensor:
        x = x.pad2d(pad=self.pad)
        x = x.conv2d(self.weight, stride=self.stride)
        x = x + self.bias
        return x


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()


class Flatten(Module):
    def forward(self, x: Tensor) -> Tensor:
        shape = x.shape
        return x.reshape((shape[0], np.prod(shape[1:])))

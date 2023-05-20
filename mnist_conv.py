#!/usr/bin/env python3
import numpy as np

from turtlegrad.module import Conv, Flatten, Linear, ReLU, Sequential
from turtlegrad.tensor import Tensor
from turtlegrad.utils import fetch_mnist, unbroadcast

x_train, y_train, x_test, y_test = fetch_mnist()
x_train = x_train.reshape(-1, 1, 28, 28)
x_test = x_test.reshape(-1, 1, 28, 28)
batch_size = 16
steps = 2000

net = Sequential(
    Conv(ic=1, oc=32, kernel=3, stride=1, pad=1),
    ReLU(),
    Conv(ic=32, oc=16, kernel=3, stride=1, pad=1),
    ReLU(),
    Flatten(),
    Linear(16 * 28 * 28, 10),
)

x_train = Tensor(x_train, requires_grad=False)
y_train = Tensor(y_train, requires_grad=False)
for step in range(steps):
    ri = np.random.permutation(x_train.shape[0])[:batch_size]
    Xb = x_train[ri]
    yb = y_train[ri]

    prob = net(Xb).log_softmax()
    outb = -(yb * prob).sum()  # cross entropy loss

    outb.backprop()
    for p in net.trainable_tensors():
        p.data -= unbroadcast(0.01 * p.grad, p.shape)

    print(f"train loss in step {step:>6} is {outb.data[0] / Xb.shape[0]:.8f}")

y_pred = np.argmax(net(Tensor(x_test)).data, axis=1)
y_lbls = np.argmax(y_test, axis=1)
score = np.mean(np.equal(y_lbls, y_pred))
print(f"score {score * 100:.2f}%")

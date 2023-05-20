#!/usr/bin/env python3
import numpy as np

from turtlegrad.module import Linear, ReLU, Sequential
from turtlegrad.tensor import Tensor
from turtlegrad.utils import fetch_mnist, unbroadcast

x_train, y_train, x_test, y_test = fetch_mnist()
batch_size = 64
steps = 20000

net = Sequential(
    Linear(784, 128),
    ReLU(),
    Linear(128, 10),
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

    if step % 1000 == 0:  # average cross-entropy loss on training set
        prob = net(x_train).log_softmax()
        loss = -(y_train * prob).sum()
        avg_loss = loss.data[0] / x_train.shape[0]
        print(f"loss in step {step:>6} is {avg_loss:.8f}")

y_pred = np.argmax(net(Tensor(x_test)).data, axis=1)
y_lbls = np.argmax(y_test, axis=1)
score = np.mean(np.equal(y_lbls, y_pred))
print(f"score {score * 100:.2f}%")

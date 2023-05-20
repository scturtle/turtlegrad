from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from .tensor import Tensor


def fetch_mnist():
    import gzip
    import urllib.request
    from pathlib import Path

    datadir = Path("datasets")
    datadir.mkdir(exist_ok=True)
    for filename in [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]:
        file = datadir / filename
        if not file.exists():
            url = "https://github.com/geohot/tinygrad/raw/master/datasets/mnist/"
            resp = urllib.request.urlopen(url + filename)
            file.write_bytes(resp.read())
        arr = np.frombuffer(gzip.open(file).read(), dtype=np.uint8)
        if "images" in filename:
            yield arr[16:].reshape(-1, 784).astype(np.float32) / 255.0
        else:
            yield np.eye(10, dtype=np.float32)[arr[8:]]


def unbroadcast(grad: np.ndarray, shape: Tuple[int]) -> np.ndarray:
    # unbroadcast for grad is a sum
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    assert grad.ndim == len(shape)
    for i in range(grad.ndim):
        if grad.shape[i] > shape[i]:
            assert shape[i] == 1
            grad = grad.sum(axis=i, keepdims=True)
    return grad


def init_weights(shape: Tuple[int]) -> "Tensor":
    from .tensor import Tensor

    std = np.sqrt(2.0 / (shape[0] + shape[1]))
    return Tensor(np.random.uniform(-std, std, shape).astype(np.float32))

# coding: utf-8
from common.np import *  # import numpy as np
from common.config import GPU
from common.functions import softmax, cross_entropy_error


# 矩阵乘积称为 MatMul 节点
# np.dot(x, W) 实现了 MatMul 层
class MatMul:

    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        # grads[0] = dW 的赋值相当于浅复制
        # grads[0][...] = dW 的覆盖相当于深复制
        self.grads[0][...] = dW
        return dx


# np.dot(x, W) + b 实现了 Affine 层
class Affine:

    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        self.x = x
        return out

    def backward(self, dout):
        W, b = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class Softmax:

    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx


# Softmax 函数和交叉熵误差一起实现为 Softmax with Loss 层
class SoftmaxWithLoss:

    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmax的输出
        self.t = None  # 监督标签

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 在监督标签为one-hot向量的情况下, 转换为正确解标签的索引
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx


# Sigmoid
class Sigmoid:

    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * (1.0 - self.out) * self.out
        return dx


class SigmoidWithLoss:

    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None  # sigmoid的输出
        self.t = None  # 监督标签

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))

        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = (self.y - self.t) * dout / batch_size
        return dx


class Dropout:
    '''
    http://arxiv.org/abs/1207.0580
    '''

    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None

    def forward(self, x, train_flg=True):
        if train_flg:
            self.mask = np.random.rand(*x.shape) > self.dropout_ratio
            return x * self.mask
        else:
            return x * (1.0 - self.dropout_ratio)

    def backward(self, dout):
        return dout * self.mask


class Embedding:

    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        # print("Embedding forward idx:", idx)
        # Embedding forward idx: [4 1 1]

        W, = self.params
        # print("Embedding forward W:", W)
        # print("Embedding forward W:", W.shape)
        # Embedding forward W:
        # [
        #     [-0.00264533  0.00962379  0.00976532 -0.00759188  0.003047  ]
        #     [ 0.01875792  0.00838653  0.01763416 -0.00275273 -0.01376551]
        #     [ 0.00417071 -0.02550101 -0.00050339  0.0208165  -0.010699  ]
        #     [-0.0036964  -0.00669661  0.0057202  -0.00550012 -0.00081953]
        #     [ 0.00830469  0.01215741  0.0028348   0.0058751  -0.01643123]
        #     [-0.00399287  0.01282025 -0.01045556  0.01149138  0.00257922]
        #     [-0.00461878 -0.01553948 -0.00872226  0.00758368  0.00033413]
        # ]
        # Embedding forward W: (7, 5)

        self.idx = idx
        out = W[idx]
        # print("Embedding forward out:", out)
        # print("Embedding forward out:", out.shape)
        # Embedding forward out:
        # [
        #     [ 0.00830469  0.01215741  0.0028348   0.0058751  -0.01643123]
        #     [ 0.01875792  0.00838653  0.01763416 -0.00275273 -0.01376551]
        #     [ 0.01875792  0.00838653  0.01763416 -0.00275273 -0.01376551]
        # ]
        # Embedding forward out: (3, 5)

        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        if GPU:
            np.scatter_add(dW, self.idx, dout)
        else:
            np.add.at(dW, self.idx, dout)
        return None

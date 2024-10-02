# coding: utf-8
import sys

sys.path.append('..')
import numpy as np

from common.np import *  # import numpy as np (or import cupy as np)
from common.layers import *
from common.functions import softmax, sigmoid


class RNN:

    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        # print("RNN x:", x.shape)  # RNN x: (5, 5)
        # print("RNN h_prev:", h_prev.shape)  # RNN h_prev: (5, 5)
        Wx, Wh, b = self.params
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
        # print("RNN h_next:", h_next.shape)
        return h_next

    def backward(self, dh_next):
        Wx, Wh, b = self.params
        x, h_prev, h_next = self.cache

        dt = dh_next * (1 - h_next**2)
        db = np.sum(dt, axis=0)
        dWh = np.dot(h_prev.T, dt)
        dh_prev = np.dot(dt, Wh.T)
        dWx = np.dot(x.T, dt)
        dx = np.dot(dt, Wx.T)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        return dx, dh_prev


class TimeRNN:

    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.dh = None, None
        self.stateful = stateful

    def forward(self, xs):
        """
        批大小是 N
        输入向量的维数是 D
        隐藏状态向量的维数是 H

        Truncated BPTT 的时间跨度大小是 T
        Time RNN 层由 T 个 RNN 层构成(T 可以设置为任意值)
        """
        # print("TimeRNN xs:", xs.shape)  # TimeRNN xs: (10, 50, 100)
        Wx, Wh, b = self.params
        N, T, D = xs.shape  # (10, 50, 100)
        # print(N, T, D)
        D, H = Wx.shape  # (100, 100)
        # print(D, H)

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')  # (10, 50, 100)
        # print("TimeRNN hs:", hs.shape)

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')  # (10, 100)
            # print("TimeRNN self.h:", self.h.shape)

        for t in range(T):
            # print("##" * 20, "t = ", t)
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            # print("TimeRNN hs:", hs)
            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D, H = Wx.shape

        dxs = np.empty((N, T, D), dtype='f')
        dh = 0
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh

        return dxs

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None


class LSTM:

    def __init__(self, Wx, Wh, b):
        '''
        Parameters
        ----------
        Wx: 输入 `x` 用的权重参数(整合了 4 个权重)
        Wh: 隐藏状态 `h` 用的权重参数(整合了 4 个权重)
        b: 偏置(整合了 4 个偏置)

        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')  # Rnnlm predict lstm_Wx: (100, 400)
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')  # Rnnlm predict lstm_Wh: (100, 400)
        lstm_b = np.zeros(4 * H).astype('f')  # Rnnlm predict lstm_b: (400,)
        '''
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev, c_prev):
        Wx, Wh, b = self.params
        """
        批大小是 N
        输入数据的维数是 D
        记忆单元和隐藏状态的维数都是 H
        """
        # print("LSTM forward x:", x.shape)  # LSTM forward x: (20, 100)
        # print("LSTM forward h_prev:", h_prev.shape)  # LSTM forward h_prev: (20, 100)
        # print("LSTM forward c_prev:", c_prev.shape)  # LSTM forward c_prev: (20, 100)
        N, H = h_prev.shape

        # (20, 100) * (100, 400) + (20, 100) * (100, 400) + ((400,))
        A = np.dot(x, Wx) + np.dot(h_prev, Wh) + b

        f = A[:, :H]
        # print("LSTM forward f:", f.shape)  # LSTM forward f: (20, 100)
        g = A[:, H:2 * H]
        # print("LSTM forward g:", g.shape)  # LSTM forward g: (20, 100)
        i = A[:, 2 * H:3 * H]
        # print("LSTM forward i:", i.shape)  # LSTM forward i: (20, 100)
        o = A[:, 3 * H:]
        # print("LSTM forward o:", o.shape)  # LSTM forward o: (20, 100)

        f = sigmoid(f)  # 遗忘门
        g = np.tanh(g)
        i = sigmoid(i)  # 输入门
        o = sigmoid(o)  # 输出门

        c_next = f * c_prev + g * i
        h_next = o * np.tanh(c_next)

        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        return h_next, c_next

    def backward(self, dh_next, dc_next):
        Wx, Wh, b = self.params
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache

        tanh_c_next = np.tanh(c_next)

        ds = dc_next + (dh_next * o) * (1 - tanh_c_next**2)

        dc_prev = ds * f

        di = ds * g
        df = ds * c_prev
        do = dh_next * tanh_c_next
        dg = ds * i

        di *= i * (1 - i)
        df *= f * (1 - f)
        do *= o * (1 - o)
        dg *= (1 - g**2)

        dA = np.hstack((df, dg, di, do))

        dWh = np.dot(h_prev.T, dA)
        dWx = np.dot(x.T, dA)
        db = dA.sum(axis=0)

        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db

        dx = np.dot(dA, Wx.T)
        dh_prev = np.dot(dA, Wh.T)

        return dx, dh_prev, dc_prev


class TimeLSTM:

    def __init__(self, Wx, Wh, b, stateful=False):
        """
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')  # Rnnlm predict lstm_Wx: (100, 400)
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')  # Rnnlm predict lstm_Wh: (100, 400)
        lstm_b = np.zeros(4 * H).astype('f')  # Rnnlm predict lstm_b: (400,)
        TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
        """
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None

        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful

    def forward(self, xs):
        # print("TimeLSTM forward xs:", xs.shape)  # TimeLSTM forward xs: (20, 35, 100)
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]
        # print(N, T, D, H)  # 20 35 100 100

        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
        if not self.stateful or self.c is None:
            self.c = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = LSTM(*self.params)
            self.h, self.c = layer.forward(xs[:, t, :], self.h, self.c)
            hs[:, t, :] = self.h

            self.layers.append(layer)

        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')
        dh, dc = 0, 0

        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh, dc = layer.backward(dhs[:, t, :] + dh, dc)
            dxs[:, t, :] = dx
            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad
        self.dh = dh
        return dxs

    def set_state(self, h, c=None):
        self.h, self.c = h, c

    def reset_state(self):
        self.h, self.c = None, None


class TimeEmbedding:

    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs):
        # print("TimeEmbedding xs:", xs.shape)  # TimeEmbedding xs: (10, 50)
        N, T = xs.shape  # (10, 50)
        V, D = self.W.shape  # (1002, 100)
        # print(N, T, V, D)  # 10 50 1002 100

        out = np.empty((N, T, D), dtype='f')  # (10, 50, 100)
        # print("TimeEmbedding out:", out.shape)  # TimeEmbedding out: (10, 50, 100)
        self.layers = []

        for t in range(T):
            # print("##" * 20, "t = ", t)
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)
            # print("TimeEmbedding out:", out)

        return out

    def backward(self, dout):
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]

        self.grads[0][...] = grad
        return None


class TimeAffine:

    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]
        self.x = None

    def forward(self, x):
        # print("TimeAffine x:", x.shape)  # TimeAffine x: (10, 50, 100)
        N, T, D = x.shape  # (10, 50, 100)
        W, b = self.params

        rx = x.reshape(N * T, -1)
        # print("TimeAffine rx:", rx.shape)  # TimeAffine rx: (500, 100)
        out = np.dot(rx, W) + b  # (500, 100) * (100, 1002) = (500, 1002)
        self.x = x
        return out.reshape(N, T, -1)

    def backward(self, dout):
        x = self.x
        N, T, D = x.shape
        W, b = self.params

        dout = dout.reshape(N * T, -1)
        rx = x.reshape(N * T, -1)

        db = np.sum(dout, axis=0)
        dW = np.dot(rx.T, dout)
        dx = np.dot(dout, W.T)
        dx = dx.reshape(*x.shape)

        self.grads[0][...] = dW
        self.grads[1][...] = db

        return dx


class TimeSoftmaxWithLoss:

    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.ignore_label = -1

    def forward(self, xs, ts):
        N, T, V = xs.shape

        if ts.ndim == 3:  # 在监督标签为one-hot向量的情况下
            ts = ts.argmax(axis=2)

        mask = (ts != self.ignore_label)

        # 按批次大小和时序大小进行整理（reshape）
        xs = xs.reshape(N * T, V)
        ts = ts.reshape(N * T)
        mask = mask.reshape(N * T)

        ys = softmax(xs)
        ls = np.log(ys[np.arange(N * T), ts])
        ls *= mask  # 与ignore_label相应的数据将损失设为0
        loss = -np.sum(ls)
        loss /= mask.sum()

        self.cache = (ts, ys, mask, (N, T, V))
        return loss

    def backward(self, dout=1):
        ts, ys, mask, (N, T, V) = self.cache

        dx = ys
        dx[np.arange(N * T), ts] -= 1
        dx *= dout
        dx /= mask.sum()
        dx *= mask[:, np.newaxis]  # 与ignore_label相应的数据将梯度设为0

        dx = dx.reshape((N, T, V))

        return dx


class TimeDropout:

    def __init__(self, dropout_ratio=0.5):
        self.params, self.grads = [], []
        self.dropout_ratio = dropout_ratio
        self.mask = None
        self.train_flg = True

    def forward(self, xs):
        if self.train_flg:
            flg = np.random.rand(*xs.shape) > self.dropout_ratio
            scale = 1 / (1.0 - self.dropout_ratio)
            self.mask = flg.astype(np.float32) * scale

            return xs * self.mask
        else:
            return xs

    def backward(self, dout):
        return dout * self.mask


class TimeBiLSTM:

    def __init__(self, Wx1, Wh1, b1, Wx2, Wh2, b2, stateful=False):
        self.forward_lstm = TimeLSTM(Wx1, Wh1, b1, stateful)
        self.backward_lstm = TimeLSTM(Wx2, Wh2, b2, stateful)
        self.params = self.forward_lstm.params + self.backward_lstm.params
        self.grads = self.forward_lstm.grads + self.backward_lstm.grads

    def forward(self, xs):
        o1 = self.forward_lstm.forward(xs)
        o2 = self.backward_lstm.forward(xs[:, ::-1])
        o2 = o2[:, ::-1]

        out = np.concatenate((o1, o2), axis=2)
        return out

    def backward(self, dhs):
        H = dhs.shape[2] // 2
        do1 = dhs[:, :, :H]
        do2 = dhs[:, :, H:]

        dxs1 = self.forward_lstm.backward(do1)
        do2 = do2[:, ::-1]
        dxs2 = self.backward_lstm.backward(do2)
        dxs2 = dxs2[:, ::-1]
        dxs = dxs1 + dxs2
        return dxs


# ====================================================================== #
# 如下所示的层是本书中没有说明的层的实现
# 或者为了使代码容易理解而牺牲了处理速度的层的实现。
#
# TimeSigmoidWithLoss: 用于时序数据的sigmoid损失层
# GRU: GRU层
# TimeGRU: 用于时序数据的GRU层
# BiTimeLSTM: 双向LSTM层
# Simple_TimeSoftmaxWithLoss：简单的TimeSoftmaxWithLoss层的实现
# Simple_TimeAffine: 简单的TimeAffine层的实现
# ====================================================================== #


class TimeSigmoidWithLoss:

    def __init__(self):
        self.params, self.grads = [], []
        self.xs_shape = None
        self.layers = None

    def forward(self, xs, ts):
        N, T = xs.shape
        self.xs_shape = xs.shape

        self.layers = []
        loss = 0

        for t in range(T):
            layer = SigmoidWithLoss()
            loss += layer.forward(xs[:, t], ts[:, t])
            self.layers.append(layer)

        return loss / T

    def backward(self, dout=1):
        N, T = self.xs_shape
        dxs = np.empty(self.xs_shape, dtype='f')

        dout *= 1 / T
        for t in range(T):
            layer = self.layers[t]
            dxs[:, t] = layer.backward(dout)

        return dxs


class GRU:

    def __init__(self, Wx, Wh, b):
        '''

        Parameters
        ----------
        Wx: 用于输入`x`的权重参数（整合了3个权重）
        Wh: 用于隐藏状态`h` 的权重参数（整合了3个权重）
        b: 偏置（整合了3个偏置）
        '''
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        Wx, Wh, b = self.params
        H = Wh.shape[0]
        Wxz, Wxr, Wxh = Wx[:, :H], Wx[:, H:2 * H], Wx[:, 2 * H:]
        Whz, Whr, Whh = Wh[:, :H], Wh[:, H:2 * H], Wh[:, 2 * H:]
        bz, br, bh = b[:H], b[H:2 * H], b[2 * H:]

        z = sigmoid(np.dot(x, Wxz) + np.dot(h_prev, Whz) + bz)
        r = sigmoid(np.dot(x, Wxr) + np.dot(h_prev, Whr) + br)
        h_hat = np.tanh(np.dot(x, Wxh) + np.dot(r * h_prev, Whh) + bh)
        h_next = (1 - z) * h_prev + z * h_hat

        self.cache = (x, h_prev, z, r, h_hat)

        return h_next

    def backward(self, dh_next):
        Wx, Wh, b = self.params
        H = Wh.shape[0]
        Wxz, Wxr, Wxh = Wx[:, :H], Wx[:, H:2 * H], Wx[:, 2 * H:]
        Whz, Whr, Whh = Wh[:, :H], Wh[:, H:2 * H], Wh[:, 2 * H:]
        x, h_prev, z, r, h_hat = self.cache

        dh_hat = dh_next * z
        dh_prev = dh_next * (1 - z)

        # tanh
        dt = dh_hat * (1 - h_hat**2)
        dbh = np.sum(dt, axis=0)
        dWhh = np.dot((r * h_prev).T, dt)
        dhr = np.dot(dt, Whh.T)
        dWxh = np.dot(x.T, dt)
        dx = np.dot(dt, Wxh.T)
        dh_prev += r * dhr

        # update gate(z)
        dz = dh_next * h_hat - dh_next * h_prev
        dt = dz * z * (1 - z)
        dbz = np.sum(dt, axis=0)
        dWhz = np.dot(h_prev.T, dt)
        dh_prev += np.dot(dt, Whz.T)
        dWxz = np.dot(x.T, dt)
        dx += np.dot(dt, Wxz.T)

        # rest gate(r)
        dr = dhr * h_prev
        dt = dr * r * (1 - r)
        dbr = np.sum(dt, axis=0)
        dWhr = np.dot(h_prev.T, dt)
        dh_prev += np.dot(dt, Whr.T)
        dWxr = np.dot(x.T, dt)
        dx += np.dot(dt, Wxr.T)

        self.dWx = np.hstack((dWxz, dWxr, dWxh))
        self.dWh = np.hstack((dWhz, dWhr, dWhh))
        self.db = np.hstack((dbz, dbr, dbh))

        self.grads[0][...] = self.dWx
        self.grads[1][...] = self.dWh
        self.grads[2][...] = self.db

        return dx, dh_prev


class TimeGRU:

    def __init__(self, Wx, Wh, b, stateful=False):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.layers = None
        self.h, self.dh = None, None
        self.stateful = stateful

    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        H = Wh.shape[0]
        self.layers = []
        hs = np.empty((N, T, H), dtype='f')

        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')

        for t in range(T):
            layer = GRU(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            self.layers.append(layer)
        return hs

    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]

        dxs = np.empty((N, T, D), dtype='f')

        dh = 0
        grads = [0, 0, 0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            dx, dh = layer.backward(dhs[:, t, :] + dh)
            dxs[:, t, :] = dx

            for i, grad in enumerate(layer.grads):
                grads[i] += grad

        for i, grad in enumerate(grads):
            self.grads[i][...] = grad

        self.dh = dh
        return dxs

    def set_state(self, h):
        self.h = h

    def reset_state(self):
        self.h = None


class Simple_TimeSoftmaxWithLoss:

    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, xs, ts):
        N, T, V = xs.shape
        layers = []
        loss = 0

        for t in range(T):
            layer = SoftmaxWithLoss()
            loss += layer.forward(xs[:, t, :], ts[:, t])
            layers.append(layer)
        loss /= T

        self.cache = (layers, xs)
        return loss

    def backward(self, dout=1):
        layers, xs = self.cache
        N, T, V = xs.shape
        dxs = np.empty(xs.shape, dtype='f')

        dout *= 1 / T
        for t in range(T):
            layer = layers[t]
            dxs[:, t, :] = layer.backward(dout)

        return dxs


class Simple_TimeAffine:

    def __init__(self, W, b):
        self.W, self.b = W, b
        self.dW, self.db = None, None
        self.layers = None

    def forward(self, xs):
        N, T, D = xs.shape
        D, M = self.W.shape

        self.layers = []
        out = np.empty((N, T, M), dtype='f')
        for t in range(T):
            layer = Affine(self.W, self.b)
            out[:, t, :] = layer.forward(xs[:, t, :])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, M = dout.shape
        D, M = self.W.shape

        dxs = np.empty((N, T, D), dtype='f')
        self.dW, self.db = 0, 0
        for t in range(T):
            layer = self.layers[t]
            dxs[:, t, :] = layer.backward(dout[:, t, :])

            self.dW += layer.dW
            self.db += layer.db

        return dxs


if __name__ == '__main__':
    embed_W = np.arange(125).reshape(25, 5)
    # print("embed_W:", embed_W)
    te = TimeEmbedding(embed_W)
    xs = np.arange(25).reshape(5, 5)
    # print("xs:", xs)
    # print("xs[:, 0]:", xs[:, 0])
    # print("xs[:, 1]:", xs[:, 1])
    # print("xs[:, 2]:", xs[:, 2])
    # print("xs[:, 3]:", xs[:, 3])
    # print("xs[:, 4]:", xs[:, 4])
    # print("xs[:, 4]:", xs[:, 4].shape)  # xs[:, 4]: (5,)
    xs = te.forward(xs)
    # print("xs:", xs)
    # print("xs:", xs.shape)  # xs: (5, 5, 5)

    # embed_W: [[  0   1   2   3   4]
    #  [  5   6   7   8   9]
    #  [ 10  11  12  13  14]
    #  [ 15  16  17  18  19]
    #  [ 20  21  22  23  24]
    #  [ 25  26  27  28  29]
    #  [ 30  31  32  33  34]
    #  [ 35  36  37  38  39]
    #  [ 40  41  42  43  44]
    #  [ 45  46  47  48  49]
    #  [ 50  51  52  53  54]
    #  [ 55  56  57  58  59]
    #  [ 60  61  62  63  64]
    #  [ 65  66  67  68  69]
    #  [ 70  71  72  73  74]
    #  [ 75  76  77  78  79]
    #  [ 80  81  82  83  84]
    #  [ 85  86  87  88  89]
    #  [ 90  91  92  93  94]
    #  [ 95  96  97  98  99]
    #  [100 101 102 103 104]
    #  [105 106 107 108 109]
    #  [110 111 112 113 114]
    #  [115 116 117 118 119]
    #  [120 121 122 123 124]]
    # xs: [[ 0  1  2  3  4]
    #  [ 5  6  7  8  9]
    #  [10 11 12 13 14]
    #  [15 16 17 18 19]
    #  [20 21 22 23 24]]
    # xs[:, 0]: [ 0  5 10 15 20]
    # xs[:, 1]: [ 1  6 11 16 21]
    # xs[:, 2]: [ 2  7 12 17 22]
    # xs[:, 3]: [ 3  8 13 18 23]
    # xs[:, 4]: [ 4  9 14 19 24]
    # xs[:, 4]: (5,)
    # xs: [[[  0.   1.   2.   3.   4.]
    #   [  5.   6.   7.   8.   9.]
    #   [ 10.  11.  12.  13.  14.]
    #   [ 15.  16.  17.  18.  19.]
    #   [ 20.  21.  22.  23.  24.]]

    #  [[ 25.  26.  27.  28.  29.]
    #   [ 30.  31.  32.  33.  34.]
    #   [ 35.  36.  37.  38.  39.]
    #   [ 40.  41.  42.  43.  44.]
    #   [ 45.  46.  47.  48.  49.]]

    #  [[ 50.  51.  52.  53.  54.]
    #   [ 55.  56.  57.  58.  59.]
    #   [ 60.  61.  62.  63.  64.]
    #   [ 65.  66.  67.  68.  69.]
    #   [ 70.  71.  72.  73.  74.]]

    #  [[ 75.  76.  77.  78.  79.]
    #   [ 80.  81.  82.  83.  84.]
    #   [ 85.  86.  87.  88.  89.]
    #   [ 90.  91.  92.  93.  94.]
    #   [ 95.  96.  97.  98.  99.]]

    #  [[100. 101. 102. 103. 104.]
    #   [105. 106. 107. 108. 109.]
    #   [110. 111. 112. 113. 114.]
    #   [115. 116. 117. 118. 119.]
    #   [120. 121. 122. 123. 124.]]]
    # xs: (5, 5, 5)

    print("******" * 20)

    ##########################################################################

    rnn_Wx = np.arange(25).reshape(5, 5)
    # print("rnn_Wx:", rnn_Wx)

    rnn_Wh = np.arange(25).reshape(5, 5)
    # print("rnn_Wh:", rnn_Wh)

    rnn_b = np.zeros(5)
    # print("rnn_b:", rnn_b)

    tn = TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True)
    xs = np.arange(125).reshape(5, 5, 5)
    # print("xs:", xs)
    # print("xs[:, 0, :]", xs[:, 0, :])
    # print("xs[:, 1, :]", xs[:, 1, :])
    # print("xs[:, 2, :]", xs[:, 2, :])
    # print("xs[:, 3, :]", xs[:, 3, :])
    # print("xs[:, 4, :]", xs[:, 4, :])
    # print("xs[:, 4, :]", xs[:, 4, :].shape)  # (5, 5)
    xs = tn.forward(xs)
    # print("xs:", xs)
    # print("xs:", xs.shape)

    # rnn_Wx: [[ 0  1  2  3  4]
    #  [ 5  6  7  8  9]
    #  [10 11 12 13 14]
    #  [15 16 17 18 19]
    #  [20 21 22 23 24]]
    # rnn_Wh: [[ 0  1  2  3  4]
    #  [ 5  6  7  8  9]
    #  [10 11 12 13 14]
    #  [15 16 17 18 19]
    #  [20 21 22 23 24]]
    # rnn_b: [0. 0. 0. 0. 0.]
    # xs: [[[  0   1   2   3   4]
    #   [  5   6   7   8   9]
    #   [ 10  11  12  13  14]
    #   [ 15  16  17  18  19]
    #   [ 20  21  22  23  24]]

    #  [[ 25  26  27  28  29]
    #   [ 30  31  32  33  34]
    #   [ 35  36  37  38  39]
    #   [ 40  41  42  43  44]
    #   [ 45  46  47  48  49]]

    #  [[ 50  51  52  53  54]
    #   [ 55  56  57  58  59]
    #   [ 60  61  62  63  64]
    #   [ 65  66  67  68  69]
    #   [ 70  71  72  73  74]]

    #  [[ 75  76  77  78  79]
    #   [ 80  81  82  83  84]
    #   [ 85  86  87  88  89]
    #   [ 90  91  92  93  94]
    #   [ 95  96  97  98  99]]

    #  [[100 101 102 103 104]
    #   [105 106 107 108 109]
    #   [110 111 112 113 114]
    #   [115 116 117 118 119]
    #   [120 121 122 123 124]]]
    # xs[:, 0, :] [[  0   1   2   3   4]
    #  [ 25  26  27  28  29]
    #  [ 50  51  52  53  54]
    #  [ 75  76  77  78  79]
    #  [100 101 102 103 104]]
    # xs[:, 1, :] [[  5   6   7   8   9]
    #  [ 30  31  32  33  34]
    #  [ 55  56  57  58  59]
    #  [ 80  81  82  83  84]
    #  [105 106 107 108 109]]
    # xs[:, 2, :] [[ 10  11  12  13  14]
    #  [ 35  36  37  38  39]
    #  [ 60  61  62  63  64]
    #  [ 85  86  87  88  89]
    #  [110 111 112 113 114]]
    # xs[:, 3, :] [[ 15  16  17  18  19]
    #  [ 40  41  42  43  44]
    #  [ 65  66  67  68  69]
    #  [ 90  91  92  93  94]
    #  [115 116 117 118 119]]
    # xs[:, 4, :] [[ 20  21  22  23  24]
    #  [ 45  46  47  48  49]
    #  [ 70  71  72  73  74]
    #  [ 95  96  97  98  99]
    #  [120 121 122 123 124]]
    # xs[:, 4, :] (5, 5)

    print("******" * 20)

    ##########################################################################
    affine_W = np.arange(100200).reshape(100, 1002)
    # print("affine_W:", affine_W.shape)  # affine_W: (100, 1002)

    affine_b = np.zeros(1002)
    # print("affine_b:", affine_b.shape)  # affine_b: (1002,)

    ta = TimeAffine(affine_W, affine_b)

    xs = np.arange(50000).reshape(10, 50, 100)
    # print("xs:", xs.shape)  # xs: (10, 50, 100)

    xs = ta.forward(xs)
    # print("xs:", xs.shape)  # xs: (10, 50, 1002)

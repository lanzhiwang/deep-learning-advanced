# coding: utf-8

import numpy as np


class RNN:

    def __init__(self, Wx, Wh, b):
        self.params = [Wx, Wh, b]
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)]
        self.cache = None

    def forward(self, x, h_prev):
        # h_prev: (3, 7)
        # Wh: (7, 7)
        # (3, 7)

        # x: (3, 2)
        # Wx: (2, 7)
        # (3, 7)

        # b (7, )

        Wx, Wh, b = self.params
        t = np.dot(h_prev, Wh) + np.dot(x, Wx) + b
        h_next = np.tanh(t)

        self.cache = (x, h_prev, h_next)
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
        Wx, Wh, b = self.params
        N, T, D = xs.shape
        D, H = Wx.shape
        print("N, T, D:", N, T, D)
        print("D, H:", D, H)
        # N, T, D: 3 2 2
        # D, H: 2 7

        self.layers = []
        hs = np.arange(42).reshape(N, T, H)
        # hs = np.empty((N, T, H), dtype='f')
        print("hs:", hs)
        print("hs:", hs.shape)
        # hs:
        # [
        #     [
        #         [ 0  1  2  3  4  5  6]
        #         [ 7  8  9 10 11 12 13]
        #     ]
        #     [
        #         [14 15 16 17 18 19 20]
        #         [21 22 23 24 25 26 27]
        #     ]
        #     [
        #         [28 29 30 31 32 33 34]
        #         [35 36 37 38 39 40 41]
        #     ]
        # ]
        # hs: (3, 2, 7)

        print(not self.stateful or self.h is None)
        if not self.stateful or self.h is None:
            self.h = np.zeros((N, H), dtype='f')
            print("self.h:", self.h.shape)
            # self.h: (3, 7)

        for t in range(T):
            layer = RNN(*self.params)
            self.h = layer.forward(xs[:, t, :], self.h)
            hs[:, t, :] = self.h
            print("hs:", hs)
            self.layers.append(layer)
            # hs:
            # [
            #     [
            #         [ 0  0  0  0  0  0  0]
            #         [ 7  8  9 10 11 12 13]
            #     ]
            #     [
            #         [ 0  0  0  0  0  0  0]
            #         [21 22 23 24 25 26 27]
            #     ]
            #     [
            #         [ 0  0  0  0  0  0  0]
            #         [35 36 37 38 39 40 41]
            #     ]
            # ]
            # hs:
            # [
            #     [
            #         [0 0 0 0 0 0 0]
            #         [0 0 0 0 0 0 0]
            #     ]
            #     [
            #         [0 0 0 0 0 0 0]
            #         [0 0 0 0 0 0 0]
            #     ]
            #     [
            #         [0 0 0 0 0 0 0]
            #         [0 0 0 0 0 0 0]
            #     ]
            # ]
        print("hs:", hs)
        # hs:
        # [
        #     [
        #         [0 0 0 0 0 0 0]
        #         [0 0 0 0 0 0 0]
        #     ]
        #     [
        #         [0 0 0 0 0 0 0]
        #         [0 0 0 0 0 0 0]
        #     ]
        #     [
        #         [0 0 0 0 0 0 0]
        #         [0 0 0 0 0 0 0]
        #     ]
        # ]
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


if __name__ == "__main__":
    rn = np.random.randn
    D, H = 2, 7

    rnn_Wx = (rn(D, H) / np.sqrt(D)).astype('f')
    rnn_Wh = (rn(H, H) / np.sqrt(H)).astype('f')
    rnn_b = np.zeros(H).astype('f')
    print("rnn_Wx:", rnn_Wx.shape)
    print("rnn_Wh:", rnn_Wh.shape)
    print("rnn_b:", rnn_b.shape)
    # rnn_Wx: (2, 7)
    # rnn_Wh: (7, 7)
    # rnn_b: (7,)

    network = TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True)
    xs = np.arange(12).reshape(3, 2, 2)
    print("xs:", xs)
    print("xs:", xs.shape)
    print("xs:", xs[:, 0, :])
    print("xs:", xs[:, 1, :])
    print("xs:", xs[:, 1, :].shape)
    # xs:
    # [
    #     [
    #         [ 0  1]
    #         [ 2  3]
    #     ]
    #     [
    #         [ 4  5]
    #         [ 6  7]
    #     ]
    #     [
    #         [ 8  9]
    #         [10 11]
    #     ]
    # ]
    # xs: (3, 2, 2)
    # xs:
    # [
    #     [0 1]
    #     [4 5]
    #     [8 9]
    # ]
    # xs:
    # [
    #     [ 2  3]
    #     [ 6  7]
    #     [10 11]
    # ]
    # xs: (3, 2)

    network.forward(xs)

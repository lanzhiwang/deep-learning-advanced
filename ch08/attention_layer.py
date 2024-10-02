# coding: utf-8
import sys

sys.path.append('..')
from common.np import *  # import numpy as np
from common.layers import Softmax


class WeightSum:
    """
    >>> import numpy as np
    >>> T, H = 5, 4
    >>> hs = np.ones((T, H))
    >>> hs
    array([[1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.]])
    >>> a = np.array([1, 2, 3, 4, 5])
    >>> ar = a.reshape(5, 1).repeat(4, axis=1)
    >>> ar
    array([[1, 1, 1, 1],
           [2, 2, 2, 2],
           [3, 3, 3, 3],
           [4, 4, 4, 4],
           [5, 5, 5, 5]])
    >>> ar.shape
    (5, 4)
    >>> t = hs * ar
    >>> t
    array([[1., 1., 1., 1.],
           [2., 2., 2., 2.],
           [3., 3., 3., 3.],
           [4., 4., 4., 4.],
           [5., 5., 5., 5.]])
    >>> t.shape
    (5, 4)
    >>> c = np.sum(t, axis=0)
    >>> c
    array([15., 15., 15., 15.])
    >>> c.shape
    (4,)
    >>>##################################################################
    >>> import numpy as np
    >>> T, H = 5, 4
    >>> hs = np.random.randn(T, H)
    >>> hs
    array([[-0.76725388,  1.28622985, -0.35009828, -0.24396786],
           [-0.11967557, -0.8606123 ,  0.30711723, -1.39132448],
           [ 1.57865771,  0.90535792, -0.04800526, -0.24125219],
           [-0.0228418 ,  0.41805872, -1.30673039, -0.54724207],
           [-0.061642  ,  0.98202804, -1.24393938, -0.32951619]])
    >>> a = np.array([0.8, 0.1, 0.03, 0.05, 0.02])
    >>> ar = a.reshape(5, 1).repeat(4, axis=1)
    >>> ar
    array([[0.8 , 0.8 , 0.8 , 0.8 ],
           [0.1 , 0.1 , 0.1 , 0.1 ],
           [0.03, 0.03, 0.03, 0.03],
           [0.05, 0.05, 0.05, 0.05],
           [0.02, 0.02, 0.02, 0.02]])
    >>> ar.shape
    (5, 4)
    >>> t = hs * ar
    >>> t
    array([[-0.6138031 ,  1.02898388, -0.28007863, -0.19517429],
           [-0.01196756, -0.08606123,  0.03071172, -0.13913245],
           [ 0.04735973,  0.02716074, -0.00144016, -0.00723757],
           [-0.00114209,  0.02090294, -0.06533652, -0.0273621 ],
           [-0.00123284,  0.01964056, -0.02487879, -0.00659032]])
    >>> t.shape
    (5, 4)
    >>> c = np.sum(t, axis=0)
    >>> c
    array([-0.58078586,  1.01062689, -0.34102237, -0.37549673])
    >>> c.shape
    (4,)
    >>>##################################################################
    >>> import numpy as np
    >>> N, T, H = 10, 5, 4
    >>> hs = np.random.randn(N, T, H)
    >>> hs.shape
    (10, 5, 4)
    >>> hs[0]
    array([[ 1.32875079,  0.61593305,  0.6739201 , -0.06463189],
           [ 0.15794099, -1.01586992,  0.28403224,  3.03221237],
           [ 1.14789601, -0.96666483, -0.16490949, -1.29039829],
           [ 1.83119305, -0.54142947,  0.65980797, -0.40728028],
           [ 2.52404451,  1.69666976, -1.07702416, -1.16595838]])
    >>> a = np.random.randn(N, T)
    >>> a
    array([[ 1.06407464,  0.02955272,  0.09259367, -1.81455133, -0.21257824],
           [ 0.01638732,  0.8052832 , -1.14217262,  0.5900963 ,  0.22390845],
           [ 0.19854423, -1.0459053 , -0.17480309,  0.05997129,  0.35625848],
           [ 0.41818267, -0.2280246 ,  0.70434895, -0.36924854, -0.24814507],
           [-0.23498638, -0.85115978,  0.6945216 , -0.90105656,  2.0619564 ],
           [-0.77652681, -0.0872629 , -1.32007573,  1.06827626,  1.71018737],
           [ 0.5075379 , -0.16284246, -0.22078514,  1.10111539,  0.61555054],
           [ 0.01843952,  0.38471728,  1.95159142,  1.00349189,  2.00729172],
           [-1.47097011, -1.04419518, -1.44458764, -0.089885  ,  2.29053722],
           [ 1.50103797,  0.35868943,  1.05105602, -0.66209585,  0.0499561 ]])
    >>> ar = a.reshape(N, T, 1).repeat(H, axis=2)
    >>> ar.shape()
    Traceback (most recent call last):
      File "<stdin>", line 1, in <module>
    TypeError: 'tuple' object is not callable
    >>> ar.shape
    (10, 5, 4)
    >>> a.reshape(N, T, 1)
    array([[[ 1.06407464],
            [ 0.02955272],
            [ 0.09259367],
            [-1.81455133],
            [-0.21257824]],

           [[ 0.01638732],
            [ 0.8052832 ],
            [-1.14217262],
            [ 0.5900963 ],
            [ 0.22390845]],

           [[ 0.19854423],
            [-1.0459053 ],
            [-0.17480309],
            [ 0.05997129],
            [ 0.35625848]],

           [[ 0.41818267],
            [-0.2280246 ],
            [ 0.70434895],
            [-0.36924854],
            [-0.24814507]],

           [[-0.23498638],
            [-0.85115978],
            [ 0.6945216 ],
            [-0.90105656],
            [ 2.0619564 ]],

           [[-0.77652681],
            [-0.0872629 ],
            [-1.32007573],
            [ 1.06827626],
            [ 1.71018737]],

           [[ 0.5075379 ],
            [-0.16284246],
            [-0.22078514],
            [ 1.10111539],
            [ 0.61555054]],

           [[ 0.01843952],
            [ 0.38471728],
            [ 1.95159142],
            [ 1.00349189],
            [ 2.00729172]],

           [[-1.47097011],
            [-1.04419518],
            [-1.44458764],
            [-0.089885  ],
            [ 2.29053722]],

           [[ 1.50103797],
            [ 0.35868943],
            [ 1.05105602],
            [-0.66209585],
            [ 0.0499561 ]]])
    >>>
    >>>
    >>> t = hs * ar
    >>> t.shape
    (10, 5, 4)
    >>> c = np.sum(t, axis=1)
    >>> c.shape
    (10, 4)
    >>>
    """

    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None

    def forward(self, hs, a):
        """
        N: 批次大小
        T:
        H:
        """
        N, T, H = hs.shape

        ar = a.reshape(N, T, 1).repeat(H, axis=2)
        t = hs * ar
        c = np.sum(t, axis=1)

        self.cache = (hs, ar)
        return c

    def backward(self, dc):
        hs, ar = self.cache
        N, T, H = hs.shape
        dt = dc.reshape(N, 1, H).repeat(T, axis=1)
        dar = dt * hs
        dhs = dt * ar
        da = np.sum(dar, axis=2)

        return dhs, da


class AttentionWeight:
    """
    >>> import sys
    >>> sys.path.append('..')
    >>> from common.layers import Softmax
    >>> import numpy as np
    >>>
    >>> T, H = 5, 4
    >>> hs = np.ones((T, H))
    >>> hs
    array([[1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.],
           [1., 1., 1., 1.]])
    >>>
    >>> h = np.array([1, 2, 3, 4])
    >>> hr = h.reshape(1, H).repeat(T, axis=0)
    >>> hr
    array([[1, 2, 3, 4],
           [1, 2, 3, 4],
           [1, 2, 3, 4],
           [1, 2, 3, 4],
           [1, 2, 3, 4]])
    >>>
    >>> t = hs * hr
    >>> t
    array([[1., 2., 3., 4.],
           [1., 2., 3., 4.],
           [1., 2., 3., 4.],
           [1., 2., 3., 4.],
           [1., 2., 3., 4.]])
    >>>
    >>> s = np.sum(t, axis=1)
    >>> s
    array([10., 10., 10., 10., 10.])
    >>>
    >>> softmax = Softmax()
    >>> a = softmax.forward(s)
    >>> a
    array([0.2, 0.2, 0.2, 0.2, 0.2])
    >>>
    """

    def __init__(self):
        self.params, self.grads = [], []
        self.softmax = Softmax()
        self.cache = None

    def forward(self, hs, h):
        N, T, H = hs.shape

        hr = h.reshape(N, 1, H).repeat(T, axis=1)
        t = hs * hr
        s = np.sum(t, axis=2)
        a = self.softmax.forward(s)

        self.cache = (hs, hr)
        return a

    def backward(self, da):
        hs, hr = self.cache
        N, T, H = hs.shape

        ds = self.softmax.backward(da)
        dt = ds.reshape(N, T, 1).repeat(H, axis=2)
        dhs = dt * hr
        dhr = dt * hs
        dh = np.sum(dhr, axis=1)

        return dhs, dh


class Attention:

    def __init__(self):
        self.params, self.grads = [], []
        self.attention_weight_layer = AttentionWeight()
        self.weight_sum_layer = WeightSum()
        self.attention_weight = None

    def forward(self, hs, h):
        a = self.attention_weight_layer.forward(hs, h)
        out = self.weight_sum_layer.forward(hs, a)
        self.attention_weight = a
        return out

    def backward(self, dout):
        dhs0, da = self.weight_sum_layer.backward(dout)
        dhs1, dh = self.attention_weight_layer.backward(da)
        dhs = dhs0 + dhs1
        return dhs, dh


class TimeAttention:

    def __init__(self):
        self.params, self.grads = [], []
        self.layers = None
        self.attention_weights = None

    def forward(self, hs_enc, hs_dec):
        N, T, H = hs_dec.shape
        out = np.empty_like(hs_dec)
        self.layers = []
        self.attention_weights = []

        for t in range(T):
            layer = Attention()
            out[:, t, :] = layer.forward(hs_enc, hs_dec[:, t, :])
            self.layers.append(layer)
            self.attention_weights.append(layer.attention_weight)

        return out

    def backward(self, dout):
        N, T, H = dout.shape
        dhs_enc = 0
        dhs_dec = np.empty_like(dout)

        for t in range(T):
            layer = self.layers[t]
            dhs, dh = layer.backward(dout[:, t, :])
            dhs_enc += dhs
            dhs_dec[:, t, :] = dh

        return dhs_enc, dhs_dec

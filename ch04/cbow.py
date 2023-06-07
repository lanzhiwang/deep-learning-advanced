# coding: utf-8
import sys

sys.path.append('..')
from common.np import *  # import numpy as np
from common.layers import Embedding
from ch04.negative_sampling_layer import NegativeSamplingLoss


class CBOW:

    def __init__(self, vocab_size, hidden_size, window_size, corpus):
        V, H = vocab_size, hidden_size
        # print("V, H:", V, H)
        # V, H: 7 5

        # 初始化权重
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(V, H).astype('f')
        # print("W_in:", W_in.shape)
        # W_in: (7, 5)
        # print("W_out:", W_out.shape)
        # W_out: (7, 5)

        # 生成层
        self.in_layers = []
        for i in range(2 * window_size):
            layer = Embedding(W_in)  # 使用Embedding层
            self.in_layers.append(layer)
        # print("self.in_layers:", self.in_layers)
        # self.in_layers:
        # [
        #     <common.layers.Embedding object at 0x7fd01e232fe0>,
        #     <common.layers.Embedding object at 0x7fd01e233fa0>
        # ]

        self.ns_loss = NegativeSamplingLoss(W_out,
                                            corpus,
                                            power=0.75,
                                            sample_size=5)
        # print("self.ns_loss:", self.ns_loss)
        # self.ns_loss: <ch04.negative_sampling_layer.NegativeSamplingLoss object at 0x7fd01e233fd0>

        # 将所有的权重和梯度整理到列表中
        layers = self.in_layers + [self.ns_loss]
        # print("layers:", layers)
        # layers:
        # [
        #     <common.layers.Embedding object at 0x7fbad474efe0>,
        #     <common.layers.Embedding object at 0x7fbad474ffa0>,
        #     <ch04.negative_sampling_layer.NegativeSamplingLoss object at 0x7fbad474ffd0>
        # ]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        # print("self.params:", len(self.params))
        # print("self.grads:", len(self.grads))
        # self.params: 8
        # self.grads: 8

        # 将单词的分布式表示设置为成员变量
        self.word_vecs = W_in

    def forward(self, contexts, target):
        # print("forward contexts:", contexts)
        # print("forward contexts:", contexts.shape)
        # forward contexts:
        #     [
        #         [4 5]
        #         [1 3]
        #         [1 6]
        #     ]
        # forward contexts: (3, 2)

        # print("forward target:", target)
        # print("forward target:", target.shape)
        # forward target: [1 2 5]
        # forward target: (3,)

        # print("forward contexts[:, 0]:", contexts[:, 0])
        # print("forward contexts[:, 1]:", contexts[:, 1])
        # forward contexts[:, 0]: [4 1 1]
        # forward contexts[:, 1]: [5 3 6]

        h = 0
        for i, layer in enumerate(self.in_layers):
            # print("i, layer:", i, layer)
            # i, layer: 0 <common.layers.Embedding object at 0x7f4ec60cba90>
            # i, layer: 1 <common.layers.Embedding object at 0x7f4e85550ac0>
            h += layer.forward(contexts[:, i])
        # print("h:", h)
        # h:
        # [
        #     [ 0.00532575 -0.0199242  -0.0181355  -0.01742779 -0.00909117]
        #     [-0.00465839 -0.00551104  0.01468217  0.01932374  0.00520242]
        #     [-0.01346372 -0.02272223 -0.04297481 -0.040435   -0.00042664]
        # ]
        '''
        >>> import numpy as np
        >>> out = np.arange(15).reshape((3, 5))
        >>> out
        array([[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14]])
        >>> h = 0
        >>> h
        0
        >>> h += out
        >>> h
        array([[ 0,  1,  2,  3,  4],
            [ 5,  6,  7,  8,  9],
            [10, 11, 12, 13, 14]])
        >>> h += out
        >>> h
        array([[ 0,  2,  4,  6,  8],
            [10, 12, 14, 16, 18],
            [20, 22, 24, 26, 28]])
        >>>
        '''

        h *= 1 / len(self.in_layers)
        loss = self.ns_loss.forward(h, target)
        return loss

    def backward(self, dout=1):
        dout = self.ns_loss.backward(dout)
        dout *= 1 / len(self.in_layers)
        for layer in self.in_layers:
            layer.backward(dout)
        return None

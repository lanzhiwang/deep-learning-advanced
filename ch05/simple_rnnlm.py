# coding: utf-8
import sys

sys.path.append('..')
import numpy as np
from common.time_layers import *


class SimpleRnnlm:

    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        # print("V, D, H:", V, D, H)
        # V, D, H: 7 3 7
        rn = np.random.randn

        # 初始化权重
        embed_W = (rn(V, D) / 100).astype('f')
        rnn_Wx = (rn(D, H) / np.sqrt(D)).astype('f')
        rnn_Wh = (rn(H, H) / np.sqrt(H)).astype('f')
        rnn_b = np.zeros(H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')
        # print("embed_W:", embed_W.shape)
        # print("rnn_Wx:", rnn_Wx.shape)
        # print("rnn_Wh:", rnn_Wh.shape)
        # print("rnn_b:", rnn_b.shape)
        # print("affine_W:", affine_W.shape)
        # print("affine_b:", affine_b.shape)
        # embed_W: (7, 3)
        # rnn_Wx: (3, 7)
        # rnn_Wh: (7, 7)
        # rnn_b: (7,)
        # affine_W: (7, 7)
        # affine_b: (7,)

        # 生成层
        self.layers = [
            TimeEmbedding(embed_W),
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[1]
        # print("self.layers:", self.layers)
        # print("self.loss_layer:", self.loss_layer)
        # print("self.rnn_layer:", self.rnn_layer)
        # self.layers:
        # [
        #     <common.time_layers.TimeEmbedding object at 0x7f6f75907f70>,
        #     <common.time_layers.TimeRNN object at 0x7f6f34e9f7f0>,
        #     <common.time_layers.TimeAffine object at 0x7f6f34e9e8c0>
        # ]
        # self.loss_layer: <common.time_layers.TimeSoftmaxWithLoss object at 0x7f6f34e9e830>
        # self.rnn_layer: <common.time_layers.TimeRNN object at 0x7f6f34e9f7f0>

        # 将所有的权重和梯度整理到列表中
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, ts):
        for layer in self.layers:
            xs = layer.forward(xs)
        loss = self.loss_layer.forward(xs, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.rnn_layer.reset_state()

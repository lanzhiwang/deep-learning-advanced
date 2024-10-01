# coding: utf-8
import sys

sys.path.append('..')
import numpy as np
from common.time_layers import *


class SimpleRnnlm:

    def __init__(self, vocab_size, wordvec_size, hidden_size):
        """
        vocab_size = int(max(corpus) + 1)
        wordvec_size = 100
        hidden_size = 100
        model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
        """
        V, D, H = vocab_size, wordvec_size, hidden_size
        # print(V, D, H)  # 1002 100 100

        rn = np.random.randn

        ############### TimeEmbedding 层 ###############
        # 初始化权重
        # 输入 # print("xs:", xs.shape)  # (batch_size, time_size) (10, 50)
        # 输入代表的含义是一个批次包含 10 条数据，每条数据由 50 个词语组成
        embed_W = (rn(V, D) / 100).astype('f')
        # print("embed_W:", embed_W.shape)  # embed_W: (1002, 100)
        # 输出 (10, 50, 100)
        # 输出代表的含义是一个批次包含 10 条数据，每条数据由 50 个词语组成，每个词语由 100 个元素的向量表示
        ##############################

        ############### TimeRNN 层 ###############
        # 输入 (10, 50, 100)
        # 输入代表的含义是一个批次包含 10 条数据，每条数据由 50 个词语组成，每个词语由 100 个元素的向量表示

        # (10, 50, 100) -> (10, 100)
        # (10, 100) * (100, 100) = (10, 100)
        rnn_Wx = (rn(D, H) / np.sqrt(D)).astype('f')
        # print("rnn_Wx:", rnn_Wx.shape)  # rnn_Wx: (100, 100)

        # (10, 100) * (100, 100) = (10, 100)
        rnn_Wh = (rn(H, H) / np.sqrt(H)).astype('f')
        # print("rnn_Wh:", rnn_Wh.shape)  # rnn_Wh: (100, 100)

        rnn_b = np.zeros(H).astype('f')
        # print("rnn_b:", rnn_b.shape)  # rnn_b: (100,)

        # 输出
        # (10, 100) -> (10, 50, 100)
        ##############################

        ############### TimeAffine 层 ###############
        # 输入 (10, 50, 100)
        # 输入代表的含义是一个批次包含 10 条数据，每条数据由 50 个词语组成，每个词语由 100 个元素的向量表示

        # (10, 50, 100) -> (10, 100)
        # (10, 100) * (100, 1002) = (10, 1002)
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        # print("affine_W:", affine_W.shape)  # affine_W: (100, 1002)

        affine_b = np.zeros(V).astype('f')
        # print("affine_b:", affine_b.shape)  # affine_b: (1002,)

        # 输出
        # (10, 1002) -> (10, 50, 1002)
        ##############################

        # 生成层
        self.layers = [
            TimeEmbedding(embed_W),
            TimeRNN(rnn_Wx, rnn_Wh, rnn_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.rnn_layer = self.layers[1]

        # 将所有的权重和梯度整理到列表中
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, ts):
        # print("forward xs:", xs.shape)  # forward xs: (10, 50)
        for layer in self.layers:
            xs = layer.forward(xs)
            # print("forward xs:", xs.shape)
            # forward xs: (10, 50, 100)
            # forward xs: (10, 50, 100)
            # forward xs: (10, 50, 1002)

        loss = self.loss_layer.forward(xs, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.rnn_layer.reset_state()

# coding: utf-8
import sys

sys.path.append('..')  # 为了引入父目录的文件而进行的设定
import numpy as np
from common.layers import Affine, Sigmoid, SoftmaxWithLoss


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size
        # print("I, H, O:", I, H, O)
        # I, H, O: 2 10 3

        # 初始化权重和偏置
        W1 = 0.01 * np.random.randn(I, H)
        # print("W1:", W1.shape)
        # W1: (2, 10)
        b1 = np.zeros(H)
        # print("b1:", b1.shape)
        # b1: (10,)
        W2 = 0.01 * np.random.randn(H, O)
        # print("W2:", W2.shape)
        # W2: (10, 3)
        b2 = np.zeros(O)
        # print("b2:", b2.shape)
        # b2: (3,)

        # 生成层
        self.layers = [Affine(W1, b1), Sigmoid(), Affine(W2, b2)]
        # print("self.layers:", self.layers)
        # self.layers:
        # [
        #     <common.layers.Affine object at 0x7f075c1bee90>,
        #     <common.layers.Sigmoid object at 0x7f071ce33cd0>,
        #     <common.layers.Affine object at 0x7f071b63ba60>
        # ]

        self.loss_layer = SoftmaxWithLoss()
        # print("self.loss_layer:", self.loss_layer)
        # self.loss_layer: <common.layers.SoftmaxWithLoss object at 0x7f071b63ac20>

        # 将所有的权重和偏置整理到列表中
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
        # print("self.params:", self.params)
        # self.params:
        # [
        #     array(
        #         [
        #             [ 0.00090477,  0.00204578,  0.0032964 ,  0.00433086, -0.01705616, 0.00053312, -0.0071672 , -0.00901328, -0.01411135,  0.02651132],
        #             [ 0.00039253,  0.01452478, -0.00792331, -0.00376801,  0.00529584, -0.00047179, -0.00683927, -0.0112335 , -0.02220007, -0.00502841]
        #         ]
        #     ),
        #     array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        #     array(
        #         [
        #             [ 8.79442449e-03, -1.44664165e-02, -9.11353614e-05],
        #             [ 2.20911881e-02, -1.62870734e-02,  1.25057461e-02],
        #             [ 3.31731595e-03, -3.46797983e-03, -1.34491722e-02],
        #             [-1.91859493e-02,  3.33893824e-03,  1.80190708e-03],
        #             [ 5.73398781e-03, -6.08343752e-03, -9.15566459e-03],
        #             [-2.83794064e-03, -5.92508411e-03,  8.90279760e-04],
        #             [ 5.17973858e-03,  9.87278162e-03, -7.10820861e-03],
        #             [-4.94125234e-04, -1.56211517e-02, -2.43546927e-02],
        #             [-8.72618449e-03, -2.83630957e-04, -6.72287471e-03],
        #             [ 1.50462578e-02,  9.64682929e-04,  2.14176135e-02]
        #         ]
        #     ),
        #     array([0., 0., 0.])
        # ]

        # print("self.grads:", self.grads)
        # self.grads:
        # [
        #     array(
        #         [
        #             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        #             [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        #         ]
        #     ),
        #     array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]),
        #     array(
        #         [
        #             [0., 0., 0.],
        #             [0., 0., 0.],
        #             [0., 0., 0.],
        #             [0., 0., 0.],
        #             [0., 0., 0.],
        #             [0., 0., 0.],
        #             [0., 0., 0.],
        #             [0., 0., 0.],
        #             [0., 0., 0.],
        #             [0., 0., 0.]
        #         ]
        #     ),
        #     array([0., 0., 0.])
        # ]

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

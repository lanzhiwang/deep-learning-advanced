# coding: utf-8
import numpy as np


class Sigmoid:

    def __init__(self):
        self.params = []

    def forward(self, x):
        return 1 / (1 + np.exp(-x))


class Affine:

    def __init__(self, W, b):
        self.params = [W, b]

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        return out


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size):
        I, H, O = input_size, hidden_size, output_size
        # print("I, H, O:", I, H, O)
        # # I, H, O: 2 4 3

        # 初始化权重和偏置
        W1 = np.random.randn(I, H)
        b1 = np.random.randn(H)
        W2 = np.random.randn(H, O)
        b2 = np.random.randn(O)

        # print("W1:", W1.shape)
        # # W1: (2, 4)
        # print("b1:", b1.shape)
        # # b1: (4,)
        # print("W2:", W2.shape)
        # # W2: (4, 3)
        # print("b2:", b2.shape)
        # # b2: (3,)

        # 生成层
        self.layers = [Affine(W1, b1), Sigmoid(), Affine(W2, b2)]

        # 将所有的权重整理到列表中
        self.params = []
        for layer in self.layers:
            self.params += layer.params
        # print("self.params:", self.params)
        # self.params:
        # [
        #     array(
        #         [
        #             [ 0.95397918,  0.58446251, -0.13190497, -1.02840822],
        #             [-0.40795267, -1.01096328, -0.39041573,  0.48496326]
        #         ]
        #     ),
        #     array([0.20882839, 0.24513952, 0.54788197, 0.74970611]),
        #     array(
        #         [
        #             [ 1.13352883, -0.10349613,  2.079881  ],
        #             [-0.39274366, -2.07861635,  0.20024586],
        #             [-0.5633373 , -0.75790733,  2.0376581 ],
        #             [ 0.29037278, -0.78416905,  0.3345657 ]
        #         ]
        #     ),
        #     array([ 0.46459265, -0.3170786 ,  0.13490017])
        # ]

    def predict(self, x):
        for layer in self.layers:
            print("x:", x.shape)
            x = layer.forward(x)
            print("x:", x.shape)
        return x


x = np.random.randn(10, 2)
print("x:", x)

model = TwoLayerNet(2, 4, 3)
s = model.predict(x)
print(s)

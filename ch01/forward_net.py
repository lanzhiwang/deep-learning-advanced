# coding: utf-8
import numpy as np


class Sigmoid:

    def __init__(self):
        """
        Sigmoid()
        """
        self.params = []

    def forward(self, x):
        return 1 / (1 + np.exp(-x))


class Affine:

    def __init__(self, W, b):
        """
        Affine(W1, b1)
        Affine(W2, b2)
        """
        self.params = [W, b]

    def forward(self, x):
        W, b = self.params
        out = np.dot(x, W) + b
        return out


class TwoLayerNet:

    def __init__(self, input_size, hidden_size, output_size):
        """
        TwoLayerNet(2, 4, 3)
        """
        I, H, O = input_size, hidden_size, output_size

        # 初始化权重和偏置
        W1 = np.random.randn(I, H)  # (2, 4)
        b1 = np.random.randn(H)  # (4, )
        W2 = np.random.randn(H, O)  # (4, 3)
        b2 = np.random.randn(O)  # (3, )

        # 生成层
        self.layers = [Affine(W1, b1), Sigmoid(), Affine(W2, b2)]

        # 将所有的权重整理到列表中
        """
        >>> a = ['A', 'B']
        >>> a += ['C', 'D']
        >>> a
        ['A', 'B', 'C', 'D']
        """
        self.params = []
        for layer in self.layers:
            self.params += layer.params
        # print("self.params:", self.params)
        """
        self.params: [
            array([[-1.16533464, -0.87946329, -0.13285725, -0.24595533],
                   [ 0.4257846 ,  0.99898608,  1.04343144, -2.66701499]]),
            array([ 1.31901113, -1.42503588,  1.4279264 , -0.46614344]),
            array([[-0.42492321,  0.86849701, -0.6103826 ],
                   [-0.63357208, -1.75437395,  1.00636698],
                   [ 1.41769175,  1.05346407,  0.0961829 ],
                   [ 1.13157322,  0.10539358, -0.32881792]]),
            array([-0.32774122, -0.38686183, -1.25844291])
       ]
        """

    def predict(self, x):
        """
        x = np.random.randn(10, 2)

        (10, 2) * (2, 4) + (4, ) = (10, 4)
        (10 ,4) * (4, 3) + (3, ) = (10, 3)
        """
        for layer in self.layers:
            x = layer.forward(x)
        return x


if __name__ == "__main__":
    x = np.random.randn(10, 2)
    model = TwoLayerNet(2, 4, 3)
    s = model.predict(x)
    print(s)

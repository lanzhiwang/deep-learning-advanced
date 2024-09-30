# coding: utf-8
import numpy as np


def load_data(seed=1984):
    np.random.seed(seed)
    N = 100  # 各类的样本数
    DIM = 2  # 数据的元素个数
    CLS_NUM = 3  # 类别数

    x = np.zeros((N * CLS_NUM, DIM))
    t = np.zeros((N * CLS_NUM, CLS_NUM), dtype=np.int32)
    # print('x', x.shape)  # (300, 2)
    # print('t', t.shape)  # (300, 3)

    for j in range(CLS_NUM):  # 0 1 2
        for i in range(N):  # 0 1 2 ... 99
            rate = i / N
            radius = 1.0 * rate
            theta = j * 4.0 + 4.0 * rate + np.random.randn() * 0.2

            ix = N * j + i
            x[ix] = np.array([radius * np.sin(theta),
                              radius * np.cos(theta)]).flatten()
            t[ix, j] = 1

    return x, t


if __name__ == '__main__':
    x, t = load_data()
    print('x', x.shape)  # (300, 2)
    print('t', t.shape)  # (300, 3)

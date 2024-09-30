# coding: utf-8
import sys

sys.path.append('..')  # 为了引入父目录的文件而进行的设定
from dataset import spiral
import matplotlib.pyplot as plt
"""
x 是输入数据, t 是监督标签.
观察 x 和 t 的形状, 可知它们各自有 300 笔样本数据，其中 x 是二维数据, t 是三维数据.
另外, t 是 one-hot 向量, 对应的正确解标签的类标记为 1, 其余的标记为 0.
"""
x, t = spiral.load_data()
print('x', x.shape)  # (300, 2)
print('t', t.shape)  # (300, 3)
# print('x[0]', x[0])  # x[0] [-0.  0.]
# print('x[1]', x[1])  # x[1] [-0.00097699  0.00995216]
# print('t[0]', t[0])  # t[0] [1 0 0]
# print('t[1]', t[1])  # t[1] [1 0 0]

# 绘制数据点
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i * N:(i + 1) * N, 0],
                x[i * N:(i + 1) * N, 1],
                s=40,
                marker=markers[i])
plt.show()
plt.savefig('show_spiral_dataset.png')

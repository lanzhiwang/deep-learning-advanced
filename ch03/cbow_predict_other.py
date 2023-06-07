# coding: utf-8
import sys

sys.path.append('..')
import numpy as np
from common.layers import MatMul

# 样本的上下文数据
c = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0]])
print("c:", c)
print("c:", c.shape)
# c:
# [
#     [1 0 0 0 0 0 0]
#     [0 0 1 0 0 0 0]
# ]
# c: (2, 7)

# 初始化权重
# W_in = np.random.randn(7, 3)
# W_out = np.random.randn(3, 7)
W_in = np.arange(21).reshape((7, 3))
W_out = np.arange(21).reshape((3, 7))
print("W_in:", W_in)
print("W_out:", W_out)
# W_in:
# [
#     [ 0  1  2]
#     [ 3  4  5]
#     [ 6  7  8]
#     [ 9 10 11]
#     [12 13 14]
#     [15 16 17]
#     [18 19 20]
# ]
# W_out:
# [
#     [ 0  1  2  3  4  5  6]
#     [ 7  8  9 10 11 12 13]
#     [14 15 16 17 18 19 20]
# ]

# 生成层
in_layer = MatMul(W_in)
out_layer = MatMul(W_out)

# 正向传播
h = in_layer.forward(c)
print("h:", h)
print("h:", h.shape)
# h:
# [
#     [0 1 2]
#     [6 7 8]
# ]
# h: (2, 3)

h = np.sum(h, axis=0, keepdims=True)
print("h:", h)
print("h:", h.shape)
# h: [[ 6  8 10]]
# h: (1, 3)

h = 0.5 * h
print("h:", h)
print("h:", h.shape)
# h: [[3. 4. 5.]]
# h: (1, 3)

s = out_layer.forward(h)
print("s:", s.shape)
# s: (1, 7)

print(s)
# [[ 98. 110. 122. 134. 146. 158. 170.]]

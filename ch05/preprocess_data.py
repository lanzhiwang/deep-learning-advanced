# coding: utf-8
import sys

sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
from common.optimizer import SGD
from dataset import ptb
from simple_rnnlm import SimpleRnnlm

# 设定超参数
batch_size = 3
wordvec_size = 100
hidden_size = 100
time_size = 4  # Truncated BPTT的时间跨度大小
lr = 0.1
max_epoch = 1

# 读入训练数据（缩小了数据集）
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus_size = 51
corpus = corpus[:corpus_size]
corpus = np.arange(corpus_size)
print("corpus:", corpus)
vocab_size = int(max(corpus) + 1)

xs = corpus[:-1]  # 输入
print("xs:", xs)

ts = corpus[1:]  # 输出（监督标签）
data_size = len(xs)

# 学习用的参数
max_iters = data_size // (batch_size * time_size)  # 50 // 12
time_idx = 0
total_loss = 0
loss_count = 0
ppl_list = []

# 生成模型
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)

# 计算读入mini-batch的各笔样本数据的开始位置
jump = (corpus_size - 1) // batch_size  # (51 - 1) // 3 = 16
offsets = [i * jump for i in range(batch_size)]
# print("jump:", jump)
# print("offsets:", offsets)
# jump: 16
# offsets: [0, 16, 32]

for epoch in range(max_epoch):
    for iter in range(max_iters):
        print("iter:", iter)
        # 获取mini-batch
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')
        for t in range(time_size):
            for i, offset in enumerate(offsets):
                # print("i, t, time_idx:", i, t, time_idx)
                batch_x[i, t] = xs[(offset + time_idx) % data_size]
                batch_t[i, t] = ts[(offset + time_idx) % data_size]
            time_idx += 1

        print("batch_x:", batch_x)
        # print("batch_x:", batch_x.shape)

#         # 计算梯度，更新参数
#         loss = model.forward(batch_x, batch_t)
#         model.backward()
#         optimizer.update(model.params, model.grads)
#         total_loss += loss
#         loss_count += 1

#     # 各个epoch的困惑度评价
#     ppl = np.exp(total_loss / loss_count)
#     print('| epoch %d | perplexity %.2f' % (epoch + 1, ppl))
#     ppl_list.append(float(ppl))
#     total_loss, loss_count = 0, 0

# # 绘制图形
# x = np.arange(len(ppl_list))
# plt.plot(x, ppl_list, label='train')
# plt.xlabel('epochs')
# plt.ylabel('perplexity')
# plt.show()
'''
corpus: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
 48 49 50]

xs: [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47
 48 49]

50 个数, 分 3 行 4 列进行处理
也就是 50 个数, 每次处理 12 个, 一共需要处理 4 轮

offsets: [0, 16, 32]

[0  1  2  3]
[4  5  6  7]
[8  9 10 11]
[12 13 14 15]

[16 17 18 19]
[20 21 22 23]
[24 25 26 27]
[28 29 30 31]

[32 33 34 35]
[36 37 38 39]
[40 41 42 43]
[44 45 46 47]

[48 49]

iter: 0
batch_x:
[
    [ 0  1  2  3]
    [16 17 18 19]
    [32 33 34 35]
]

iter: 1
batch_x:
[
    [ 4  5  6  7]
    [20 21 22 23]
    [36 37 38 39]
]

iter: 2
batch_x:
[
    [ 8  9 10 11]
    [24 25 26 27]
    [40 41 42 43]
]

iter: 3
batch_x:
[
    [12 13 14 15]
    [28 29 30 31]
    [44 45 46 47]
]

'''

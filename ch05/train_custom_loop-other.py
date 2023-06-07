# coding: utf-8
import sys

sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
from common.optimizer import SGD
from dataset import ptb
from simple_rnnlm import SimpleRnnlm
from common.util import preprocess

# 设定超参数
batch_size = 2
wordvec_size = 3
hidden_size = 7
time_size = 2  # Truncated BPTT的时间跨度大小
lr = 0.1
max_epoch = 2

# 读入训练数据(缩小了数据集)
text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
# print("corpus:", corpus)
# print("corpus:", len(corpus))
# print("word_to_id:", word_to_id)
# print("word_to_id:", len(word_to_id))
# print("id_to_word:", id_to_word)
# print("id_to_word:", len(id_to_word))
# corpus: [0 1 2 3 4 1 5 6]
# corpus: 8
# word_to_id: {'you': 0, 'say': 1, 'goodbye': 2, 'and': 3, 'i': 4, 'hello': 5, '.': 6}
# word_to_id: 7
# id_to_word: {0: 'you', 1: 'say', 2: 'goodbye', 3: 'and', 4: 'i', 5: 'hello', 6: '.'}
# id_to_word: 7

corpus_size = 10
corpus = corpus[:corpus_size]
vocab_size = int(max(corpus) + 1)
# print("corpus:", corpus)
# print("corpus:", len(corpus))
# print("vocab_size:", vocab_size)
# corpus: [0 1 2 3 4 1 5 6]
# corpus: 8
# vocab_size: 7

xs = corpus[:-1]  # 输入
ts = corpus[1:]  # 输出(监督标签)
data_size = len(xs)
# print("xs:", xs)
# print("xs:", len(xs))
# print("ts:", ts)
# print("ts:", len(ts))
# print("data_size:", data_size)
# xs: [0 1 2 3 4 1 5]
# xs: 7
# ts: [1 2 3 4 1 5 6]
# ts: 7
# data_size: 7

# print('corpus size: %d, vocabulary size: %d' % (corpus_size, vocab_size))
# corpus size: 10, vocabulary size: 7

# 学习用的参数
max_iters = data_size // (batch_size * time_size)
# print("max_iters:", max_iters)
# max_iters: 1

time_idx = 0
total_loss = 0
loss_count = 0
ppl_list = []

# 生成模型
model = SimpleRnnlm(vocab_size, wordvec_size, hidden_size)
optimizer = SGD(lr)

# 计算读入mini-batch的各笔样本数据的开始位置
jump = (corpus_size - 1) // batch_size
# print("jump:", jump)
# jump: 4

offsets = [i * jump for i in range(batch_size)]
# print("offsets:", offsets)
# offsets: [0, 4]

for epoch in range(max_epoch):
    # print("epoch:", epoch)
    for iter in range(max_iters):
        # print("iter:", iter)
        # 获取mini-batch
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')
        # print("batch_x:", batch_x)
        # print("batch_x:", batch_x.shape)
        # print("batch_t:", batch_t)
        # print("batch_t:", batch_t.shape)
        for t in range(time_size):
            # print("t:", t)
            for i, offset in enumerate(offsets):
                # print("i, offset:", i, offset)
                # print("i, t:", i, t)
                # print("(offset + time_idx) % data_size:", (offset + time_idx) % data_size)
                # print("xs[(offset + time_idx) % data_size]:", xs[(offset + time_idx) % data_size])
                # print("ts[(offset + time_idx) % data_size]:", ts[(offset + time_idx) % data_size])
                batch_x[i, t] = xs[(offset + time_idx) % data_size]
                batch_t[i, t] = ts[(offset + time_idx) % data_size]
            time_idx += 1

        # print("batch_x:", batch_x)
        # print("batch_x:", batch_x.shape)
        # print("batch_t:", batch_t)
        # print("batch_t:", batch_t.shape)
        # # 计算梯度, 更新参数
        # loss = model.forward(batch_x, batch_t)
        # model.backward()
        # optimizer.update(model.params, model.grads)
        # total_loss += loss
        # loss_count += 1
'''
epoch: 0
    iter: 0
        batch_x: [
            [          0 -1073741824]
            [ -573210625       32767]
        ]
        batch_x: (2, 2)
        batch_t: [
            [ 1262746869 -2117160148]
            [-1979039848       32708]
        ]
        batch_t: (2, 2)
            t: 0
                i, offset: 0 0
                    i, t: 0 0
                    (offset + time_idx) % data_size: 0
                    xs[(offset + time_idx) % data_size]: 0
                    ts[(offset + time_idx) % data_size]: 1
                i, offset: 1 4
                    i, t: 1 0
                    (offset + time_idx) % data_size: 4
                    xs[(offset + time_idx) % data_size]: 4
                    ts[(offset + time_idx) % data_size]: 1
            t: 1
                i, offset: 0 0
                    i, t: 0 1
                    (offset + time_idx) % data_size: 1
                    xs[(offset + time_idx) % data_size]: 1
                    ts[(offset + time_idx) % data_size]: 2
                i, offset: 1 4
                    i, t: 1 1
                    (offset + time_idx) % data_size: 5
                    xs[(offset + time_idx) % data_size]: 1
                    ts[(offset + time_idx) % data_size]: 5
        batch_x: [
            [0 1]
            [4 1]
        ]
        batch_x: (2, 2)
        batch_t: [
            [1 2]
            [1 5]
        ]
        batch_t: (2, 2)
epoch: 1
    iter: 0
        batch_x: [
            [2022303497      22012]
            [         0          0]
        ]
        batch_x: (2, 2)
        batch_t: [
            [0 1]
            [4 1]
        ]
        batch_t: (2, 2)
            t: 0
                i, offset: 0 0
                    i, t: 0 0
                    (offset + time_idx) % data_size: 2
                    xs[(offset + time_idx) % data_size]: 2
                    ts[(offset + time_idx) % data_size]: 3
                i, offset: 1 4
                    i, t: 1 0
                    (offset + time_idx) % data_size: 6
                    xs[(offset + time_idx) % data_size]: 5
                    ts[(offset + time_idx) % data_size]: 6
            t: 1
                i, offset: 0 0
                    i, t: 0 1
                    (offset + time_idx) % data_size: 3
                    xs[(offset + time_idx) % data_size]: 3
                    ts[(offset + time_idx) % data_size]: 4
                i, offset: 1 4
                    i, t: 1 1
                    (offset + time_idx) % data_size: 0
                    xs[(offset + time_idx) % data_size]: 0
                    ts[(offset + time_idx) % data_size]: 1
        batch_x: [
            [2 3]
            [5 0]
        ]
        batch_x: (2, 2)
        batch_t: [
            [3 4]
            [6 1]
        ]
        batch_t: (2, 2)
'''

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

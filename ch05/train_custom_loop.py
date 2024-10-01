# coding: utf-8
import sys

sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
from common.optimizer import SGD
from dataset import ptb
from simple_rnnlm import SimpleRnnlm

# 设定超参数
batch_size = 10
wordvec_size = 100
hidden_size = 100
time_size = 50  # Truncated BPTT的时间跨度大小
lr = 0.1
max_epoch = 100

# 读入训练数据（缩小了数据集）
corpus, word_to_id, id_to_word = ptb.load_data('train')
corpus = [i + 1 for i in range(1002)]
# print("corpus:", corpus)
# corpus: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ... 997, 998, 999, 1000, 1001, 1002]

corpus_size = 1001
corpus = corpus[:corpus_size]
# print("corpus:", corpus)
# corpus: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ... 997, 998, 999, 1000, 1001]

vocab_size = int(max(corpus) + 1)
# print("vocab_size:", vocab_size)
# vocab_size: 1002

xs = corpus[:-1]  # 输入
# print("xs:", xs)
# xs: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, ... 996, 997, 998, 999, 1000]

ts = corpus[1:]  # 输出（监督标签）
# print("ts:", ts)
# ts: [2, 3, 4, 5, 6, 7, 8, 9, 10, ... 997, 998, 999, 1000, 1001]

data_size = len(xs)
# print("data_size:", data_size)
# data_size: 1000

print('corpus size: %d, vocabulary size: %d' % (corpus_size, vocab_size))

# 学习用的参数
max_iters = data_size // (batch_size * time_size)
# print("max_iters:", max_iters)
# max_iters: 1
# max_iters: 2

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
# jump: 100

offsets = [i * jump for i in range(batch_size)]
# print("offsets:", offsets)
# offsets: [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]

for epoch in range(max_epoch):
    for iter in range(max_iters):
        # print("iter:", iter)
        # 获取mini-batch
        batch_x = np.empty((batch_size, time_size), dtype='i')  # (10, 50)
        batch_t = np.empty((batch_size, time_size), dtype='i')
        for t in range(time_size):
            # print("    t:", t, "time_idx:", time_idx)
            for i, offset in enumerate(offsets):
                # print("        i:", i, "\toffset:", offset, "\ttime_idx:", time_idx, "\t", offset + time_idx, "\t", (offset + time_idx) % data_size)
                batch_x[i, t] = xs[(offset + time_idx) % data_size]
                batch_t[i, t] = ts[(offset + time_idx) % data_size]
            time_idx += 1

        # 计算梯度，更新参数
        loss = model.forward(batch_x, batch_t)
        model.backward()
        optimizer.update(model.params, model.grads)
        total_loss += loss
        loss_count += 1

    # 各个epoch的困惑度评价
    ppl = np.exp(total_loss / loss_count)
    print('| epoch %d | perplexity %.2f' % (epoch + 1, ppl))
    ppl_list.append(float(ppl))
    total_loss, loss_count = 0, 0

# 绘制图形
x = np.arange(len(ppl_list))
plt.plot(x, ppl_list, label='train')
plt.xlabel('epochs')
plt.ylabel('perplexity')
plt.show()

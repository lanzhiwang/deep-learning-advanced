# coding: utf-8
import sys

sys.path.append('..')  # 为了引入父目录的文件而进行的设定
import numpy as np
from common.optimizer import SGD
from dataset import spiral
import matplotlib.pyplot as plt
from two_layer_net import TwoLayerNet

# 设定超参数
max_epoch = 300
batch_size = 30
hidden_size = 10
learning_rate = 1.0

x, t = spiral.load_data()
model = TwoLayerNet(input_size=2, hidden_size=hidden_size, output_size=3)
optimizer = SGD(lr=learning_rate)

# 学习用的变量
data_size = len(x)  # 300
max_iters = data_size // batch_size  # 10
total_loss = 0
loss_count = 0
loss_list = []

for epoch in range(max_epoch):
    # 打乱数据
    # permutation(x)函数最终返回的都是乱序后的数组
    idx = np.random.permutation(data_size)
    x = x[idx]
    t = t[idx]

    for iters in range(max_iters):
        # print("iters:", iters)
        # print("iters * batch_size:", iters * batch_size)
        # print("(iters + 1) * batch_size:", (iters + 1) * batch_size)
        # iters: 1
        # iters * batch_size: 30
        # (iters + 1) * batch_size: 60
        # iters: 2
        # iters * batch_size: 60
        # (iters + 1) * batch_size: 90
        # iters: 3
        # iters * batch_size: 90
        # (iters + 1) * batch_size: 120
        # iters: 4
        # iters * batch_size: 120
        # (iters + 1) * batch_size: 150
        # iters: 5
        # iters * batch_size: 150
        # (iters + 1) * batch_size: 180
        # iters: 6
        # iters * batch_size: 180
        # (iters + 1) * batch_size: 210
        # iters: 7
        # iters * batch_size: 210
        # (iters + 1) * batch_size: 240
        # iters: 8
        # iters * batch_size: 240
        # (iters + 1) * batch_size: 270
        # iters: 9
        # iters * batch_size: 270
        # (iters + 1) * batch_size: 300
        batch_x = x[iters * batch_size:(iters + 1) * batch_size]
        batch_t = t[iters * batch_size:(iters + 1) * batch_size]

        # 计算梯度，更新参数
        loss = model.forward(batch_x, batch_t)
        model.backward()
        # 更新参数
        optimizer.update(model.params, model.grads)

        total_loss += loss
        loss_count += 1

        # 定期输出学习过程
        if (iters + 1) % 10 == 0:
            avg_loss = total_loss / loss_count
            print('| epoch %d |  iter %d / %d | loss %.2f' %
                  (epoch + 1, iters + 1, max_iters, avg_loss))
            loss_list.append(avg_loss)
            total_loss, loss_count = 0, 0

# 绘制学习结果
plt.plot(np.arange(len(loss_list)), loss_list, label='train')
plt.xlabel('iterations (x10)')
plt.ylabel('loss')
plt.show()

# 绘制决策边界
h = 0.001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]
score = model.predict(X)
predict_cls = np.argmax(score, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)
plt.axis('off')

# 绘制数据点
x, t = spiral.load_data()
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i * N:(i + 1) * N, 0],
                x[i * N:(i + 1) * N, 1],
                s=40,
                marker=markers[i])
plt.show()

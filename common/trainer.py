# coding: utf-8
import sys

sys.path.append('..')
import numpy
import time
import matplotlib.pyplot as plt
from common.np import *  # import numpy as np
from common.util import clip_grads


class Trainer:

    # trainer = Trainer(model, optimizer)
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    # trainer.fit(x, t, max_epoch, batch_size, eval_interval=10)
    def fit(self,
            x,
            t,
            max_epoch=10,
            batch_size=32,
            max_grad=None,
            eval_interval=20):
        data_size = len(x)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        # print("max_epoch:", max_epoch)
        # print("batch_size:", batch_size)
        # print("eval_interval:", eval_interval)
        # print("data_size:", data_size)
        # print("max_iters:", max_iters)
        # x (300, 2)
        # t (300, 3)
        # max_epoch: 300
        # batch_size: 30
        # eval_interval: 10
        # data_size: 300
        # max_iters: 10

        start_time = time.time()
        for epoch in range(max_epoch):
            # 打乱
            idx = numpy.random.permutation(numpy.arange(data_size))
            # print("idx:", idx)
            # print("idx:", idx.shape)
            # idx:
            # [275 219  85 280   0  49 136 108  87 288  21 252 139 121 223 161 251  69
            # 133 186 165 182  46 248  63 213 255   3  37  83  27  13 184 135 169 221
            # 250   4 293  81 225  89 245 122  31 218  70  45 167 102 134 144  40  68
            # 172  47 145 200  11  64 227  71 290 224 278  19 249  98  59  20 106 178
            # 126  36 176 193 189  30 210 214 260 111 242 132  12 268 270 207 299 110
            # 159 128  55 284 239  17 231  58 247 209 297 195 116 202 276 197 163 240
            # 257 230 164 124  67 298 191 107  75  78 156 265 119 241 259 115 170 103
            # 42  88 162 157  25 281 205  91  26  56 215 181  16  41  24 177  23  28
            # 267   8 285 123 118 190 168 143 130  52 266 264 283  18 206 296 154 196
            # 246 291 185 233 183 212 146  95 105 269 289 125 220 155 188  65 232 277
            # 80 120 152 203  79 101 235  53  32  73  48 294  66 113 229 192  77 173
            # 138  22 244   2  29 179  92   9  39 279 238 174   6 261  93 228 137 158
            # 253 160 141 236  62 198  72  14 149  35 286  57 292 131 199 153 109   5
            # 99  76 226  34  10 148 258 187 175 287 295  33  61  15 274  74 272 117
            # 208 129  84  96   1 194  82 142 273 217  90 147  60 150 254 263 151  50
            # 243 127 256 171 262  38  51 100  44 234  86 104 114 180 112  43  97 201
            # 222 216 271 237 204 282   7 140 166 211  54  94]
            # idx: (300,)

            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):
                # print(iters)
                # print(iters * batch_size, (iters + 1) * batch_size)
                # 1
                # 30 60
                # 2
                # 60 90
                # 3
                # 90 120
                # 4
                # 120 150
                # 5
                # 150 180
                # 6
                # 180 210
                # 7
                # 210 240
                # 8
                # 240 270
                # 9
                # 270 300
                batch_x = x[iters * batch_size:(iters + 1) * batch_size]
                batch_t = t[iters * batch_size:(iters + 1) * batch_size]

                # 计算梯度, 更新参数
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params,
                                                 model.grads)  # 将共享的权重整合为1个
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                # 评价
                if (eval_interval
                        is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    print(
                        '| epoch %d |  iter %d / %d | time %d[s] | loss %.2f' %
                        (self.current_epoch + 1, iters + 1, max_iters,
                         elapsed_time, avg_loss))
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = numpy.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
        plt.ylabel('loss')
        plt.show()


class RnnlmTrainer:

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.time_idx = None
        self.ppl_list = None
        self.eval_interval = None
        self.current_epoch = 0

    def get_batch(self, x, t, batch_size, time_size):
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')

        data_size = len(x)
        jump = data_size // batch_size
        offsets = [i * jump
                   for i in range(batch_size)]  # mini-batch的各笔样本数据的开始位置

        for time in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, time] = x[(offset + self.time_idx) % data_size]
                batch_t[i, time] = t[(offset + self.time_idx) % data_size]
            self.time_idx += 1
        return batch_x, batch_t

    def fit(self,
            xs,
            ts,
            max_epoch=10,
            batch_size=20,
            time_size=35,
            max_grad=None,
            eval_interval=20):
        data_size = len(xs)
        max_iters = data_size // (batch_size * time_size)
        self.time_idx = 0
        self.ppl_list = []
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            for iters in range(max_iters):
                batch_x, batch_t = self.get_batch(xs, ts, batch_size,
                                                  time_size)

                # 计算梯度, 更新参数
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params,
                                                 model.grads)  # 将共享的权重整合为1个
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1

                # 评价困惑度
                if (eval_interval
                        is not None) and (iters % eval_interval) == 0:
                    ppl = np.exp(total_loss / loss_count)
                    elapsed_time = time.time() - start_time
                    print(
                        '| epoch %d |  iter %d / %d | time %d[s] | perplexity %.2f'
                        % (self.current_epoch + 1, iters + 1, max_iters,
                           elapsed_time, ppl))
                    self.ppl_list.append(float(ppl))
                    total_loss, loss_count = 0, 0

            self.current_epoch += 1

    def plot(self, ylim=None):
        x = numpy.arange(len(self.ppl_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.ppl_list, label='train')
        plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
        plt.ylabel('perplexity')
        plt.show()


def remove_duplicate(params, grads):
    '''
    将参数列表中重复的权重整合为1个,
    加上与该权重对应的梯度
    '''
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 在共享权重的情况下
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 加上梯度
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 在作为转置矩阵共享权重的情况下(weight tying)
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads

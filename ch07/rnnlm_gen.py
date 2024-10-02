# coding: utf-8
import sys

sys.path.append('..')
import numpy as np
from common.functions import softmax
from ch06.rnnlm import Rnnlm
from ch06.better_rnnlm import BetterRnnlm


class RnnlmGen(Rnnlm):

    def generate(self, start_id, skip_ids=None, sample_size=100):
        """
        参数 start_id 是第 1 个单词 ID
        参数 sample_size 表示要采样的单词数量
        参数 skip_ids 是单词 ID 列表(比如，[12, 20]), 它指定的单词将不被采样. 这个参数用于排除 PTB 数据集中的 <unk>、N 等被预处理过的单词
        """
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            # print("generate x:", x)
            # generate x: [[316]]

            score = self.predict(x)
            # print("generate score:", score)
            # generate score: [[[-1.2524025  -0.8631877  -1.3894839  ... -0.42280924 -0.01607323 -0.39050454]]]
            # print("generate score:", score.shape)
            # generate score: (1, 1, 10000)
            # print("generate score:", score.flatten())
            # generate score: [-1.2524025  -0.8631877  -1.3894839  ... -0.42280924 -0.01607323 -0.39050454]

            p = softmax(score.flatten())
            # print("generate p:", p)
            # generate p: [2.4980120e-06 3.6866193e-06 2.1780154e-06 ... 5.7264056e-06 8.6005348e-06 5.9144159e-06]

            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

        return word_ids

    def get_state(self):
        return self.lstm_layer.h, self.lstm_layer.c

    def set_state(self, state):
        self.lstm_layer.set_state(*state)


class BetterRnnlmGen(BetterRnnlm):

    def generate(self, start_id, skip_ids=None, sample_size=100):
        word_ids = [start_id]

        x = start_id
        while len(word_ids) < sample_size:
            x = np.array(x).reshape(1, 1)
            score = self.predict(x).flatten()
            p = softmax(score).flatten()

            sampled = np.random.choice(len(p), size=1, p=p)
            if (skip_ids is None) or (sampled not in skip_ids):
                x = sampled
                word_ids.append(int(x))

        return word_ids

    def get_state(self):
        states = []
        for layer in self.lstm_layers:
            states.append((layer.h, layer.c))
        return states

    def set_state(self, states):
        for layer, state in zip(self.lstm_layers, states):
            layer.set_state(*state)

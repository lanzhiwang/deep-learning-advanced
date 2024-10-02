# coding: utf-8
import sys

sys.path.append('..')
from common.time_layers import *
from common.base_model import BaseModel


class Rnnlm(BaseModel):

    def __init__(self, vocab_size=10000, wordvec_size=100, hidden_size=100):
        """
        corpus, word_to_id, id_to_word = ptb.load_data('train')
        vocab_size = len(word_to_id)

        wordvec_size = 100
        hidden_size = 100

        model = Rnnlm(vocab_size, wordvec_size, hidden_size)
        """
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn

        # 初始化权重
        ############### TimeEmbedding 层 ###############
        # 输入 (batch_size, time_size) (20, 35)
        # 输入代表的含义是一个批次包含 20 条数据，每条数据由 35 个词语组成
        embed_W = (rn(V, D) / 100).astype('f')
        # print("Rnnlm predict embed_W:", embed_W.shape)  # Rnnlm predict embed_W: (10000, 100)
        # 输出 (20, 35, 100)
        # 输出代表的含义是一个批次包含 20 条数据，每条数据由 35 个词语组成，每个词语由 100 个元素的向量表示
        ##############################

        ############### TimeLSTM 层 ###############
        # 输入 (20, 35, 100)
        # 输入代表的含义是一个批次包含 20 条数据，每条数据由 35 个词语组成，每个词语由 100 个元素的向量表示

        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        # print("Rnnlm predict lstm_Wx:", lstm_Wx.shape)  # Rnnlm predict lstm_Wx: (100, 400)
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        # print("Rnnlm predict lstm_Wh:", lstm_Wh.shape)  # Rnnlm predict lstm_Wh: (100, 400)
        lstm_b = np.zeros(4 * H).astype('f')
        # print("Rnnlm predict lstm_b:", lstm_b.shape)  # Rnnlm predict lstm_b: (400,)
        ##############################

        ############### TimeAffine 层 ###############
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        # print("Rnnlm predict affine_W:", affine_W.shape)  # Rnnlm predict affine_W: (100, 10000)
        affine_b = np.zeros(V).astype('f')
        # print("Rnnlm predict affine_b:", affine_b.shape)  # Rnnlm predict affine_b: (10000,)
        ##############################

        # 生成层
        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layer = self.layers[1]

        # 将所有的权重和梯度整理到列表中
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def predict(self, xs):
        # print("Rnnlm predict xs:", xs.shape)  # Rnnlm predict xs: (20, 35)
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs

    def forward(self, xs, ts):
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss

    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout

    def reset_state(self):
        self.lstm_layer.reset_state()

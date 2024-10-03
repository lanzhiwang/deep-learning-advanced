# coding: utf-8
import sys

sys.path.append('..')
from common.time_layers import *
from ch07.seq2seq import Encoder, Seq2seq
from ch08.attention_layer import TimeAttention


class AttentionEncoder(Encoder):

    def forward(self, xs):
        # print("AttentionEncoder forward xs:", xs.shape)  # AttentionEncoder forward xs: (128, 29)
        xs = self.embed.forward(xs)
        # print("AttentionEncoder forward xs:", xs.shape)  # AttentionEncoder forward xs: (128, 29, 16)
        hs = self.lstm.forward(xs)
        # print("AttentionEncoder forward hs:", hs.shape)  # AttentionEncoder forward hs: (128, 29, 256)
        return hs

    def backward(self, dhs):
        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout


class AttentionDecoder:

    def __init__(self, vocab_size, wordvec_size, hidden_size):
        """
        vocab_size = len(char_to_id)  # vocab_size: 59

        wordvec_size = 16

        hidden_size = 256
        """
        V, D, H = vocab_size, wordvec_size, hidden_size
        # print("AttentionDecoder __init__ V, D, H:", V, D, H)  # AttentionDecoder __init__ V, D, H: 59 16 256
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        # print("AttentionDecoder __init__ embed_W:", embed_W.shape)  # AttentionDecoder __init__ embed_W: (59, 16)

        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        # print("AttentionDecoder __init__ lstm_Wx:", lstm_Wx.shape)  # AttentionDecoder __init__ lstm_Wx: (16, 1024)
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        # print("AttentionDecoder __init__ lstm_Wh:", lstm_Wh.shape)  # AttentionDecoder __init__ lstm_Wh: (256, 1024)
        lstm_b = np.zeros(4 * H).astype('f')
        # print("AttentionDecoder __init__ lstm_b:", lstm_b.shape)  # AttentionDecoder __init__ lstm_b: (1024,)

        affine_W = (rn(2 * H, V) / np.sqrt(2 * H)).astype('f')
        # print("AttentionDecoder __init__ affine_W:", affine_W.shape)  # AttentionDecoder __init__ affine_W: (512, 59)
        affine_b = np.zeros(V).astype('f')
        # print("AttentionDecoder __init__ affine_b:", affine_b.shape)  # AttentionDecoder __init__ affine_b: (59,)

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.attention = TimeAttention()
        self.affine = TimeAffine(affine_W, affine_b)
        layers = [self.embed, self.lstm, self.attention, self.affine]

        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, enc_hs):
        # print("AttentionDecoder forward xs:", xs.shape)  # AttentionDecoder forward xs: (128, 10)
        # print("AttentionDecoder forward enc_hs:", enc_hs.shape)  # AttentionDecoder forward enc_hs: (128, 29, 256)

        h = enc_hs[:, -1]
        # print("AttentionDecoder forward h:", h.shape)  # AttentionDecoder forward h: (128, 256)
        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        # print("AttentionDecoder forward out:", out.shape)  # (128, 10, 16)

        dec_hs = self.lstm.forward(out)
        # print("AttentionDecoder forward dec_hs:", dec_hs.shape)  # (128, 10, 256)

        c = self.attention.forward(enc_hs, dec_hs)
        # print("AttentionDecoder forward c:", c.shape)  # AttentionDecoder forward c: (128, 10, 256)

        out = np.concatenate((c, dec_hs), axis=2)
        # print("AttentionDecoder forward out:", out.shape)  # AttentionDecoder forward out: (128, 10, 512)

        score = self.affine.forward(out)
        # print("AttentionDecoder forward score:", score.shape)  # AttentionDecoder forward score: (128, 10, 59)

        return score

    def backward(self, dscore):
        dout = self.affine.backward(dscore)
        N, T, H2 = dout.shape
        H = H2 // 2

        dc, ddec_hs0 = dout[:, :, :H], dout[:, :, H:]
        denc_hs, ddec_hs1 = self.attention.backward(dc)
        ddec_hs = ddec_hs0 + ddec_hs1
        dout = self.lstm.backward(ddec_hs)
        dh = self.lstm.dh
        denc_hs[:, -1] += dh
        self.embed.backward(dout)

        return denc_hs

    def generate(self, enc_hs, start_id, sample_size):
        sampled = []
        sample_id = start_id
        h = enc_hs[:, -1]
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = np.array([sample_id]).reshape((1, 1))

            out = self.embed.forward(x)
            dec_hs = self.lstm.forward(out)
            c = self.attention.forward(enc_hs, dec_hs)
            out = np.concatenate((c, dec_hs), axis=2)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(sample_id)

        return sampled


class AttentionSeq2seq(Seq2seq):

    def __init__(self, vocab_size, wordvec_size, hidden_size):
        """
        vocab_size = len(char_to_id)  # vocab_size: 59

        wordvec_size = 16

        hidden_size = 256

        model = AttentionSeq2seq(vocab_size, wordvec_size, hidden_size)
        """
        args = vocab_size, wordvec_size, hidden_size
        self.encoder = AttentionEncoder(*args)
        self.decoder = AttentionDecoder(*args)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

# coding: utf-8
import sys

sys.path.append('..')
from common.time_layers import *
from common.base_model import BaseModel


class Encoder:

    def __init__(self, vocab_size, wordvec_size, hidden_size):
        """
        vocab_size = len(char_to_id)  # vocab_size: 59

        wordvec_size = 16

        hidden_size = 256
        """
        V, D, H = vocab_size, wordvec_size, hidden_size
        # print("Encoder __init__ V, D, H:", V, D, H)  # Encoder __init__ V, D, H: 59 16 256
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        # print("Encoder __init__ embed_W:", embed_W.shape)  # Encoder __init__ embed_W: (59, 16)

        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        # print("Encoder __init__ lstm_Wx:", lstm_Wx.shape)  # Encoder __init__ lstm_Wx: (16, 1024)
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        # print("Encoder __init__ lstm_Wh:", lstm_Wh.shape)  # Encoder __init__ lstm_Wh: (256, 1024)
        lstm_b = np.zeros(4 * H).astype('f')
        # print("Encoder __init__ lstm_b:", lstm_b.shape)  # Encoder __init__ lstm_b: (1024,)

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False)

        self.params = self.embed.params + self.lstm.params
        self.grads = self.embed.grads + self.lstm.grads
        self.hs = None

    def forward(self, xs):
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        self.hs = hs
        return hs[:, -1, :]

    def backward(self, dh):
        dhs = np.zeros_like(self.hs)
        dhs[:, -1, :] = dh

        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout


class Decoder:

    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        # print("Decoder __init__ V, D, H:", V, D, H)  # Decoder __init__ V, D, H: 13 16 128
        rn = np.random.randn

        embed_W = (rn(V, D) / 100).astype('f')
        lstm_Wx = (rn(D, 4 * H) / np.sqrt(D)).astype('f')
        lstm_Wh = (rn(H, 4 * H) / np.sqrt(H)).astype('f')
        lstm_b = np.zeros(4 * H).astype('f')
        affine_W = (rn(H, V) / np.sqrt(H)).astype('f')
        affine_b = np.zeros(V).astype('f')

        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)

        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, xs, h):
        self.lstm.set_state(h)

        out = self.embed.forward(xs)
        out = self.lstm.forward(out)
        score = self.affine.forward(out)
        return score

    def backward(self, dscore):
        dout = self.affine.backward(dscore)
        dout = self.lstm.backward(dout)
        dout = self.embed.backward(dout)
        dh = self.lstm.dh
        return dh

    def generate(self, h, start_id, sample_size):
        sampled = []
        sample_id = start_id
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = np.array(sample_id).reshape((1, 1))
            out = self.embed.forward(x)
            out = self.lstm.forward(out)
            score = self.affine.forward(out)

            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))

        return sampled


class Seq2seq(BaseModel):

    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        # print("Seq2seq __init__ V, D, H:", V, D, H)  # Seq2seq __init__ V, D, H: 13 16 128
        self.encoder = Encoder(V, D, H)
        self.decoder = Decoder(V, D, H)
        self.softmax = TimeSoftmaxWithLoss()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def forward(self, xs, ts):
        # print("Seq2seq forward xs:", xs.shape)  # Seq2seq forward xs: (128, 7)
        # print("Seq2seq forward ts:", ts.shape)  # Seq2seq forward ts: (128, 5)

        decoder_xs, decoder_ts = ts[:, :-1], ts[:, 1:]
        # print("Seq2seq forward decoder_xs:", decoder_xs.shape)  # Seq2seq forward decoder_xs: (128, 4)
        # print("Seq2seq forward decoder_ts:", decoder_ts.shape)  # Seq2seq forward decoder_ts: (128, 4)

        h = self.encoder.forward(xs)
        score = self.decoder.forward(decoder_xs, h)
        loss = self.softmax.forward(score, decoder_ts)
        return loss

    def backward(self, dout=1):
        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)
        return dout

    def generate(self, xs, start_id, sample_size):
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled

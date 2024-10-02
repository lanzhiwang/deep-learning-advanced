# coding: utf-8
import sys

sys.path.append('..')
from rnnlm_gen import RnnlmGen
from dataset import ptb

corpus, word_to_id, id_to_word = ptb.load_data('train')
# print("corpus[:10]:", corpus[:10])  # corpus[:10]: [0 1 2 3 4 5 6 7 8 9]

vocab_size = len(word_to_id)
# print("vocab_size:", vocab_size)  # vocab_size: 10000

corpus_size = len(corpus)
# print("corpus_size:", corpus_size)  # corpus_size: 929589

model = RnnlmGen()
model.load_params('../ch06/Rnnlm.pkl')

# 设定start单词和skip单词
start_word = 'you'
start_id = word_to_id[start_word]
# print("start_id:", start_id)  # start_id: 316

skip_words = ['N', '<unk>', '$']
skip_ids = [word_to_id[w] for w in skip_words]
# print("skip_ids:", skip_ids)  # skip_ids: [27, 26, 416]

# 文本生成
word_ids = model.generate(start_id, skip_ids)
# print("word_ids:", word_ids)
# word_ids: [316, 1490, 32, 9059, 8978, 181, 64, 805, 4356, 1124, 229, 32, 711, 24, 4423, 956, 9878, 636, 42, 8423, 2203, 1378, 119, 6056, 4950, 87, 2247, 4281, 62, 34, 1775, 6065, 32, 3180, 24, 7322, 338, 152, 307, 7135, 2352, 48, 313, 718, 65, 556, 842, 1546, 1031, 3229, 24, 5298, 620, 3392, 3348, 2507, 689, 4955, 330, 289, 987, 7422, 48, 35, 79, 80, 32, 1523, 499, 24, 315, 93, 586, 32, 635, 54, 7643, 42, 9352, 581, 64, 2271, 48, 5788, 570, 4593, 48, 531, 145, 556, 1380, 48, 1282, 134, 32, 5857, 42, 504, 6106, 5603]

txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')
print(txt)

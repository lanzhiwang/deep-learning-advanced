# coding: utf-8
import sys

sys.path.append('..')
from common.np import *
from rnnlm_gen import BetterRnnlmGen
from dataset import ptb

corpus, word_to_id, id_to_word = ptb.load_data('train')
# print("corpus[:10]:", corpus[:10])  # corpus[:10]: [0 1 2 3 4 5 6 7 8 9]

vocab_size = len(word_to_id)
# print("vocab_size:", vocab_size)  # vocab_size: 10000

corpus_size = len(corpus)
# print("corpus_size:", corpus_size)  # corpus_size: 929589

model = BetterRnnlmGen()
model.load_params('../ch06/BetterRnnlm.pkl')

# 设定start字符和skip字符
start_word = 'you'
start_id = word_to_id[start_word]
skip_words = ['N', '<unk>', '$']
skip_ids = [word_to_id[w] for w in skip_words]
# 文本生成
word_ids = model.generate(start_id, skip_ids)
txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')

print(txt)

model.reset_state()

start_words = 'the meaning of life is'
start_ids = [word_to_id[w] for w in start_words.split(' ')]

for x in start_ids[:-1]:
    x = np.array(x).reshape(1, 1)
    model.predict(x)

word_ids = model.generate(start_ids[-1], skip_ids)
word_ids = start_ids[:-1] + word_ids
txt = ' '.join([id_to_word[i] for i in word_ids])
txt = txt.replace(' <eos>', '.\n')
print('-' * 50)
print(txt)

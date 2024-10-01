# coding: utf-8
import sys

sys.path.append('..')  # 为了引入父目录的文件而进行的设定
from common.trainer import Trainer
from common.optimizer import Adam
from simple_cbow import SimpleCBOW
from common.util import preprocess, create_contexts_target, convert_one_hot

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
# print("corpus:", corpus)
# corpus: [0 1 2 3 4 1 5 6]
# print("word_to_id:", word_to_id)
# word_to_id: {'you': 0, 'say': 1, 'goodbye': 2, 'and': 3, 'i': 4, 'hello': 5, '.': 6}
print("id_to_word:", id_to_word)
# id_to_word: {0: 'you', 1: 'say', 2: 'goodbye', 3: 'and', 4: 'i', 5: 'hello', 6: '.'}

vocab_size = len(word_to_id)
# print("vocab_size:", vocab_size)
# vocab_size: 7

contexts, target = create_contexts_target(corpus, window_size)
# print("contexts:", contexts)
# contexts: [[0 2]
#  [1 3]
#  [2 4]
#  [3 1]
#  [4 5]
#  [1 6]]
# print("target:", target)
# target: [1 2 3 4 1 5]

target = convert_one_hot(target, vocab_size)
# print("target:", target)
# target: [[0 1 0 0 0 0 0]
#  [0 0 1 0 0 0 0]
#  [0 0 0 1 0 0 0]
#  [0 0 0 0 1 0 0]
#  [0 1 0 0 0 0 0]
#  [0 0 0 0 0 1 0]]

contexts = convert_one_hot(contexts, vocab_size)
# print("contexts:", contexts)
# contexts: [[[1 0 0 0 0 0 0]
#   [0 0 1 0 0 0 0]]

#  [[0 1 0 0 0 0 0]
#   [0 0 0 1 0 0 0]]

#  [[0 0 1 0 0 0 0]
#   [0 0 0 0 1 0 0]]

#  [[0 0 0 1 0 0 0]
#   [0 1 0 0 0 0 0]]

#  [[0 0 0 0 1 0 0]
#   [0 0 0 0 0 1 0]]

#  [[0 1 0 0 0 0 0]
#   [0 0 0 0 0 0 1]]]

model = SimpleCBOW(vocab_size, hidden_size)
optimizer = Adam()
trainer = Trainer(model, optimizer)

trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

word_vecs = model.word_vecs
for word_id, word in id_to_word.items():
    print(word, word_vecs[word_id])

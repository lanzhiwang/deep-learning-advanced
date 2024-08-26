# coding: utf-8
import sys

sys.path.append('..')
from common.util import preprocess, create_co_matrix, cos_similarity

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
# print("corpus:", corpus)
# corpus: [0 1 2 3 4 1 5 6]
# print("word_to_id:", word_to_id)
# word_to_id: {'you': 0, 'say': 1, 'goodbye': 2, 'and': 3, 'i': 4, 'hello': 5, '.': 6}
# print("id_to_word:", id_to_word)
# id_to_word: {0: 'you', 1: 'say', 2: 'goodbye', 3: 'and', 4: 'i', 5: 'hello', 6: '.'}

vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
print(C)
# [[0 1 0 0 0 0 0]
#  [1 0 1 0 1 1 0]
#  [0 1 0 1 0 0 0]
#  [0 0 1 0 1 0 0]
#  [0 1 0 1 0 0 0]
#  [0 1 0 0 0 0 1]
#  [0 0 0 0 0 1 0]]

c0 = C[word_to_id['you']]  # you 的单词向量
c1 = C[word_to_id['i']]  # i 的单词向量
print(cos_similarity(c0, c1))
# 0.7071067691154799

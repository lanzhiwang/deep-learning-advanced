# coding: utf-8
import sys

sys.path.append('..')
from common.util import preprocess, create_co_matrix

# >>> import numpy as np
# >>> text = 'You say goodbye and I say hello.'
# >>> text
# 'You say goodbye and I say hello.'
# >>> text = text.lower()
# >>> text
# 'you say goodbye and i say hello.'
# >>> text = text.replace('.', ' .')
# >>> text
# 'you say goodbye and i say hello .'
# >>> words = text.split(' ')
# >>> words
# ['you', 'say', 'goodbye', 'and', 'i', 'say', 'hello', '.']
# >>> word_to_id = {}
# >>> id_to_word = {}
# >>> for word in words:
# ...     if word not in word_to_id:
# ...         new_id = len(word_to_id)
# ...         word_to_id[word] = new_id
# ...         id_to_word[new_id] = word
# ...
# >>> word_to_id
# {'you': 0, 'say': 1, 'goodbye': 2, 'and': 3, 'i': 4, 'hello': 5, '.': 6}
# >>> id_to_word
# {0: 'you', 1: 'say', 2: 'goodbye', 3: 'and', 4: 'i', 5: 'hello', 6: '.'}
# >>> [word_to_id[w] for w in words]
# [0, 1, 2, 3, 4, 1, 5, 6]
# >>> corpus = np.array([word_to_id[w] for w in words])
# >>> corpus
# array([0, 1, 2, 3, 4, 1, 5, 6])
# >>>
'''
        you say goodbye and i hello .
you      0   1     0     0  0   0   0
say      1   0     1     0  1   1   0
goodbye  0   1     0     1  0   0   0
and      0   0     1     0  1   0   0
i        0   1     0     1  0   0   0
hello    0   1     0     0  0   0   1
.        0   0     0     0  0   1   0
'''

text = 'You say goodbye and I say hello.'
print("text:", text)
corpus, word_to_id, id_to_word = preprocess(text)
print("corpus:", corpus)
print("word_to_id:", word_to_id)
print("id_to_word:", id_to_word)

vocab_size = len(id_to_word)
C = create_co_matrix(corpus, vocab_size, window_size=1)
print("C:", C)

# text: You say goodbye and I say hello.
# corpus: [0 1 2 3 4 1 5 6]
# word_to_id: {'you': 0, 'say': 1, 'goodbye': 2, 'and': 3, 'i': 4, 'hello': 5, '.': 6}
# id_to_word: {0: 'you', 1: 'say', 2: 'goodbye', 3: 'and', 4: 'i', 5: 'hello', 6: '.'}
# C:
# [
#     [0 1 0 0 0 0 0]
#     [1 0 1 0 1 1 0]
#     [0 1 0 1 0 0 0]
#     [0 0 1 0 1 0 0]
#     [0 1 0 1 0 0 0]
#     [0 1 0 0 0 0 1]
#     [0 0 0 0 0 1 0]
# ]

# 计算余弦相似度
# >>> import numpy as np
# >>> x = np.array([1, 2, 3, 4, 5, 6, 7])
# >>> y = np.array([7, 6, 5, 4, 3, 2, 1])
# >>> x**2
# array([ 1,  4,  9, 16, 25, 36, 49])
# >>> np.sum(x**2)
# 140
# >>> np.sqrt(np.sum(x**2))
# 11.832159566199232
# >>> x / np.sqrt(np.sum(x**2))
# array([0.08451543, 0.16903085, 0.25354628, 0.3380617 , 0.42257713,
#        0.50709255, 0.59160798])
# >>> y**2
# array([49, 36, 25, 16,  9,  4,  1])
# >>> np.sum(y**2)
# 140
# >>> np.sqrt(np.sum(y**2))
# 11.832159566199232
# >>> y / np.sqrt(np.sum(y**2))
# array([0.59160798, 0.50709255, 0.42257713, 0.3380617 , 0.25354628,
#        0.16903085, 0.08451543])
# >>> np.dot(x / np.sqrt(np.sum(x**2)), y / np.sqrt(np.sum(y**2)))
# 0.6
# >>>

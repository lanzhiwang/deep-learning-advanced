# coding: utf-8
import sys

sys.path.append('..')
import numpy as np
from common.util import most_similar, create_co_matrix, ppmi
from dataset import ptb

window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data('train')
print("corpus:", len(corpus))
# corpus: 929589

vocab_size = len(word_to_id)
print('counting  co-occurrence ...')
C = create_co_matrix(corpus, vocab_size, window_size)
print("C:", C.shape)
# C: (10000, 10000)

print('calculating PPMI ...')
W = ppmi(C, verbose=True)
print("W:", W.shape)
# W: (10000, 10000)

print('calculating SVD ...')
try:
    # truncated SVD (fast!)
    from sklearn.utils.extmath import randomized_svd
    U, S, V = randomized_svd(W,
                             n_components=wordvec_size,
                             n_iter=5,
                             random_state=None)
except ImportError:
    # SVD (slow)
    U, S, V = np.linalg.svd(W)
print("U:", U.shape)
# U: (10000, 100)
print("S:", S.shape)
# S: (100,)
print("V:", V.shape)
# V: (100, 10000)

word_vecs = U[:, :wordvec_size]

querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

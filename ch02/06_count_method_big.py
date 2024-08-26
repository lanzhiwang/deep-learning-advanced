# coding: utf-8
import sys

sys.path.append('..')
import numpy as np
from common.util import most_similar, create_co_matrix, ppmi
from dataset import ptb

window_size = 2
wordvec_size = 100

corpus, word_to_id, id_to_word = ptb.load_data('train')
vocab_size = len(word_to_id)
print('counting  co-occurrence ...')
C = create_co_matrix(corpus, vocab_size, window_size)
print('calculating PPMI ...')
W = ppmi(C, verbose=True)

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

word_vecs = U[:, :wordvec_size]

querys = ['you', 'year', 'car', 'toyota']
for query in querys:
    most_similar(query, word_to_id, id_to_word, word_vecs, top=5)

# counting  co-occurrence ...
# calculating PPMI ...
# 1.0% done
# 2.0% done
# 3.0% done
# 4.0% done
# 5.0% done
# 6.0% done
# 7.0% done
# 8.0% done
# 9.0% done
# 10.0% done
# 11.0% done
# 12.0% done
# 13.0% done
# 14.0% done
# 15.0% done
# 16.0% done
# 17.0% done
# 18.0% done
# 19.0% done
# 20.0% done
# 21.0% done
# 22.0% done
# 23.0% done
# 24.0% done
# 25.0% done
# 26.0% done
# 27.0% done
# 28.0% done
# 29.0% done
# 30.0% done
# 31.0% done
# 32.0% done
# 33.0% done
# 34.0% done
# 35.0% done
# 36.0% done
# 37.0% done
# 38.0% done
# 39.0% done
# 40.0% done
# 41.0% done
# 42.0% done
# 43.0% done
# 44.0% done
# 45.0% done
# 46.0% done
# 47.0% done
# 48.0% done
# 49.0% done
# 50.0% done
# 51.0% done
# 52.0% done
# 53.0% done
# 54.0% done
# 55.0% done
# 56.0% done
# 57.0% done
# 58.0% done
# 59.0% done
# 60.0% done
# 61.0% done
# 62.0% done
# 63.0% done
# 64.0% done
# 65.0% done
# 66.0% done
# 67.0% done
# 68.0% done
# 69.0% done
# 70.0% done
# 71.0% done
# 72.0% done
# 73.0% done
# 74.0% done
# 75.0% done
# 76.0% done
# 77.0% done
# 78.0% done
# 79.0% done
# 80.0% done
# 81.0% done
# 82.0% done
# 83.0% done
# 84.0% done
# 85.0% done
# 86.0% done
# 87.0% done
# 88.0% done
# 89.0% done
# 90.0% done
# 91.0% done
# 92.0% done
# 93.0% done
# 94.0% done
# 95.0% done
# 96.0% done
# 97.0% done
# 98.0% done
# 99.0% done
# calculating SVD ...

# [query] you
#  i: 0.6705275177955627
#  do: 0.6088383793830872
#  we: 0.5567812323570251
#  'll: 0.5236623883247375
#  always: 0.5020409226417542

# [query] year
#  quarter: 0.6607533097267151
#  month: 0.659645140171051
#  earlier: 0.5970752835273743
#  last: 0.5935207605361938
#  third: 0.5873794555664062

# [query] car
#  auto: 0.6500502824783325
#  luxury: 0.6243090629577637
#  cars: 0.5466026067733765
#  truck: 0.5419286489486694
#  domestic: 0.49225810170173645

# [query] toyota
#  motor: 0.7002094984054565
#  lexus: 0.6314260363578796
#  honda: 0.6184332370758057
#  motors: 0.6015571355819702
#  nissan: 0.5958844423294067

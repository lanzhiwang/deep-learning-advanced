# coding: utf-8
import sys

sys.path.append('..')  # 为了引入父目录的文件而进行的设定
from common.trainer import Trainer
from common.optimizer import Adam
from common.util import preprocess, create_contexts_target
from cbow import CBOW

window_size = 1
hidden_size = 5
batch_size = 3
max_epoch = 1000

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

# print("text:", text)
# text: You say goodbye and I say hello.

# print("corpus:", corpus)
# corpus: [0 1 2 3 4 1 5 6]
# print("word_to_id:", word_to_id)
# word_to_id: {'you': 0, 'say': 1, 'goodbye': 2, 'and': 3, 'i': 4, 'hello': 5, '.': 6}

# print("id_to_word:", id_to_word)
# id_to_word: {0: 'you', 1: 'say', 2: 'goodbye', 3: 'and', 4: 'i', 5: 'hello', 6: '.'}
'''
|            corpus                |   contexts       | target  |
| You say goodbye and I say hello. | You      goodbye | say     |
| You say goodbye and I say hello. | say      and     | goodbye |
| You say goodbye and I say hello. | goodbye  I       | and     |
| You say goodbye and I say hello. | and      say     | I       |
| You say goodbye and I say hello. | I        hello   | say     |
| You say goodbye and I say hello. | say      .       | hello   |

|            corpus                | contexts | target |
| You say goodbye and I say hello. | 0   2    |  1     |
| You say goodbye and I say hello. | 1   3    |  2     |
| You say goodbye and I say hello. | 2   4    |  3     |
| You say goodbye and I say hello. | 3   1    |  4     |
| You say goodbye and I say hello. | 4   5    |  1     |
| You say goodbye and I say hello. | 1   6    |  5     |

| 0 | you     | 1 0 0 0 0 0 0 |
| 1 | say     | 0 1 0 0 0 0 0 |
| 2 | goodbye | 0 0 1 0 0 0 0 |
| 3 | and     | 0 0 0 1 0 0 0 |
| 4 | i       | 0 0 0 0 1 0 0 |
| 5 | hello   | 0 0 0 0 0 1 0 |
| 6 | .       | 0 0 0 0 0 0 1 |
'''

vocab_size = len(word_to_id)
contexts, target = create_contexts_target(corpus, window_size)

# print("contexts:", contexts)
# contexts:
# [
#     [0 2]
#     [1 3]
#     [2 4]
#     [3 1]
#     [4 5]
#     [1 6]
# ]

# print("target:", target)
# target: [1 2 3 4 1 5]

# 生成模型等
model = CBOW(vocab_size, hidden_size, window_size, corpus)
# model = SkipGram(vocab_size, hidden_size, window_size, corpus)
optimizer = Adam()
trainer = Trainer(model, optimizer)

# 开始学习
trainer.fit(contexts, target, max_epoch, batch_size)
trainer.plot()

# # 保存必要数据, 以便后续使用
# word_vecs = model.word_vecs
# if config.GPU:
#     word_vecs = to_cpu(word_vecs)
# params = {}
# params['word_vecs'] = word_vecs.astype(np.float16)
# params['word_to_id'] = word_to_id
# params['id_to_word'] = id_to_word
# pkl_file = 'cbow_params.pkl'  # or 'skipgram_params.pkl'
# with open(pkl_file, 'wb') as f:
#     pickle.dump(params, f, -1)

# coding: utf-8
import sys
import os

sys.path.append('..')
try:
    import urllib.request
except ImportError:
    raise ImportError('Use Python3!')
import pickle
import numpy as np

url_base = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/'
key_file = {
    'train': 'ptb.train.txt',
    'test': 'ptb.test.txt',
    'valid': 'ptb.valid.txt'
}
save_file = {
    'train': 'ptb.train.npy',
    'test': 'ptb.test.npy',
    'valid': 'ptb.valid.npy'
}
vocab_file = 'ptb.vocab.pkl'

dataset_dir = os.path.dirname(os.path.abspath(__file__))
# print("dataset_dir:", dataset_dir)
# dataset_dir: /workspaces/deep-learning-advanced/dataset


def _download(file_name):
    # print("_download file_name:", file_name)
    # _download file_name: ptb.train.txt
    # _download file_name: ptb.train.txt
    # _download file_name: ptb.valid.txt
    # _download file_name: ptb.test.txt

    file_path = dataset_dir + '/' + file_name
    # print("_download file_path:", file_path)
    # _download file_path: /workspaces/deep-learning-advanced/dataset/ptb.train.txt
    # _download file_path: /workspaces/deep-learning-advanced/dataset/ptb.train.txt
    # _download file_path: /workspaces/deep-learning-advanced/dataset/ptb.valid.txt
    # _download file_path: /workspaces/deep-learning-advanced/dataset/ptb.test.txt

    if os.path.exists(file_path):
        return

    print('Downloading ' + file_name + ' ... ')
    # Downloading ptb.train.txt ...

    try:
        urllib.request.urlretrieve(url_base + file_name, file_path)
    except urllib.error.URLError:
        import ssl
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(url_base + file_name, file_path)

    print('Done')


def load_vocab():
    vocab_path = dataset_dir + '/' + vocab_file
    # print("load_vocab vocab_path:", vocab_path)
    # load_vocab vocab_path: /workspaces/deep-learning-advanced/dataset/ptb.vocab.pkl

    if os.path.exists(vocab_path):
        with open(vocab_path, 'rb') as f:
            word_to_id, id_to_word = pickle.load(f)
        return word_to_id, id_to_word

    word_to_id = {}
    id_to_word = {}
    data_type = 'train'
    file_name = key_file[data_type]
    file_path = dataset_dir + '/' + file_name
    # print("load_vocab file_name:", file_name)
    # print("load_vocab file_path:", file_path)
    # load_vocab file_name: ptb.train.txt
    # load_vocab file_path: /workspaces/deep-learning-advanced/dataset/ptb.train.txt

    _download(file_name)

    words = open(file_path).read().replace('\n', '<eos>').strip().split()
    # print("load_vocab words:", words)
    """
    ['aer', 'banknote', 'berlitz', 'calloway', 'centrust', 'cluett', 'fromstein', 'gitano', 'guterman', 'hydro-quebec', 'ipo', 'kia', 'memotec', 'mlx', 'nahb', 'punts', 'rake', 'regatta', 'rubens', 'sim', 'snack-food', 'ssangyong', 'swapo', 'wachter',
     '<eos>',
     'pierre', '<unk>', 'N', 'years', 'old', 'will', 'join', 'the', 'board', 'as', 'a', 'nonexecutive', 'director', 'nov.', 'N',
     '<eos>',
     'mr.', '<unk>', 'is', 'chairman', 'of', '<unk>', 'n.v.', 'the', 'dutch', 'publishing', 'group'
    ]
    """

    for i, word in enumerate(words):
        if word not in word_to_id:
            tmp_id = len(word_to_id)
            word_to_id[word] = tmp_id
            id_to_word[tmp_id] = word
    # print("load_vocab word_to_id:", word_to_id)
    # print("load_vocab id_to_word:", id_to_word)
    """
    {'aer': 0, 'banknote': 1, 'berlitz': 2, 'calloway': 3, 'centrust': 4, 'cluett': 5, 'fromstein': 6, 'gitano': 7, 'guterman': 8, 'hydro-quebec': 9, 'ipo': 10, 'kia': 11, 'memotec': 12, 'mlx': 13, 'nahb': 14, 'punts': 15, 'rake': 16, 'regatta': 17, 'rubens': 18, 'sim': 19, 'snack-food': 20, 'ssangyong': 21, 'swapo': 22, 'wachter': 23, '<eos>': 24, 'pierre': 25, '<unk>': 26, 'N': 27, 'years': 28, 'old': 29, 'will': 30, 'join': 31, 'the': 32, 'board': 33, 'as': 34, 'a': 35, 'nonexecutive': 36, 'director': 37, 'nov.': 38, 'mr.': 39, 'is': 40, 'chairman': 41, 'of': 42, 'n.v.': 43, 'dutch': 44, 'publishing': 45, 'group': 46, '<eos><eos><eos>': 47, "['aer',": 48, "'banknote',": 49, "'berlitz',": 50, "'calloway',": 51, "'centrust',": 52, "'cluett',": 53, "'fromstein',": 54, "'gitano',": 55, "'guterman',": 56, "'hydro-quebec',": 57, "'ipo',": 58, "'kia',": 59, "'memotec',": 60, "'mlx',": 61, "'nahb',": 62, "'punts',": 63, "'rake',": 64, "'regatta',": 65, "'rubens',": 66, "'sim',": 67, "'snack-food',": 68, "'ssangyong',": 69, "'swapo',": 70, "'wachter',": 71, "'<eos>',": 72, "'pierre',": 73, "'<unk>',": 74, "'N',": 75, "'years',": 76, "'old',": 77, "'will',": 78, "'join',": 79, "'the',": 80, "'board',": 81, "'as',": 82, "'a',": 83, "'nonexecutive',": 84, "'director',": 85, "'nov.',": 86, "'mr.',": 87, "'is',": 88, "'chairman',": 89, "'of',": 90, "'n.v.',": 91, "'dutch',": 92, "'publishing',": 93, "'group'<eos>]<eos><eos>": 94}
    {0: 'aer', 1: 'banknote', 2: 'berlitz', 3: 'calloway', 4: 'centrust', 5: 'cluett', 6: 'fromstein', 7: 'gitano', 8: 'guterman', 9: 'hydro-quebec', 10: 'ipo', 11: 'kia', 12: 'memotec', 13: 'mlx', 14: 'nahb', 15: 'punts', 16: 'rake', 17: 'regatta', 18: 'rubens', 19: 'sim', 20: 'snack-food', 21: 'ssangyong', 22: 'swapo', 23: 'wachter', 24: '<eos>', 25: 'pierre', 26: '<unk>', 27: 'N', 28: 'years', 29: 'old', 30: 'will', 31: 'join', 32: 'the', 33: 'board', 34: 'as', 35: 'a', 36: 'nonexecutive', 37: 'director', 38: 'nov.', 39: 'mr.', 40: 'is', 41: 'chairman', 42: 'of', 43: 'n.v.', 44: 'dutch', 45: 'publishing', 46: 'group', 47: '<eos><eos><eos>', 48: "['aer',", 49: "'banknote',", 50: "'berlitz',", 51: "'calloway',", 52: "'centrust',", 53: "'cluett',", 54: "'fromstein',", 55: "'gitano',", 56: "'guterman',", 57: "'hydro-quebec',", 58: "'ipo',", 59: "'kia',", 60: "'memotec',", 61: "'mlx',", 62: "'nahb',", 63: "'punts',", 64: "'rake',", 65: "'regatta',", 66: "'rubens',", 67: "'sim',", 68: "'snack-food',", 69: "'ssangyong',", 70: "'swapo',", 71: "'wachter',", 72: "'<eos>',", 73: "'pierre',", 74: "'<unk>',", 75: "'N',", 76: "'years',", 77: "'old',", 78: "'will',", 79: "'join',", 80: "'the',", 81: "'board',", 82: "'as',", 83: "'a',", 84: "'nonexecutive',", 85: "'director',", 86: "'nov.',", 87: "'mr.',", 88: "'is',", 89: "'chairman',", 90: "'of',", 91: "'n.v.',", 92: "'dutch',", 93: "'publishing',", 94: "'group'<eos>]<eos><eos>"}
    """

    with open(vocab_path, 'wb') as f:
        pickle.dump((word_to_id, id_to_word), f)

    return word_to_id, id_to_word


def load_data(data_type='train'):
    '''
        :param data_type: 数据的种类：'train' or 'test' or 'valid (val)'
        :return:
    '''
    # print("load_data data_type:", data_type)
    # load_data data_type: train
    # load_data data_type: val
    # load_data data_type: test

    if data_type == 'val':
        data_type = 'valid'
    save_path = dataset_dir + '/' + save_file[data_type]
    # print("load_data save_path:", save_path)
    # load_data save_path: /workspaces/deep-learning-advanced/dataset/ptb.train.npy
    # load_data save_path: /workspaces/deep-learning-advanced/dataset/ptb.valid.npy
    # load_data save_path: /workspaces/deep-learning-advanced/dataset/ptb.test.npy

    word_to_id, id_to_word = load_vocab()

    if os.path.exists(save_path):
        corpus = np.load(save_path)
        return corpus, word_to_id, id_to_word

    file_name = key_file[data_type]
    file_path = dataset_dir + '/' + file_name
    # print("load_data file_name:", file_name)
    # print("load_data file_path:", file_path)
    # load_data file_name: ptb.train.txt
    # load_data file_path: /workspaces/deep-learning-advanced/dataset/ptb.train.txt
    # load_data file_name: ptb.valid.txt
    # load_data file_path: /workspaces/deep-learning-advanced/dataset/ptb.valid.txt
    # load_data file_name: ptb.test.txt
    # load_data file_path: /workspaces/deep-learning-advanced/dataset/ptb.test.txt
    _download(file_name)

    words = open(file_path).read().replace('\n', '<eos>').strip().split()
    corpus = np.array([word_to_id[w] for w in words])
    # print("load_data corpus:", corpus)
    """
    [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
      24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 27 24 39 26 40 41 42 26 43
      32 44 45 46]
    """

    np.save(save_path, corpus)
    return corpus, word_to_id, id_to_word


if __name__ == '__main__':
    for data_type in ('train', 'val', 'test'):
        load_data(data_type)

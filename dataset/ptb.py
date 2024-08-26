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

    for i, word in enumerate(words):
        if word not in word_to_id:
            tmp_id = len(word_to_id)
            word_to_id[word] = tmp_id
            id_to_word[tmp_id] = word

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

    np.save(save_path, corpus)
    return corpus, word_to_id, id_to_word


if __name__ == '__main__':
    for data_type in ('train', 'val', 'test'):
        load_data(data_type)

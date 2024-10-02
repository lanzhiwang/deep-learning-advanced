# coding: utf-8
import sys

sys.path.append('..')
import os
import numpy

id_to_char = {}
char_to_id = {}


def _update_vocab(txt):
    chars = list(txt)

    for i, char in enumerate(chars):
        if char not in char_to_id:
            tmp_id = len(char_to_id)
            char_to_id[char] = tmp_id
            id_to_char[tmp_id] = char


def load_data(file_name='addition.txt', seed=1984):
    file_path = os.path.dirname(os.path.abspath(__file__)) + '/' + file_name
    # print("file_path:", file_path)  # file_path: /02-deep-learning-advanced/dataset/addition.txt

    if not os.path.exists(file_path):
        print('No file: %s' % file_name)
        return None

    questions, answers = [], []

    for line in open(file_path, 'r'):
        # print("line:", line)  # line: 431+98 _529

        idx = line.find('_')
        # print("idx:", idx)  # idx: 7

        # print("line[:idx]:", line[:idx])  # line[:idx]: 431+98
        # print("line[idx:-1]:", line[idx:-1])  # line[idx:-1]: _529
        questions.append(line[:idx])
        answers.append(line[idx:-1])

    # print("questions:", questions[:10])
    # print("answers:", answers[:10])

    # create vocab dict
    for i in range(len(questions)):
        q, a = questions[i], answers[i]
        _update_vocab(q)
        _update_vocab(a)

    # create numpy array
    x = numpy.zeros((len(questions), len(questions[0])), dtype=numpy.int32)
    # print("x:", x.shape)  # x: (50000, 7)

    t = numpy.zeros((len(questions), len(answers[0])), dtype=numpy.int32)
    # print("t:", t.shape)  # t: (50000, 5)

    for i, sentence in enumerate(questions):
        x[i] = [char_to_id[c] for c in list(sentence)]
    for i, sentence in enumerate(answers):
        t[i] = [char_to_id[c] for c in list(sentence)]

    # shuffle
    indices = numpy.arange(len(x))
    if seed is not None:
        numpy.random.seed(seed)
    numpy.random.shuffle(indices)
    x = x[indices]
    t = t[indices]

    # 10% for validation set
    split_at = len(x) - len(x) // 10
    (x_train, x_test) = x[:split_at], x[split_at:]
    (t_train, t_test) = t[:split_at], t[split_at:]

    return (x_train, t_train), (x_test, t_test)


def get_vocab():
    return char_to_id, id_to_char


if __name__ == '__main__':
    (x_train, t_train), (x_test, t_test) = load_data('addition.txt', seed=1984)
    print("x_train:", x_train.shape)  # x_train: (45000, 7)
    print("t_train:", t_train.shape)  # t_train: (45000, 5)
    print("x_test:", x_test.shape)  # x_test: (5000, 7)
    print("t_test:", t_test.shape)  # t_test: (5000, 5)

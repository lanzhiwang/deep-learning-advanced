import numpy as np

max_epoch = 1
time_idx = 0
corpus_size = 5

batch_size = 2
time_size = 2

xs = [0, 1, 2, 3]
data_size = len(xs)

max_iters = data_size // (batch_size * time_size)
print("max_iters:", max_iters)

jump = (corpus_size - 1) // batch_size
offsets = [i * jump for i in range(batch_size)]

for epoch in range(max_epoch):
    for iter in range(max_iters):
        # print("iter:", iter)
        # 获取mini-batch
        batch_x = np.empty((batch_size, time_size), dtype='i')  # (10, 50)
        for t in range(time_size):
            # print("    t:", t, "time_idx:", time_idx)
            for i, offset in enumerate(offsets):
                # print("        i:", i, "\toffset:", offset, "\ttime_idx:", time_idx, "\t", offset + time_idx, "\t", (offset + time_idx) % data_size)
                batch_x[i, t] = xs[(offset + time_idx) % data_size]
            time_idx += 1

        print("batch_x:", batch_x)
"""

(2, 2) -> (2, 2, 2)

[
    [
        0
        1
    ]

    [
        2
        3
    ]
]

->

[
    [
        [0 1]
        [2 3]
    ]

    [
        [4 5]
        [6 7]
    ]
]

xs[:, 0, :]
[
    [0 1]
    [4 5]
]
xs[:, 1, :]
[
    [2 3]
    [6 7]
]


rnn_Wx:
[
    [1 1]
    [1 1]
]
rnn_Wh:
[
    [1 1]
    [1 1]
]
rnn_b: [1 1]

        TimeRNN before self.h: [[0 0]
                                [0 0]]
        TimeRNN before xs[:, t, :]: [[0 1]
                                     [4 5]]
        TimeRNN after self.h: [[ 2  2]
                               [10 10]]

        TimeRNN before self.h: [[ 2  2]
                                [10 10]]
        TimeRNN before xs[:, t, :]: [[2 3]
                                     [6 7]]
        TimeRNN after self.h: [[10 10]
                               [34 34]]

"""

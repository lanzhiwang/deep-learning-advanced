# coding: utf-8
'''
>>> import numpy as np
>>> contexts = np.arange(42).reshape((3, 2, 7))
>>> contexts
array(
    [
        [
            [ 0,  1,  2,  3,  4,  5,  6],
            [ 7,  8,  9, 10, 11, 12, 13]
        ],
        [
            [14, 15, 16, 17, 18, 19, 20],
            [21, 22, 23, 24, 25, 26, 27]
        ],
        [
            [28, 29, 30, 31, 32, 33, 34],
            [35, 36, 37, 38, 39, 40, 41]
        ]
    ]
)
>>> contexts.shape
(3, 2, 7)
>>> contexts[:, 0]
array(
    [
        [ 0,  1,  2,  3,  4,  5,  6],
        [14, 15, 16, 17, 18, 19, 20],
        [28, 29, 30, 31, 32, 33, 34]
    ]
)
>>> contexts[:, 0].shape
(3, 7)
>>> contexts[:, 1]
array(
    [
        [ 7,  8,  9, 10, 11, 12, 13],
        [21, 22, 23, 24, 25, 26, 27],
        [35, 36, 37, 38, 39, 40, 41]
    ]
)
>>> contexts[:, 1].shape
(3, 7)
>>>
'''

from __future__ import print_function, division
import numpy as np

def get_padding(padding_type, kernel_size):
    assert padding_type in ['SAME', 'VALID']
    if padding_type == 'SAME':
        return tuple((k - 1) // 2 for k in kernel_size)
    return tuple(0 for _ in kernel_size)


def sigmoid_schedule(t,k=5):
    t0 = t/2
    k = k/t0
    t = np.arange(t)
    return (1/(1+np.exp(-k*(t-t0)))).astype(np.float32)

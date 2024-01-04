"""
We check the numpy's advanced indexing using ranges etc
"""
import numpy as np
from benchmarker import benchmark

@benchmark(times=100)
def using_range(x, r):
    return x[range(r)]

@benchmark(times=100)
def using_list(x, r):
    return x[list(range(r))]

@benchmark(times=100)
def using_arange(x, r):
    return x[np.arange(r)]


if __name__=='__main__':
    N = 1000000
    R = 900000
    vec = np.random.choice(100, N)
    using_range(vec,R)
    using_list(vec,R)
    using_arange(vec,R)

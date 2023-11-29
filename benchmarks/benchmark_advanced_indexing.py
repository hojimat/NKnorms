'''
We check the numpy's advanced indexing using ranges etc
'''
import numpy as np
import time
from functools import wraps

def benchmark(func, times=100):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Running {func.__name__}()...")
        start_time = time.perf_counter()
        for i in range(times):
            func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Finished in {total_time:.4f} seconds.")
    return wrapper

@benchmark
def using_range(x, r):
    return x[range(r)]

@benchmark
def using_list(x, r):
    return x[list(range(r))]

@benchmark
def using_arange(x, r):
    return x[np.arange(r)]


if __name__=='__main__':
    N = 1000000
    R = 900000
    vec = np.random.choice(100, N)
    using_range(vec,R)
    using_list(vec,R)
    using_arange(vec,R)

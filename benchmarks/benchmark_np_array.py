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
def converter(list_):
    return np.array(list_)

@benchmark
def directer(n):
    return np.arange(n)

if __name__=='__main__':
    N = 1000000
    vec = list(range(N))
    converter(vec)
    directer(N)
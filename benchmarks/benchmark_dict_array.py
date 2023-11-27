import sys
sys.path.append("/home/ravshan/Seafile/research/codes/NKnorms/")
import nkpack as nk
import numpy as np
from numpy.typing import NDArray
import time
from functools import wraps

def benchmark(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Running {func.__name__}()...")
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Finished in {total_time:.4f} seconds.")
        return result
    return wrapper

p = 1
n = 30
k = 20
c = 0
s = 0
rho = 0.0
imat = nk.interaction_matrix(n, k)
landscape = nk.generate_landscape(p, n, k, c, s, rho)


@benchmark
def using_array():
    target = np.random.choice(2,k)
    target_index = nk.bin2dec(target)
    value = landscape[target_index]
    print(value)

if __name__=='__main__':   
    using_array()
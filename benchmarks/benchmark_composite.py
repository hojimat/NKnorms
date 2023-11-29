import numpy as np
import itertools
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

        result = func(*args, **kwargs)
        #print("Result:")
        print(result)
        return result
    return wrapper

@benchmark
def using_itertools(proposals_, p, prop, n, comp):
    # randomly pick combination indices
    all_indices = itertools.product(range(prop), repeat=p)
    random_picks = np.random.choice(prop**p, comp)
    picked_indices = [indices for i,indices in enumerate(all_indices) if i in random_picks]

    # composite:
    composites = [proposals_[np.arange(p), idx, :].reshape(-1) for idx in picked_indices]
    return np.array(composites)

@benchmark
def using_unravel(proposals_, p, prop, n, comp):
    random_picks = np.random.choice(prop**p, comp)
    picked_indices = [np.unravel_index(i, [prop]*p) for i in random_picks]

    composites = [proposals_[np.arange(p), idx, :].reshape(-1) for idx in picked_indices]
    return np.array(composites)



if __name__=='__main__':
    P = 5
    PROP = 3
    N = 4
    COMP = 6
    PROPS = np.array([np.random.choice(2, (PROP, N))]*P)

    using_itertools(PROPS, P, PROP, N, COMP)
    using_unravel(PROPS, P, PROP, N, COMP)
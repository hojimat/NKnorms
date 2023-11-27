import sys
sys.path.append("/home/ravshan/Seafile/research/codes/NKnorms/")
import nkpack as nk
import numpy as np
from numpy.typing import NDArray
import time
from functools import wraps
from numba import njit

def benchmark(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(f"Running {func.__name__}()...")
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Finished in {total_time:.4f} seconds. Answer={result:.5f}")
        return result
    return wrapper

p = 1
n = 16
k = 8
c = 0
s = 0
rho = 0.0
imat = nk.interaction_matrix(n, k)
landscape = nk.generate_landscape(p, n, k, c, s, rho)
    
@njit
def calculate_performances(bstring: NDArray[np.int8], imat: NDArray[np.int8], cmat: NDArray[np.float32], n: int, p: int) -> NDArray[np.float32]:
    phi = np.zeros(n*p)
    for i in range(n*p):
        coupled_bits = bstring[np.where(imat[:,i]>0)]
        bin_to_dec = sum(coupled_bits * 2**(np.arange(len(coupled_bits))[::-1]))
        phi[i] = cmat[bin_to_dec, i] 
    #Phi = phi.reshape(-1,p).mean(axis=1)
    Phis = np.zeros(p, dtype=np.float32)
    for i in range(p):
        Phis[i] = phi[n*i : n*(i+1)].mean()
    return Phis

@benchmark
def using_for_loop():
    max_performance = 0.0

    for i in range(2 ** (n*p) ):
        dec_to_bin = ( (i // 2**np.arange(n*p)[::-1]) % 2 ).astype(np.int8)
        phis = calculate_performances(dec_to_bin, imat, landscape, n, p)

        if sum(phis) > max_performance:
            max_performance = sum(phis)

    return max_performance

@benchmark
def using_for_loop_with_inner_function():
    def calc_perf(num):
        dec_to_bin = ( (num // 2**np.arange(n*p)[::-1]) % 2 ).astype(np.int8)
        phis = calculate_performances(dec_to_bin, imat, landscape, n, p)
        return np.sum(phis)

    max_performance = 0.0
    for i in range(2 ** (n*p) ):
        phi = calc_perf(i)
        if phi > max_performance:
            max_performance = phi

    return max_performance

@benchmark
def using_list_comprehension_with_inner_function():
    def calc_perf(num):
        dec_to_bin = ( (num // 2**np.arange(n*p)[::-1]) % 2 ).astype(np.int8)
        phis = calculate_performances(dec_to_bin, imat, landscape, n, p)
        return np.sum(phis)

    phis = [calc_perf(i) for i in range(2 ** (n*p) )]

    return max(phis)

@benchmark
@njit
def using_njit():
    max_performance = 0.0

    for i in range(2 ** (n*p) ):
        dec_to_bin = ( (i // 2**np.arange(n*p)[::-1]) % 2 ).astype(np.int8)
        phis = calculate_performances(dec_to_bin, imat, landscape, n, p)

        if sum(phis) > max_performance:
            max_performance = sum(phis)

    return max_performance


@benchmark
@njit
def using_njit_list_comprehension():
    def calc_perf(num):
        dec_to_bin = ( (num // 2**np.arange(n*p)[::-1]) % 2 ).astype(np.int8)
        phis = calculate_performances(dec_to_bin, imat, landscape, n, p)
        return np.sum(phis)

    phis = [calc_perf(i) for i in range(2 ** (n*p) )]

    return max(phis)


if __name__=='__main__':   
    #using_for_loop() # 9.229 sec
    #using_for_loop_with_inner_function() # 8.9965
    #using_list_comprehension_with_inner_function() # 8.8082
    using_njit() # 1.9
    #using_njit_list_comprehension() # 2.2
import sys
sys.path.append("/home/ravshan/Seafile/research/codes/NKnorms/")
import numpy as np
import nkpack as nk
from benchmarker import benchmark


@benchmark(times=1000000, verbose=True)
def using_loop(bstring, n, p, nsim):
    tmp = np.reshape(bstring, (p,n))[:,:nsim]
    sum_ = 0
    for i in range(p):
        for j in range(i,p):
            sum_ += nk.hamming_distance(tmp[i,:],tmp[j,:])
    
    return sum_

@benchmark(times=1000000, verbose=True)
def using_3d(bstring, n_, p_, nsim):
    tmp = np.reshape(bstring, (p_,n_))[:,:nsim]

    tmp1 = tmp[np.newaxis, :, :]
    tmp2 = tmp[:, np.newaxis, :]
    
    sum_ = np.sum(tmp1 != tmp2)/2
    
    return sum_



if __name__=='__main__':
    N = 4
    P = 5
    NSIM = 4
    BSTR = np.random.choice(2, N*P)
    
    using_loop(BSTR, N, P, NSIM)
    using_3d(BSTR, N, P, NSIM)
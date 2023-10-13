import numpy as np
import itertools as itool
from scipy.stats import beta
import progressbar
import matplotlib.pyplot as plt

###############################################################################
###############################################################################
###############################################################################
###############################################################################

def interaction_matrix(N,K,type="roll"):
    output = None
    if K == 0:
        output = np.eye(N,dtype=int)
    elif type=="diag":
        tmp = [np.diag(np.ones(N-abs(z)),z) for z in range((-K),(K+1))]
        tmp = np.array(tmp)
        output = np.sum(tmp,0)
    elif type=="updiag":
        tmp = [np.diag(np.ones(N-abs(z)),z) for z in range(0,(K+1))]
        tmp = np.array(tmp)
        output = np.sum(tmp,0)
    elif type=="downdiag":
        tmp = [np.diag(np.ones(N-abs(z)),z) for z in range((-K),1)]
        tmp = np.array(tmp)
        output = np.sum(tmp,0)
    elif type=="sqdiag":
        tmp = np.eye(N,dtype=int)
        tmp = tmp.repeat(K+1,axis=0)
        tmp = tmp.repeat(K+1,axis=1)
        output = tmp[0:N,0:N]
    elif type=="roll":
        tmp = [1]*K + [0]*(N-K)
        tmp = [np.roll(tmp,z) for z in range(N)]
        tmp = np.array(tmp)
        output = tmp.transpose()
    elif type=="chess":
        print(f"Uncrecognized interaction type '{type}'")
    # print(f"Interaction type '{type}' selected")
    return output

###############################################################################
###############################################################################
###############################################################################
###############################################################################

def contrib_define(N,K):
    # np.random.seed(123)
    output = np.random.uniform(0,1,(2**(K+1),N))
    # print(f"Contribution space N={N},K={K} successfully defined")
    return output

def contrib_solve(x,imat,cmat):
    n = np.size(imat,axis=0)
    k = np.sum(imat,axis=0)[0]
    phi = [0]*n

    for i in range(n):
        tmp = x[np.where(imat[:,i]>0)]
        if i+k+1 > n:
            tmp = np.roll(tmp,n-i-k-1)
        tmp_loc = binx(tmp)
        phi[i] = cmat[tmp_loc,i]
    #capital_phi = phi.mean()
    return phi

def binx(x,size=4):
    tmp = x
    if type(tmp) is list:
        #tmp = np.packbits(tmp)[0]
        tmp = str(tmp).replace(", ","").replace("[","").replace("]","")
        tmp = int(tmp,2)
    elif type(tmp) is int:
        tmp = np.binary_repr(tmp,size)
        tmp = [int(z) for z in tmp]
    elif type(tmp) is str:
        tmp = int(tmp,2)
    elif type(tmp) is np.ndarray:
        tmp = str(tmp).replace(" ","").replace("[","").replace("]","")
        tmp = int(tmp,2)
    else:
        print("incorrect input for function binx")
    return tmp

def perf_calc(imat,cmat):
    n = np.size(imat,axis=0)
    perfmat = np.zeros(2**n,dtype=float)
    for i in range(2**n):
        bstring = np.array(binx(i,n))
        bval = np.mean(contrib_solve(bstring,imat,cmat))
        perfmat[i] = bval
    perfmax = np.max(perfmat)
    return perfmat, perfmax

###############################################################################
###############################################################################
###############################################################################
###############################################################################

def assign_tasks(N,POP,type="solo"):
    output = None
    if type=="solo":
        perCapita = N / POP
        tmp = np.eye(POP,dtype=int)
        tmp = tmp.repeat(perCapita,axis=1)
        output = tmp
    else:
        print("Task assignment type unrecognized")
    # print(f"Assignment type {type} selected")
    return output

###############################################################################
###############################################################################
###############################################################################
###############################################################################

def local_climb(x,pmat):
    output = x
    n = len(x)
    contribb = None
    oldidx = binx(x)

    contrib0 = pmat[oldidx]

    randx = np.random.choice(n)
    tmp_state = x.copy()
    tmp_state[randx] = 1 - tmp_state[randx]
    tmp_idx = binx(tmp_state)

    contrib1 = pmat[tmp_idx]

    if contrib1 > contrib0:
        output = tmp_state
        contribb = contrib1
        #print("Climb up")
    else:
        contribb = contrib0
        #print("Stay")

    return output, contribb

###############################################################################
###############################################################################
###############################################################################
###############################################################################

def reinforcement_climb(x,beat,nay,pmat):
    output = x
    n = len(x)
    contribb = None
    bbeta = beat
    oldidx = binx(x)

    contrib0 = pmat[oldidx]
    neigh = get_neighbours(x,nay)[1]
    contribt = [pmat[z] for z in neigh]
    contrib1 = max(contribt)
    maxidx = neigh[contribt.index(contrib1)]
    maxbit = binx(maxidx,n)

    force0 = beta.rvs(*bbeta[oldidx,:])
    force1 = beta.rvs(*bbeta[maxidx,:])
    #
    # if (contrib1+force1>=contrib0+force0):
    #     bbeta[maxidx][0] += 1
    #     output = maxbit
    #     contribb = contrib1
    #     #print("Climb up")
    # else:
    #     bbeta[maxidx][1] += 1
    #     contribb = contrib0
    #     #print("Stay")


    # force0 = bbeta[oldidx,0] / (bbeta[oldidx,0] + bbeta[oldidx,1])
    # force1 = bbeta[maxidx,0] / (bbeta[maxidx,0] + bbeta[maxidx,1])
    if (contrib1+force1>=contrib0+force0):
        output = maxbit
        contribb = contrib1
    else:
        contribb = contrib0

    if contrib1 >= contrib0:
        bbeta[maxidx][0] += 1
    else:
        bbeta[maxidx][1] += 1

    return output, contribb, bbeta

###############################################################################
###############################################################################
###############################################################################
###############################################################################

def generate_network(POP,S=2,type="cycle"):
    output = None
    if S == 0:
        output = np.zeros(N)
    elif type == "cycle":
        tmp = np.eye(POP,dtype=int)
        tmp0 = tmp[0,:]
        tmp1 = tmp[1:,:]
        tmp = np.vstack([tmp1,tmp0])
        output = tmp
    elif type == "complete":
        tmp = 1 - np.eye(POP,dtype=int)
        output = tmp
    elif type == "random":
        tmp = np.zeros((POP,POP),dtype=int)
        for i in range(POP):
            tmpi = [0]*(POP-S-1) + [1]*S
            np.random.shuffle(tmpi)
            tmpi.insert(i,0)
            tmp[i,:] = tmpi
        output = tmp
    elif type == "fullrandom":
        tmp = np.random.choice(2,(POP,POP))
        np.fill_diagonal(tmp,0)
        ourtput = tmp
    else:
        print(f"Unrecognized network type '{type}'")
    return(output)

###############################################################################
###############################################################################
###############################################################################
###############################################################################

def get_neighbours(vec,count):
    tmpv = []
    tmpi = []
    # np.random.seed(234)
    subbset = np.random.choice(np.arange(len(vec)),count,replace=False)

    for i in subbset:
        y = vec.copy()
        y[i] = 1 - y[i]
        tmpv.append(y)
        tmpi.append(binx(y))


    return(tmpv, tmpi)

###############################################################################
###############################################################################
###############################################################################
###############################################################################

"""
`nkpack` is library for agent-based modeling using NK framework.
It provides tools to handle binary variables, generate landscapes, find maxima.
The library heavily uses `numpy` for its data types, and is annotated with type hints.
The library also uses `numba` to speed up global maximum search.

"""

import itertools
import typing
import numpy as np
from numpy.typing import NDArray
from numba import njit
from scipy.stats import norm

###############################################################################

def interaction_matrix(N,K,shape="roll"):
    """Creates an interaction matrix for a given K

    Args:
        N (int): Number of bits
        K (int): Level of interactions
        shape (str): Shape of interactions. Takes values 'roll' (default), 'random', 'diag'.

    Returns:
        numpy.ndarray: an NxN numpy array with diagonal values equal to 1 and rowSums=colSums=K+1
    """

    output = None
    if K == 0:
        output = np.eye(N,dtype=int)
    elif shape=="diag":
        tmp = [np.diag(np.ones(N-abs(z)),z) for z in range((-K),(K+1))]
        tmp = np.array(tmp)
        output = np.sum(tmp,0)
    elif shape=="updiag":
        tmp = [np.diag(np.ones(N-abs(z)),z) for z in range(0,(K+1))]
        tmp = np.array(tmp)
        output = np.sum(tmp,0)
    elif shape=="downdiag":
        tmp = [np.diag(np.ones(N-abs(z)),z) for z in range((-K),1)]
        tmp = np.array(tmp)
        output = np.sum(tmp,0)
    elif shape=="sqdiag":
        tmp = np.eye(N,dtype=int)
        tmp = tmp.repeat(K+1,axis=0)
        tmp = tmp.repeat(K+1,axis=1)
        output = tmp[0:N,0:N]
    elif shape=="roll":
        tmp = [1]*(K+1) + [0]*(N-K-1)
        tmp = [np.roll(tmp,z) for z in range(N)]
        tmp = np.array(tmp)
        output = tmp.transpose()
    elif shape=="random":
        output = random_binary_matrix(N,K+1,1)
    elif shape=="chess":
        print(f"Uncrecognized interaction type '{type}'")
    # print(f"Interaction shape '{shape}' selected")
    return output

def binary_combinations(N,R):
    """Generates all binary vectors with sum R

    Args:
        N (int): Size of a binary vector
        R (int): Sum of elements of a vector

    Returns:
        numpy.ndarray: an (N R)xN numpy array, where
        rows are vectors of size N and sum of elements R
    """

    tmp = []
    idx = itertools.combinations(range(N),R)
    for i in idx:
        A = [0]*N
        for j in i:
            A[j] = 1
        tmp.append(A)
    output = np.reshape(tmp,(-1,N))
    return(output)
    
def random_binary_matrix(N,R,diag=None):
    """Generates a random binary square matrix with a given row/col sum

    Args:
        N (int): Number of rows/cols
        R (int): Sum of rows/cols
        diag (int or None): Fixed value for diagonal. Takes values None (default), 0, 1.

    Returns:
        numpy.ndarray: an NxN numpy array
    """

    if N==R:
        tmp = np.ones((N,R),dtype=int)
        return tmp
    elif N<R:
        print("Incorrect binary matrix. Check parameters.")
        return
    # create a minimal 2d matrix of zeros for easier indexing
    tmp = np.zeros(2*N,dtype=int).reshape(2,N) 
    cl = binary_combinations(N,R)
    for i in range(N):
        colsums = np.sum(tmp,0)
        # remove excess ones
        idx = np.empty(0,dtype=int)
        for j in np.where(colsums>=R)[0]:
            k = np.where(cl[:,j]>0)[0]
            idx = np.union1d(idx,k)
        cl = np.delete(cl,idx,0)
        # remove excess zeros
        inx = np.empty(0,dtype=int)
        for j in np.where(colsums+N-i == R)[0]:
            k = np.where(cl[:,j]==0)[0]
            inx = np.union1d(inx,k)
        cl = np.delete(cl,inx,0)
        # temporarily ignore diagonal 1s or 0s 
        cli = cl.copy()
        if diag is not None:
            ivx = np.where(cl[:,i]==diag)[0]
            cli = cl[ivx,]
            clk = (cli + colsums)[:,i+1:]
            tp = N-i-1 if diag==0 else 0 # tuning parameter
            ikx = np.where(clk+tp==R)[0]
            cli = np.delete(cli,ikx,0)

        ncl = cl.shape[0]
        ncli = cli.shape[0]
        if ncli > 0:
            ind = np.random.choice(ncli)
            tmp = np.vstack((tmp,cli[ind,:]))
        elif ncli==0 and ncl>0:
            print('Error creating non-zero diagonals. Rerun the function')
            return 0
        else:
            print('Incorrect binary matrix. Check the dimensions.')
            return 
    output = np.delete(tmp,[0,1],0) # remove first 2 empty rows created above
    return(output)

###############################################################################

@njit
def dec2bin(x: int, sz: int) -> NDArray[np.int8]:
    """Converts decimal integer to binary array
    
    Args:
        x: An input decimal integer
        sz: The desired minimum size of the output

    Returns:
        A numpy array with leading zeros to match size of sz
    """
    output = []

    while x > 0:
        output.insert(0,x%2)
        x = int(x/2)

    if len(output)<sz:
        output = [0]*(sz-len(output)) + output

    return np.array(output, dtype=np.int8)

@njit
def bin2dec(x: NDArray) -> int:
    """Converts binary list to integer
    
    Args:
        x: An input vector with binary values

    Returns:
        A decimal integer equivalent of the binary input
    """
    return sum(x * 2**(np.arange(len(x))[::-1]))

def generate_landscape(p: int, n: int, k: int, c: int, s: int, rho: float) -> NDArray[np.float32] :
    """Defines a matrix of performance contributions for given parameters.
    This is a matrix with N*P columns each corresponding to a bit,
    and 2**(1+K+C*S) rows each corresponding to a possible bitstring.

    Args:
        p: Number of landscapes (population size)
        n: Number of tasks per landscape
        k: Number of internal bits interacting with each bit
        c: Number of external bits interacting with each bit
        s: Number of landscapes considered for external bits
        rho: Correlation coefficient between landscapes

    Returns:
        An (N*P)x2**(1+K+C*S) matrix of P contribution matrices with correlation rho
    """

    corrmat = np.repeat(rho,p*p).reshape(p,p) + (1-rho) * np.eye(p)
    corrmat = 2*np.sin((np.pi / 6 ) * corrmat)
    tmp = np.random.multivariate_normal(mean=[0]*p, cov=corrmat, size=(n*2**(1+k+c*s)))
    tmp = norm.cdf(tmp)
    tmp = np.reshape(tmp.T, (n*p, (2**(1+k+c*s)))).T
    output = tmp
    return output

@njit
def calculate_performances(bstring: NDArray[np.int8], imat: NDArray[np.int8], cmat: NDArray[np.float32], n: int, p: int) -> NDArray[np.float32]:
    """Computes a performance of a bitstring given contribution matrix (landscape) and interaction matrix

    Notes:
        Uses Numba's njit compiler.

    Args:
        x : An input vector
        imat: Interaction matrix
        cmat: Contribution matrix (landscape)
        n: Number of tasks per landscape
        p: Number of landscapes (population size)

    Returns:
        A list of mean performances of an input vector X for P agents, given cmat and imat.
    """
    # get performance contributions for every bit:
    phi = np.zeros(n*p)
    for i in range(n*p):
        # subset only coupled bits, i.e. where
        # interaction matrix is not zero:
        coupled_bits = bstring[np.where(imat[:,i]>0)]
        phi[i] = cmat[bin2dec(coupled_bits), i] # vector of size N*P

    # get agents' performances by averaging their bits'
    # performances, thus getting vector of P mean performances
    # make rows of size P, and take mean of each row
    output = phi.reshape(-1,p).mean(axis=1)

    return output


@njit
def get_globalmax(imat, cmat, n, p) -> float:
    """Calculates global maximum"""
    nxp = n*p
    max_performance = np.zeros(p, dtype=float)
    for i in range(2**nxp):
        bval = calculate_performances(dec2bin(i, nxp), imat, cmat, n, p)
        if sum(bval) > sum(max_performance):
            max_performance = bval

    return np.mean(max_performance, dtype=float)

###############################################################################

def assign_tasks(N,POP,shape="solo"):
    """Assigns N tasks to POP agents

    Args:
        N (int): Number of tasks 
        POP (int): Number of agents (population size)
        shape (str): Type of allocation. Takes values 'solo' (default) and 'overlap' (not used at the moment)

    Returns:
        numpy.ndarray: Returns a POPxN matrix where rows represent agents and cols represent tasks
    """

    output = None
    if shape=="solo":
        perCapita = N / POP
        tmp = np.eye(POP,dtype=int)
        tmp = tmp.repeat(perCapita,axis=1)
        output = tmp
    else:
        print("Task assignment shape unrecognized")
    # print(f"Assignment shape {shape} selected")
    return output

###############################################################################

def generate_network(POP,S=2,pcom=1.0,shape="random",absval=False):
    """Generates a unidirectional network topology

    Args:
        POP (int): Number of agents (population size)
        S (int): Network degree
        pcom (float): Probability of communicating through the channel
        shape (str): Network topology. Takes values 'random' (default), 'ring', 'cycle', 'line', 'star'
        absval (bool): Indexing convention. If True, converts negative indices to positive.

    Returns:
        numpy.ndarray: A POPxPOP matrix with probabilities of connecting to other agents
    """

    if S>=POP:
        print("Error: wrong network degree")
        return 0
    if pcom>1 or pcom<0:
        print("Error: wrong probability of communication")
        return 0
    output = None
    if S == 0:
        output = []
    elif shape=="cycle":
        tmp = np.eye(POP)
        tmp = np.vstack((tmp[1:,:],tmp[0,:]))
        output = tmp * pcom
    elif shape == "line":
        tmp = np.eye(POP)
        tmp = np.vstack((tmp[1:,:],np.zeros(POP)))
        output = tmp * pcom
    elif shape == "random":
        tmp = random_binary_matrix(POP,S,0)
        output = tmp * pcom
    elif shape == "star":
        tmp = np.zeros((POP,POP))
        tmp[0,:] = 1
        tmp[0,0] = 0
        output = tmp * pcom
    elif shape == "ring":
        tmp = np.eye(POP)
        tmpA = np.vstack((tmp[1:,:],tmp[0,:]))
        tmpB = np.vstack((tmp[-1:,:],tmp[:-1,:]))
        tmp = tmpA + tmpB
        output = tmp * pcom
    else:
        print(f"Unrecognized network shape '{shape}'")

    return(output)

def generate_couples(POP,S=2,shape="cycle"):
    """Generates couplings between landscapes (external interaction) 

    Args:
        POP (int): Number of landscapes (population size)
        S (int): Number of landscapes considered for external bits
        shape (str): A network topology. Takes values 'cycle' (default)

    Returns:
        list: A list of S-sized vectors with couples for every landscape.
    """

    if S>=POP:
        print("Error: wrong network degree")
        return 0
    output = None
    if S == 0:
        output = []
    elif shape=="cycle":
        tmp = [[(z-1)%POP] + [_%POP for _ in range(z+1,z+S)] for z in range(POP)]
        output = tmp
    else:
        print(f"Unrecognized network shape '{shape}'")
    return(output)
###############################################################################

def random_neighbour(vec,myid,n):
    """Generates a random binary vector that is 1-bit away (a unit Hamming distance)

    Args:
        vec (list or numpy.ndarray): An input vector
        myid (int): An id of an agent of interest
        n (int): Number of tasks allocated to a single agent

    Returns:
        list: A vector with one bit flipped for agent myid
    """

    rnd = np.random.choice(range(myid*n,(myid+1)*n))
    vec[rnd] = 1- vec[rnd]
    output = vec
    return output

@njit
def get_index(vec,myid,n):
    """Gets a decimal equivalent for the bitstring for agent myid

    Args:
        vec (list or numpy.ndarray): An input vector
        myid (int): An id of an agent of interest
        n (int): Number of tasks allocated to a single agent

    Returns:
        int: A decimal value of vec for myid
    """

    output = bin2dec(vec[myid*n:(myid+1)*n])
    return output

def with_noise(vec,pb=0):
    """Adds a noise to a vector

    Args:
        vec (list or numpy.ndarray): An input vector
        pb (float): Probability of error/noise. Takes values between 0 (default) and 1.

    Returns:
        numpy.ndarray: A vector of same size as vec but with errors
    """

    if vec.size==0:
        return vec
    tmp = vec
    nvec = len(vec)
    noise = np.random.choice(2,p=[1-pb, pb])
    if noise:
        tmp = random_neighbour(vec,0,nvec)
    output = tmp
    return output

def hamming_distance(x,y):
    """Calculates the Hamming distance (count of different bits) between two bitstrings
    
    Args:
        x (list): first list
        y (list): second list

    Returns:
        int: An integer value

    """
    tmp = np.sum(np.abs(np.array(x) - np.array(y)))
    output = tmp
    return output
###############################################################################

def beta_mean(x,y):
    """Calculates the mean of the Beta(x,y) distribution

    Args:
        x (float): alpha (shape) parameter
        y (float): beta (scale) parameter

    Returns:
        float: A float that returns x/(x+y)
    """
    
    output = x / (x+y)
    return output

def flatten(x):
    """Converts a list of lists into a flat list

    Args:
        x (list): An input list

    Returns:
        list: A flat list
    """
    
    output = [_ for z in x for _ in z]
    return output


def calculate_freq(x,vec):
    """Calclates frequency of vec in x

    Args:
        x (numpy.ndarray): A 2d matrix
        vec (numpy.ndarray): A 1d array

    Returns:
        float: Frequency of vec in x
    """

    if x is None or vec.size==0 or vec[0,0]==-1:
        return 0.0
    freq = np.mean(vec,axis=0)
    tmp1 = np.multiply(x,freq)
    tmp2 = np.multiply(1-x,1-freq)
    tmp = tmp1 + tmp2
    output = np.mean(tmp)
    return output

def extract_soc(x,myid,n,nsoc):
    """Extracts social bits from a bitstring

    Args:
        x (numpy.ndarray): An input vector
        myid (int): An id of an agent of interest
        n (int): Number of tasks allocated to a single agent
        nsoc (int): Number of social tasks (exogeneous)

    Returns:
        numpy.ndarray: A vector of size nsoc
    """

    tmp = x[(myid+1)*n-nsoc:(myid+1)*n]
    #tmp = str(tmp).replace(" ","").replace("[","").replace("]","")
    output = tmp
    return output

###############################################################################

def pick(vec,count):
    """Picks random elements form a vector

    Args:
        vec (numpy.ndarray): An input vector
        count (int): Number of elements to pick

    Returns:
        numpy.ndarray: A vector of size count
    """

    tmp = np.random.choice(range(vec),count,replace=False)
    output = np.sort(tmp)
    return output

###############################################################################

def cobb_douglas(weights,vec):
    """A Cobb-Douglas utility function with given weights

    Args:
        weights (list): Weights
        vec (numpy.ndarray): An input vector

    Returns:
        float: A Cobb-Douglas value
    """

    x = [z+1 for z in vec]
    w = weights
    tmp = np.power(x,w).prod()
    output = tmp
    return output

def satisfice(x,y,u):
    """A satisficing utility

    Args:
        x (float): first value
        y (float): second value
        u (float): first goal

    Returns:
        float: Utility value
    """

    cond = x >= u
    tmp = cond * (y + u) + (1-cond) * x
    output = tmp
    return output

def weighted(x,y,p1,p2):
    """A weighted sum

    Args:
        x (float): first value
        y (float): second value
        p1 (float): first weight
        p2 (float): second weight

    Returns:
        float: A weighted sum
    """

    tmp = p1*x + p2*y
    output = tmp
    return output

def goal_prog(perf1,perf2,u,p1,p2):
    """A goal programming

    Args:
        perf1 (float): first performance
        perf2 (float): second performance
        u (list): goals
        p1 (float): first weigh
        p2 (float): second weight

    Returns:
        float: A GP output
    """

    d1 = np.max((u[0]-perf1,0))
    d2 = np.max((u[1]-perf2,0))
    tmp = p1*d1 + p2*d2
    output = -tmp
    return output

def similarity(x,p,n,nsoc):
    """Calculates the similarity measure of the bitstring

    Args:
        x (list): the bitstring of interest
        p (int): number of agents
        n (int): number of tasks per agent
        nsoc (int): number of imitated tasks per agent

    Returns:
        float: The float between 0 and 1
    """
    if p<2:
        print('Need at least 2 agents for similarity measure')
        return
    if nsoc<1:
        print('Please enter non-zero number of social bits')
        return

    tmp = np.reshape(x, (p,n))[:,(n-nsoc):]
    summ = 0
    for i in range(p):
        for j in range(i,p):
            summ += hamming_distance(tmp[i,:],tmp[j,:])

    max_summ = nsoc*(p/2)**2 if p%2==0 else nsoc*((p-1)/2)**2 + (p-1)*(nsoc/2)
    output = 1-summ/max_summ
    return output

def similarbits(x,p,n,nsoc):
    """Calculates the similarity measure of the bitstring

    Args:
        x (list): the bitstring of interest
        p (int): number of agents
        n (int): number of tasks per agent
        nsoc (int): number of imitated tasks per agent

    Returns:
        float: The float between 0 and 1
    """
    if nsoc>n:
        print('Number of social bits exceeds total number of bits')
    tmp = np.reshape(x, (p,n))
    tmp = np.sum(np.array(tmp), axis=0)[(n-nsoc):]
    tmp = np.max((tmp/p, 1-tmp/p), axis=0)
    output = np.mean(tmp)
    return output


def variate(param_dict: typing.Dict[str, typing.List]) -> typing.Iterator[typing.Dict]:
    """Creates parameter combinations

    Args:
        param_dict (dict): dictionary of parameters

    Returns:
        list: List of dicts
    """
    param_keys = param_dict.keys()
    value_combinations = itertools.product(*param_dict.values())
    output = (dict(zip(param_keys, i)) for i in value_combinations)
    return output

def update_values(A, m, i, j, Val):
    """Updates the square subset of the matrix in place

    Args:
        A (numpy.ndarray): a matrix to be updated
        m (int): size of a square matrix
        i (int): i\'th row-wise
        j (int): j\'th column-wise
        Val (numpy.ndarray): values set to the subset
    """
    A[i*m : (i+1)*m, j*m : (j+1)*m] = Val

class InvalidParameterError(Exception):
    pass

class InvalidBitstringError(Exception):
    pass

import numpy as np
from numpy.typing import NDArray
from numba import njit
from scipy.stats import norm


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


###############################################################################


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
    base_matrix = np.random.multivariate_normal(mean=[0]*p, cov=corrmat, size=(n*2**(1+k+c*s)))
    cdf_matrix = norm.cdf(base_matrix)
    landscape = np.reshape(cdf_matrix.T, (n*p, (2**(1+k+c*s)))).T
    
    return landscape


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
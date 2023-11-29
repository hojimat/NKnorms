import numpy as np
from numpy.typing import NDArray

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

def calculate_frequency(bstring: NDArray[np.int8], lookup_table: NDArray[np.int8]) -> float:
    """
    Calclates frequency of a bitstring in a (pre-flattened) lookup table of bistrings.

    Args:
        bstring: 1xNSOC sized array
        lookup_table: (TM*DEG)xNSOC sized array of bstrings

    Returns:
        Frequency of bstring in lookup_table

    Example:
        bstring=np.array([1,1])
        lookup_table = np.array([
            [1,1],
            [0,0],
            [0,0]
        ])

        should return 1/3
    """
    
    return np.mean(lookup_table==bstring)

def beta_mean(x,y):
    """Calculates the mean of the Beta(x,y) distribution

    Args:
        x (float): alpha (shape) parameter
        y (float): beta (scale) parameter

    Returns:
        float: A float that returns x/(x+y)
    """
    
    return x / (x+y)


def decompose_performances(performances: NDArray[np.float32], agent_id: int) \
    -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Takes individual performances for multiple bit strings
    and returns own performance and mean of other agents'
    performances

    Args:
        performances: ALTxP matrix of floats
        agent_id
    Returns:
        ALTx1 array of own performances and
        ALTx1 array of mean of others' performances

    """

    perf_own = performances[:, agent_id]
    perf_other = (np.sum(performances, axis=1) - perf_own) / (performances.shape[1] - 1)

    return perf_own, perf_other
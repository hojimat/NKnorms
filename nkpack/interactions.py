
from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from .exceptions import *
from .bitstrings import random_binary_matrix


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


def generate_network(pop: int, s: int = 2, pcom: float = 1.0, shape: str = "random", absval: bool = False) -> NDArray[np.float32]:
    """Generates a unidirectional network topology

    Args:
        pop : Number of agents (population size)
        s : Network degree
        pcom : Probability of communicating through the channel
        shape : Network topology. Takes values 'random' (default), 'ring', 'cycle', 'line', 'star'
        absval: Indexing convention. If True, converts negative indices to positive.

    Returns:
        A pop x pop matrix with probabilities of connecting to other agents
    """

    if s>=pop:
        raise InvalidParameterError("Network degree exceeds the total number of agents.")
 
    if pcom>1 or pcom<0:
        raise InvalidParameterError("Probability of communication must be between 0 and 1.")

    output = None
    if s == 0:
        output = np.array([])
    elif shape=="cycle":
        tmp = np.eye(pop)
        tmp = np.vstack((tmp[1:,:],tmp[0,:]))
        output = tmp * pcom
    elif shape == "line":
        tmp = np.eye(pop)
        tmp = np.vstack((tmp[1:,:],np.zeros(pop)))
        output = tmp * pcom
    elif shape == "random":
        tmp = random_binary_matrix(pop,s,0)
        output = tmp * pcom
    elif shape == "star":
        tmp = np.zeros((pop,pop))
        tmp[0,1:] = 1
        output = tmp * pcom
    elif shape == "ring":
        tmp = np.eye(pop)
        tmpA = np.vstack((tmp[1:,:],tmp[0,:]))
        tmpB = np.vstack((tmp[-1:,:],tmp[:-1,:]))
        
        output = (tmpA + tmpB) * pcom
    else:
        raise InvalidParameterError(f"Unrecognized network shape '{shape}'")

    return output

def generate_couples(pop:int, s:int = 2, shape:str = "random") -> list:
    """Generates couplings between landscapes (external interaction) 

    Args:
        pop: Number of landscapes (population size)
        s: Number of landscapes considered for external bits
        shape: A network topology. Takes values 'cycle' (default)

    Returns:
        A list of S-sized vectors with couples for every landscape.
    """

    if s >= pop:
        raise InvalidParameterError("Number of coupled species cannot exceed the population size")

    output = None
    if s == 0:
        output = []
    elif shape=="cycle":
        output = [[(z-1) % pop] + [i % pop for i in range(z+1,z+s)] for z in range(pop)]
    elif shape=="random":
        arr = random_binary_matrix(pop, s, 0)
        output = [np.where(arr[:, col])[0].tolist() for col in range(pop)]
    else:
        raise InvalidParameterError("Unrecognized network shape.")
    
    return output


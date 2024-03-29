"""
Main script to set parameters and run the simulation

Author:
Ravshan Hojimat
University of Klagenfurt
"""

import logging
from multiprocessing import Pool
import numpy as np
import progressbar
from models import Nature
import nkpack as nk

########
MC = 1000 # number of repetitions
BLUEPRINT = {
"p": (40,), # number of agents
"n": (4,), # number of bits
"kcs": ((3,0,0),(2,1,2),(2,2,2)), # K,C,S parameters
"t": (500,), # life span of organization
"rho": (0.9,0.6), # correlation

"nsoc": (4,), # number of social bits
"deg": (2,),  # two types of degrees
"net": (1,2,3,4), # network structures 0=random,1=line,2=cycle,3=ring,4=star
"xi": (1.0,), # probability of communicating
"tm": (50,), # memory
"coord": (1,2,0), # coordination mode 0=decentralized, 1=lateral, 2=hierarchical

"apc": ((2,2,2),(4,2,2),(4,2,4),(2,2,4)), # ALT,PROP,COMP parameters
"wf": (1.0,), # weight for phi, incentive scheme
"goals": ((1.0, 1.0),), # goals for incentives and for conformity
"w": (1.0, 0.5), # weight for incentives ; weight for conformity is 1-w

"normalize": (False,), # normalizes by global maximum
"precompute": (False,), # pre-computes performances for all bitstrings
}

########

def run_simulation(parameters, bar_, mc_):
    """A set of instructions for a single iteration"""
    nature = Nature(**parameters)
    # add random seed to ensure that no repetitions happen during multiprocessing
    np.random.seed() 
    nature.initialize()
    nature.play()
    perfs = nature.organization.performances.mean(axis=1)
    syncs = nature.organization.synchronies
    bar_.update(mc_)
    return perfs, syncs


if __name__=='__main__':
    logging.basicConfig(level=logging.WARNING)
    for params in nk.variate(BLUEPRINT):
        pbar = progressbar.ProgressBar(max_value=MC)

        def worker(i):
            """A worker function that calls iterations in a Pool"""
            return run_simulation(params, pbar, i)

        pbar.start()
        with Pool(8) as pool:
            quantum = pool.map(worker, range(MC))
        pbar.finish()

        # T x MC array of mean performance and synchrony of an
        # organization at every period for MC repetitions
        performances = [z[0] for z in quantum]
        synchronies = [z[1] for z in quantum]

        # save to files
        params_filename = "".join(f"{k}{v}" for k,v in params.items())
        params_filename = params_filename.replace(" ", "") + ".csv"

        np.savetxt("results/perf/" + params_filename, performances, delimiter=',', fmt='%10.5f')
        np.savetxt("results/sync/" + params_filename, synchronies, delimiter=',', fmt='%10.5f')

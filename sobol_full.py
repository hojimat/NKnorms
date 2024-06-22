"""
Performs full Sobol analysis by running simulations

The process flow is as follows:
   main() -> 
   get_all_outcomes() ->
   get_simrun_outcome() ->
   Nature() ->
   summarize() ->
   analyze_sobol()

"""

import gc
from multiprocessing import Pool
from copy import deepcopy
import numpy as np
from numpy.typing import NDArray
from SALib.sample import sobol as sb
from SALib.analyze import sobol
from models import Nature


NUM_SCEN = 2**11
PARAMS = {
    "p": 5, "n": 4, "t": 500, "nsoc": 4, "deg": 2, 
    "xi": 1.0, "wf": 1.0, "goals": (1.0, 1.0),
    "normalize": True, "precompute": True
    }


def sample_saltelli(num_scenarios: int, prblm) -> NDArray[np.int8]:
    """
    Quasi-randomly generates parameter combinations
    in the multidimensional space with low discrepancy
    """
    param_sets = sb.sample(prblm, num_scenarios, calc_second_order=True)
    discrete = param_sets[:,:-2].round()
    continuous = param_sets[:,-2:].round(1)
    param_sets = np.hstack((discrete, continuous))
    return param_sets

def summarize(outcm):
    """Calculates the statistics of interest"""
    f50 = outcm[:50].mean()
    f100 = outcm[:100].mean()
    f500 = outcm.mean()
    fmax = outcm.max()
    return [f50, f100, f500, fmax]

def analyze_sobol(prblm, values, fname):
    vals = ["f50", "f100", "f500", "fmax"]
    with open(fname, "w") as file:
        for i,val in enumerate(vals):
            indices = sobol.analyze(prblm, values[:,i])
            file.write(f"{val}:\n")
            file.write(f"{indices}\n\n")

def get_simrun_outcome(prms):
    """Run a single simulation give params"""

    parameters = deepcopy(PARAMS) # deep copy just in case
    parameters['kcs'] = (int(prms[0]), int(prms[1]), int(prms[2]))
    parameters['coord'] = int(prms[3])
    parameters['apc'] = (int(prms[4]), 2, int(prms[5]))
    parameters['net'] = int(prms[6])
    parameters['tm'] = int(prms[7])
    parameters['w'] = prms[8]
    parameters['rho'] = prms[9]
    print(parameters)

    nature = Nature(**parameters)
    np.random.seed()
    nature.initialize()
    nature.play()
    perfs = nature.organization.performances.mean(axis=1)
    syncs = nature.organization.synchronies

    perfSummary = summarize(perfs)
    syncSummary = summarize(syncs)

    del nature
    gc.collect()

    return perfSummary, syncSummary


def get_all_outcomes(prblm):
    '''Central place to run loops; multiprocessing needed here'''
    param_sets = sample_saltelli(NUM_SCEN, prblm)
    print(f"Params shape is {param_sets.shape}")

    with Pool(2) as pool:
        results = pool.map(get_simrun_outcome, param_sets)
    perfs = [res[0] for res in results]
    syncs = [res[1] for res in results]

    #perfs = []
    #syncs = []
    #for i,params in enumerate(param_sets):
    #    print(i)
    #    perf, sync = get_simrun_outcome(params)
    #    perfs.append(perf)
    #    syncs.append(sync)
    #
    #    if i > 0 and i % 4 == 0 :
    #        gc.collect()
    #

    return np.array(perfs), np.array(syncs)


def main():
    """main function"""   
    problem = {
        "num_vars": 10,
        "names": [    "k",   "c",   "s", "coordination", "alt", "comp", "network",     "tm",   "w", "correlation"  ],
        "bounds": [ [0,3], [0,4], [0,3],          [0,2], [2,4],  [2,4],     [1,4], [10,100], [0,1],       [0.5,1]  ]
    }

    perfs, syncs = get_all_outcomes(problem)
    analyze_sobol(problem, perfs, "perfsens.txt")
    analyze_sobol(problem, syncs, "syncsens.txt")
    
    print("Done")


if __name__=="__main__":
    main()

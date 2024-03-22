"""
Performs naive Sobol analysis on
the existing simulation results
"""

import json
import re
from pathlib import Path
import logging
import numpy as np
from numpy.typing import NDArray
from SALib.sample import sobol as sb
from SALib.analyze import sobol

NUM_SCEN = 2**14

def sample_saltelli(num_scenarios: int, prblm) -> NDArray[np.int8]:
    """
    Quasi-randomly generates parameter combinations
    in the multidimensional space with low discrepancy
    """
    param_sets = sb.sample(prblm, num_scenarios, calc_second_order=True)
    param_sets = param_sets.round().astype(np.int8)

    return param_sets

def get_filename(params: NDArray[np.int8], patts: dict) -> str:
    """
    Takes the parameter values and returns
    the matching scenario.csv
    """
    # Decipher the parameters
    keys = (
        ("kcs300", "kcs222"),
        ("decent", "lateral", "hierar"),
        ("apc222", "apc422", "apc224", "apc424"),
        ("line", "cycle", "ring", "star"),
    )

    values = [keys[i][prm-1] for i,prm in enumerate(params)]

    # Find files that match this description
    candidates = []
    top_dir = Path("./perf")
    for file in top_dir.iterdir():
        checks = [re.search(patts[value], file.name) for value in values]
        # final check to filter out "no conformity" case
        checks.append(re.search("w0.5", file.name))
        if all(checks):
            candidates.append(file.name)

    if len(candidates)>1:
        print("Matched more than one file!")
        print(candidates)
    elif len(candidates)==0:
        print("Matched no files!")

    return candidates[0]

def get_simrun_outcome(fname, rand):
    fpath = Path("perf")/fname
    quantum = np.genfromtxt(fpath, delimiter=',')
    num_runs = quantum.shape[0]
    simrun = rand.integers(0, num_runs)
    outcome = quantum[simrun,:].mean()
    return outcome

def get_all_outcomes(patts, prblm, rand) -> NDArray:
    param_sets = sample_saltelli(NUM_SCEN, prblm)
    outcomes = []
    for params in param_sets:
        filename = get_filename(params, patts)
        logging.info(f"{params}: {filename}")
        simrun_outcome = get_simrun_outcome(filename, rand)
        outcomes.append(simrun_outcome)

    return np.array(outcomes)

def analyze_sobol(prblm, values):
    indices = sobol.analyze(prblm, values)
    return indices

def main():
    """main function"""
    with open('structure.json', 'r', encoding='utf-8') as file:
        data = json.load(file)
        patterns = data['patterns']
    
    problem = {
        "num_vars": 4,
        "names": ["interaction", "coordination", "apc", "network"],
        "bounds": [[1,2], [1,3], [1,4], [1,4]]
    }

    rng = np.random.default_rng()

    outcomes = get_all_outcomes(patterns, problem, rng)
    indices = analyze_sobol(problem, outcomes)
    print(indices)
    breakpoint()

if __name__=="__main__":
    logging.basicConfig(level=logging.INFO)
    main()
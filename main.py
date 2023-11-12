from architecture import Organization
import numpy as np
import nkpack as nk
from matplotlib import pyplot as plt
import progressbar
from math import sqrt
from multiprocessing import Pool
from time import time, sleep

########
MC = 1000 # number of repetitions
BLUEPRINT = {
"p": (5,), # number of agents
"n": (4,), # number of bits
"kcs": ((3,4,3),(1,0,0)), # K,C,S parameters
"t": (500,), # life span of organization
"rho": (0.3, 0.9), # correlation

"nsoc": (0,2,4), # number of social bits
"deg": (2,),  # two types of degrees
"net": ("random","line","cycle","ring","star"), # network structures
"xi": (1.0,), # probability of communicating
"tm": (50,), # memory
"w": (0.5,), # weight for phi ; soc = 1- desc

"apc": ((2,2,4),), # ALT,PROP,COMP parameters
"wf": (1.0,), # weight for phi, incentive scheme

"lazy": (False,) # skips normalization by global maximum
}

########

for params in nk.variate(BLUEPRINT):
    bar = progressbar.ProgressBar(max_value=MC)
    bar.start()

    def single_iteration(mc):
        firm = Organization(**params)
        np.random.seed()
        firm.define_tasks()
        firm.hire_people()
        firm.form_networks()
        firm.play()
        past_perf = firm.perf_hist
        past_sim = firm.nature.past_sim
        bar.update(mc)
        return past_perf, past_sim
        
    #pool = Pool(4)
    quantum = []
    for i in range(MC):
        quantum.append(single_iteration(i))
    #quantum.append(pool.map(single_iteration,range(MC)))
    #pool.close()
    bar.finish()
    quantum = quantum[0]
    past_perf = [z[0] for z in quantum]
    past_sim = [z[1] for z in quantum]
    params_filename = "".join(f"{k}{v}" for k,v in params.items())
    np.savetxt("../tab_perf/" + params_filename + ".csv", past_perf, delimiter=',', fmt='%10.5f')
    np.savetxt("../tab_sim/" + params_filename + ".csv", past_sim, delimiter=',', fmt='%10.5f')


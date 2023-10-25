from architecture import Organization
import numpy as np
import NKPackage as nk
from matplotlib import pyplot as plt
import progressbar
from math import sqrt
from multiprocessing import Pool
from time import time, sleep

########
MC = 1000 # number of repetitions
BLUEPRINT = {
"p": [4],
"n": [4],
"kcs": [[3,4,3],[1,0,0]],
"t": [500],
"rho": [0.3, 0.9],
"eps": [0.0], #error std. dev
"eta": [0.0], #error prob for social bits
"nsoc": [0,2,4], # number of social bits
"deg": [2],  #two types of degrees
"xi": [1.0], #probability of communicating
"ts": [50], #schism time
"tm": [50], #memory
"wf": [[1.0,0.0]], # weights for phi phi_total
"w": [[0.5,0.5]], #goals for phi and desc
"ubar": [[1.0,1.0]], # goals for phi and desc
"opt": [1], # 1 - goal ; 2 - schism
"lazy": [True]
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
        perf_hist = firm.perf_hist
        soc_hist = np.mean(firm.nature.past_soc,1)
        bar.update(mc)
        return perf_hist, soc_hist
    pool = Pool(4)
    quantum = [] 
    quantum.append(pool.map(single_iteration,range(MC)))
    pool.close()
    bar.finish()
    quantum = quantum[0]
    perf_hist = [z[0] for z in quantum]
    param_string = "".join(f"{k}{v}" for k,v in params.items())
    np.savetxt("../tab_perf/" + param_string + ".csv", perf_hist, delimiter=',', fmt='%10.5f')

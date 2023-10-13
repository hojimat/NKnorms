from architecture import Organization
import numpy as np
from matplotlib import pyplot as plt
import progressbar
from math import sqrt
from multiprocessing import Pool
from time import time, sleep

########
P = 4#population
N = 4
#K = 3
#C = 4
#S = 3
T = 500
#RHO = 0.3#correlation
EPS = 0.0#error std. dev
ETA = 0.0#error prob for social bits
#NSOC = 2 
DEG = 2 #two types of degrees
XI = 1.0 #probability of communicating
TS = 50 #schism time
TM = 50 #memory
WF = [1.0,0.0]# weights for phi phi_total
W = [0.5,0.5]#goals for phi and desc
UBAR = [1.0,1.0]# goals for phi and desc
OPT = 1 # 1 - goal ; 2 - schism
LAZY = True
MC = 1000

########
for RHOx in [0.3, 0.9]:
    for NSOCx in [0,1,2,3,4]:
        for Kx,Cx,Sx in [[2,0,0],[1,1,1],[3,0,0],[1,2,1],[1,1,2],[3,3,1],[3,1,3],[2,2,2]]:
            bar = progressbar.ProgressBar(max_value=MC)
            bar.start() 
            def single_iteration(mc):
                firm = Organization(p=P,
                                    n=N,
                                    k=Kx,
                                    c=Cx,
                                    s=Sx,
                                    t=T,
                                    rho=RHOx,
                                    eps=EPS,
                                    eta=ETA,
                                    ts=TS,
                                    tm=TM,
                                    nsoc=NSOCx,
                                    degree=DEG,
                                    xi=XI,
                                    w=W,
                                    wf=WF,
                                    ubar=UBAR,
                                    opt=OPT,
                                    lazy=LAZY)
                np.random.seed()
                firm.define_tasks()
                firm.hire_people()
                firm.form_cliques()
                firm.play()
                perf_hist = firm.perf_hist
                soc_hist = np.mean(firm.nature.past_soc,1)
                bar.update(mc)
                return perf_hist, soc_hist
            pool = Pool(6)
            quantum = [] 
            quantum.append(pool.map(single_iteration,range(MC)))
            pool.close()
            bar.finish()
            quantum = quantum[0]
            perf_hist = [z[0] for z in quantum]
            #soc_hist = [z[1] for z in quantum]
            np.savetxt(f"../tab_perf/P{P}N{N}K{Kx}C{Cx}S{Sx}T{T}RHO{RHOx}EPS{EPS}ETA{ETA}TM{TM}NSOC{NSOCx}DEG{DEG}XI{XI}W{W}WF{WF}UBAR{UBAR}.csv",perf_hist,delimiter=',',fmt='%10.5f')
            #np.savetxt(f"../tab_soc/P{P}N{N}K{Kx}C{Cx}S{Sx}T{T}RHO{RHOx}EPS{EPS}ETA{ETA}TM{TM}NSOC{NSOCx}DEG{DEG}XI{XI}W{W}WF{WF}UBAR{UBAR}.csv",soc_hist,delimiter=',',fmt='%10.5f')


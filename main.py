from architecture import Organization
import numpy as np
from matplotlib import pyplot as plt
import progressbar
from math import sqrt
from multiprocessing import Pool
from time import time, sleep

########
P = 50#population
N = 4
#K = 3
#C = 4
#S = 3
T = 500
#RHO = 0.3#correlation
EPS = 0.0#error std. dev
ETA = 0.0#error prob for social bits
#NSOC = 2 
DEG = 4 #degree
XI = 1.0 #probability of communicating
#NET = 0 # 0 - full; 1 - line; 2 - cycle; 3 - ring; 4 - star;
TS = 50 #schism time
TM = 50 #memory
WF = [1.0,0.0]# weights for phi phi_total
#W = [0.5,0.5]#goals for phi and desc
UBAR = [1.0,1.0]# goals for phi and desc
OPT = 1 # 1 - goal ; 2 - schism
GMAX = False
MC = 10

########
for RHOx in [0.3, 0.9]:
    for Kx,Cx,Sx in [[3,0,0],[1,2,1],[3,3,1],[2,2,2]]:
        for NETx in [0,1,2,3,4]:
            for Wx in [[0.5,0.5], [0.75,0.25], [1,0]]:
                for NSOCx in [2, 4]:
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
                                            net=NETx,
                                            w=Wx,
                                            wf=WF,
                                            ubar=UBAR,
                                            opt=OPT,
                                            gmax=GMAX)
                        np.random.seed()
                        firm.define_tasks()
                        firm.hire_people()
                        firm.form_cliques()
                        firm.play()
                        past_perf = firm.perf_hist
                        past_sim = firm.nature.past_sim
                        past_simb = firm.nature.past_simb
                        bar.update(mc)
                        return past_perf, past_sim, past_simb
                    pool = Pool(4)
                    quantum = [] 
                    quantum.append(pool.map(single_iteration,range(MC)))
                    pool.close()
                    bar.finish()
                    quantum = quantum[0]
                    past_perf = [z[0] for z in quantum]
                    past_sim = [z[1] for z in quantum]
                    past_simb = [z[2] for z in quantum]
                    np.savetxt(f"../tab_perf/P{P}N{N}K{Kx}C{Cx}S{Sx}T{T}RHO{RHOx}EPS{EPS}ETA{ETA}TM{TM}NSOC{NSOCx}DEG{DEG}XI{XI}NET{NETx}W{Wx}WF{WF}UBAR{UBAR}.csv",past_perf,delimiter=',',fmt='%10.5f')
                    np.savetxt(f"../tab_sim/P{P}N{N}K{Kx}C{Cx}S{Sx}T{T}RHO{RHOx}EPS{EPS}ETA{ETA}TM{TM}NSOC{NSOCx}DEG{DEG}XI{XI}NET{NETx}W{Wx}WF{WF}UBAR{UBAR}.csv",past_sim,delimiter=',',fmt='%10.5f')
                    np.savetxt(f"../tab_simb/P{P}N{N}K{Kx}C{Cx}S{Sx}T{T}RHO{RHOx}EPS{EPS}ETA{ETA}TM{TM}NSOC{NSOCx}DEG{DEG}XI{XI}NET{NETx}W{Wx}WF{WF}UBAR{UBAR}.csv",past_simb,delimiter=',',fmt='%10.5f')


from architecture import Organization
import numpy as np
from matplotlib import pyplot as plt
import progressbar
from math import sqrt
from multiprocessing import Pool
from time import time,sleep

########
P = 50
N = 4
K = 3
C = 2
S = 3
T = 100
RHO = 0.0
EPS = 0.0
ETA = 0.0
NSOC = 4
DEG = 2
XI = 1.0 # probability of connecting
NET = 4 # 0 - random; 1 - line; 2 - cycle; 3 - ring; 4 - star;
TS = 50
TM = 20
W = [0.5,0.5]#,0.0]#phi,desc
WF = [1.0,0.0]# weights for phi phi_total
MC = 3
UBAR = [1.0, 1.0]
OPT = 1 # 0 - weighted ; 1 - goal ; 2 - schism
GMAX = False # global max
########

def single_iteration(p,n,k,c,s,t,rho,eps,eta,ts,tm,nsoc,degree,xi,net,w,wf,ubar,opt,gmax):
    firm = Organization(p=p,n=n,k=k,c=c,s=s,t=t,rho=rho,eps=eps,eta=eta,ts=ts,tm=tm,nsoc=nsoc,degree=degree,xi=xi,net=net,w=w,wf=wf,ubar=ubar,opt=opt,gmax=gmax)
    t1 = time()
    firm.define_tasks()
    t2 = time()
    firm.hire_people()
    firm.form_cliques()
    t4 = time()
    firm.play()
    t5 = time()
    realization = firm.perf_hist, firm.nature.past_sim
    print(f"define tasks={t2-t1}")
    print(f"firm play={t5-t4}")
    return realization

quantum = []
soctum = []
for mc in range(MC):
    print(mc)
    single_row, single_soc = single_iteration(p=P,n=N,k=K,c=C,s=S,t=T,rho=RHO,eps=EPS,eta=ETA,ts=TS,tm=TM,nsoc=NSOC,degree=DEG,xi=XI,net=NET,w=W,wf=WF,ubar=UBAR,opt=OPT,gmax=GMAX)
    quantum.append(single_row)
    soctum.append(single_soc)

superposition = np.mean(quantum,axis=0)
supersd = np.std(quantum,axis=0)
supererr = supersd*2.326/sqrt(MC)

#plt.plot(list(range(T)),superposition)
#plt.fill_between(list(range(T)),superposition-supererr,superposition+supererr,alpha=0.5)
#plt.legend(prop={'size': 5})

#plt.grid(True)
#plt.savefig(f"../fig/yeast{P}.pdf")
#plt.show()

#print(np.mean(quantum,0))
#print(np.mean(soctum,0))
#print(np.std(soctum,0))

from setup import Organization, Agent
import numpy as np
from matplotlib import pyplot as plt
import progressbar
from math import sqrt

'''
Syntax guidelines:
	- attributes and variables are lowercase
	- constants are UPPERCASE
	- don't use camelCase
'''

POP = 6
N = 4
K = 3
C = 4
S = 5
NPUB = 4
DEG = 9
NART = 6
NREV = 30
T = 200
TD = 100
TM = 25
TJ = 1400
W0 = 0.000033
W1 = 0.33
W2 = 0.33
MC = 20

#for jj in [10,20,30,40,50,60,70,80,90]:
for jj in [1]:
#for jj in [0,5]:
#for jj in [5,20,40]:
#for jj in [0,1,3,5,7,9]:
#for jj in [1,5,9]:
    quantum = np.zeros((MC,T))
    bar = progressbar.ProgressBar(max_value=MC)
    for mc in range(MC):
        firm = Organization(pop=POP,n=N,npub=NPUB,nart=NART,nrev=NREV,k=K,degree=DEG,c=C,s=S,t=T,td=TD,tm=TM,tj=TJ,w0=W0,w1=W1,w2=W2)
        firm.hire_people()
        firm.form_cliques()
        firm.play()
        quantum[mc,:] = firm.performance_history / firm.pop
        Agent.lst.clear()
        bar.update(mc)
    bar.finish()
    superposition = np.mean(quantum,axis=0)
    supersd = np.std(quantum,axis=0)
    supererr = supersd*2.326/sqrt(MC)

    plt.plot(list(range(T)),superposition,label=f"N={N},K={K},TD={TD}")
    plt.fill_between(list(range(T)),superposition-supererr,superposition+supererr,alpha=0.5)


plt.legend()
#plt.savefig(f"../../figcentral/sim/N={N},K={K},Th=sens.pdf")
plt.savefig(f"../recent.pdf")
#plt.show()

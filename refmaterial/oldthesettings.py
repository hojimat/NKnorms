import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plt
import NKPackage as nk
import progressbar

# A Landscape class
class Landscape:
    def __init__(self,N,K,T,nay):
        self.N = N
        self.T = T
        self.nay = nay # number of neighbours to look for in a single climb step
        self.decision_space = np.zeros((T,N),dtype=int)
        self.contrib_space = np.zeros(T,dtype=float)
        self.betas = np.ones((T,2**N,2), dtype=int)
        #self.betasB = np.ones((T,2**N,2), dtype=int)
        #self.betasD = np.ones((T,2**N,2), dtype=int)
        #self.betasN = np.ones((T,2**N,2), dtype=int)
        self.inmat = nk.interaction_matrix(N,K)
        self.conmat = nk.contrib_define(N,K)
        self.perfmat, self.perfmax = nk.perf_calc(self.inmat, self.conmat)
        return


    def initialize(self):
        maxbit = self.decision_space[0,:]
        maxbit = np.array(maxbit)
        maxidx = 0
        self.contrib_space[0] = self.perfmat[0]
        self.betas[0][maxidx][0] += 1
        return

    def simulation(self):
        for t in range(1,self.T):
            tmp = self.decision_space[t-1,:].copy()
            bmp = self.betas[t-1].copy()

            force = nk.reinforcement_climb(tmp,bmp,self.nay,self.perfmat)
            self.decision_space[t,:] = force[0]
            self.contrib_space[t] = force[1]
            self.betas[t] = force[2]
        return

    def share_public_bits(self):
        tmp = self.decision_space[t-Th:t,]
        output = tmp
        return output

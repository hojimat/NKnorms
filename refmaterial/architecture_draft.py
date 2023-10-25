'''
The file containing the architecture for the organization
operating on NK framework with multiple interacting agents.

The architecture features 3 objects:
1. Organization: allocates tasks, observes outcomes.
2. Agent: makes decisions on interdependent tasks,
    interacts with colleagues, shares information in networks.
3. Nature: a hidden object that does the processing:
    observes reported states, calculates the performances,
    shares the results.

The code heavily relies on the satelite NKPackage for required utilities.

Created by Ravshan S.K.
I'm on Twitter @ravshansk
'''
import numpy as np
import NKPackage as nk
from time import time,sleep

class Organization:
    ''' Defines tasks, hires people; aggregation relation with Agent class.'''
    def __init__(self,p,n,nsoc,k,c,s,degree,xi,net,t,rho,eps,eta,ts,tm,w,wf,ubar,opt,lazy):
        self.p = p # population
        self.n = n # number of tasks per agent
        self.k = k # number of internally coupled bits
        self.c = c # number of externally coupled bits
        self.s = s # number of externally coupled agents 
        self.nsoc = nsoc # number of social bits
        self.degree = degree # degree of network of agents (analog and digital)
        self.xi = xi # probability of connecting through channel
        self.net = net # network topology
        self.t = t # lifespan of the organization
        self.w = w # weights for performance and social norms
        self.wf = wf # weights for individual vs. collective incentive system
        self.rho = rho # correlation coefficient among individual landscapes
        self.eps = eps # level of accuracy of predicting performance
        self.eta = eta # error probability in sharing social bits
        self.ts = ts # list of social periods (needed if opt=2|schism)
        self.tm = tm # memory span of agents
        self.ubar = ubar # goal levels for performance and social norms
        self.opt = opt # optimization techniques; 1: goal prog 2: schism
        self.nature = None # reference to the Nature class
        self.lazy = lazy # lazy calculation of performances; RAM-efficient, but CPU-heavy
        self.agents = [] # reference to the Agents
        self.perf_hist = np.zeros(t,dtype=float) # performance history storage
    
    def define_tasks(self):
        '''Creates the Nature with given parameters'''
        nature = Nature(p=self.p,n=self.n,k=self.k,c=self.c,s=self.s,t=self.t,rho=self.rho,lazy=self.lazy)
        nature.set_interactions()
        nature.set_landscapes() # !!! processing heavy !!!
        self.nature = nature

    def hire_people(self):
        '''Creates the Agents and stores them'''
        for i in range(self.p):
            self.agents.append(Agent(employer=self))

    def form_cliques(self):
        '''Generates the network structure for agents to communicate;
        it can be argued that a firm has the means to do that,
        e.g. through hiring,communication tools etc.''' 
        p = self.p
        degree = self.degree
        xi = self.xi
        net = self.net
        netshape = ["random","line","cycle","ring","star"]
        cliques = nk.generate_network(p,degree,xi,netshape[net])
        for c,a in zip(cliques,self.agents):
            a.clique = c

    def observe_outcomes(self,tt):
        '''Receives performance report from the Nature'''
        self.perf_hist[tt] = self.nature.current_perf.mean()

    def initialize(self):
        '''Initializes the simulation'''
        for agent in self.agents:
            agent.initialize()
            agent.report_state()
            agent.nsoc_added = np.zeros(self.t,dtype=int)
            agent.nsoc_added[0] = agent.soc_memory.shape[0]

        self.nature.calculate_perf()
        self.observe_outcomes(0)
        self.nature.archive_state()

    def play(self):
        '''The central method. Runs the lifetime simulation of the organization.'''
        self.initialize()
        for t in range(1,self.t):
            # check if the period is active under schism (ignore for goal programing):
            social = True if t in [z for z in range(self.t) if int(z/self.ts)%2==1] else False
            # at exactly t==TM, the memory fills (training ends) and climbing is done from scratch
            if t==self.tm:
                for agent in self.agents:
                    agent.current_state = np.zeros(agent.n*agent.p,dtype=int)
            # every agent performs a climb and reports the state:
            for agent in self.agents:
                agent.perform_climb(soc=social)
                agent.report_state()
            # nature observes the reported state and calculates the performances
            self.nature.calculate_perf()
            # firm observes the outcomes
            self.observe_outcomes(t)
            # agents forget old social norms
            for agent in self.agents:
                agent.forget_soc(t)
            # agents share social norms and observe the realized state
            for agent in self.agents:
                agent.share_soc(t)
                agent.observe_state()
            # nature archives the state 
            self.nature.archive_state()

class Agent:
    ''' Decides on tasks, interacts with peers; aggregation relation with Organization class.'''
    def __init__(self,employer):
        # adopt variables from the organization; not an inheritance.
        self.id = len(employer.agents)
        self.employer = employer
        self.nature = employer.nature
        self.n = employer.n
        self.p = employer.p
        self.eps = employer.eps
        self.eta = employer.eta
        self.nsoc = employer.nsoc
        self.degree = employer.degree
        self.t = employer.t
        self.ts = employer.ts
        self.tm = employer.tm
        self.w = employer.w
        self.wf = employer.wf
        self.ubar = employer.ubar
        self.opt = employer.opt
        # current status
        self.current_state = np.zeros(self.n*self.p,dtype=int)
        self.current_betas = np.ones((2**self.n,2),dtype=int)
        self.phi_soc = 0.0
        self.current_util = 0.0
        self.current_perf = 0.0
        self.current_soc = np.repeat(-1,self.nsoc)
        # information about social interactions
        self.soc_memory = np.repeat(-1,2*self.nsoc).reshape(2,self.nsoc) # social storage matrix
        self.clique = [] # reference to agents in the network

    def initialize(self):
        '''Initializes agent after creation'''
        self.current_perf = self.nature.phi(None, self.current_state)
        self.current_util = self.current_perf
        self.current_betas[0,0] += 1

    def perform_climb(self,lrn=False,soc=False):
        '''The central method. Contains the main decision process of the agent'''
        # get attributes as local variables
        w = self.w.copy()
        wf = self.wf.copy()
        ubar = self.ubar
        opt = self.opt

        # get "before" parameters
        bit0 = self.current_state.copy() # current bitstring
        idx0 = nk.get_index(bit0,self.id,self.n) # location of ^
        all_phis = list(self.current_perf) # vector of performances of everybody
        my_phi = all_phis.pop(self.id) # get own perf
        other_phis = np.mean(all_phis) # get rest perfs
        phi0 = wf[0] * my_phi + wf[1] * other_phis # calculate earnings
        beta0 = self.current_betas[idx0,:] # current beliefs
        soc0 = self.current_soc # current social bits (subset of bit0) 

        # get "after" parameters
        bit1 = nk.random_neighbour(bit0,self.id,self.n) # candidate bitstring
        idx1 = nk.get_index(bit1,self.id,self.n) # location of ^
        my_phi, other_phis = self.nature.phi(self.id,bit1,self.eps) # tuple of own perf and mean of others
        phi1 = wf[0] * my_phi + wf[1] * other_phis # calc potential earnings
        beta1 = self.current_betas[idx1,:] # calc potential updated beliefs
        soc1 = nk.extract_soc(bit1,self.id,self.n,self.nsoc) # potential social bits (subset of bit1)

        # calculate mean betas
        mbeta0 = nk.beta_mean(*beta0)
        mbeta1 = nk.beta_mean(*beta1)

        # calculate soc frequency
        fsoc0 = nk.calculate_freq(soc0,self.soc_memory)
        fsoc1 = nk.calculate_freq(soc1,self.soc_memory)

        # calculate utility 
        util0 = None
        util1 = None
        if opt==1: # goal programming
            util0 = nk.goal_prog(phi0,fsoc0,ubar,w[0],w[1])
            util1 = nk.goal_prog(phi1,fsoc1,ubar,w[0],w[1])
        elif opt==2: # schism optimization
            util0 = nk.schism(phi0,fsoc0,soc)
            util1 = nk.schism(phi1,fsoc1,soc)
        # the central decision to climb or stay
        if util1 > util0:
            self.current_state = bit1
            self.phi_soc = fsoc1
        else:
            self.phi_soc = fsoc0

        # update beliefs (betas) 
        self.current_betas[idx1,int(phi1<phi0)] += 1

    def share_soc(self,tt):
        '''shares social bits with agents in a clique'''
        # get own social bits
        idd = self.id
        n = self.n
        p = self.p
        nsoc = self.nsoc
        clique = self.clique
        current = self.current_state.copy()
        current_soc = nk.extract_soc(current,idd,n,nsoc)
        noisy_soc = nk.with_noise(current_soc,self.eta)
        
        # share social bits with the clique
        for i in range(p):
            connect = np.random.choice(2,p=[1-clique[i], clique[i]])
            if connect:
                current_memory = self.employer.agents[i].soc_memory
                self.employer.agents[i].soc_memory = np.vstack((current_memory, noisy_soc))
                self.employer.agents[i].nsoc_added[tt] += 1
        
        # update own social bit attribute for future references
        self.current_soc = current_soc



    def forget_soc(self,tt):
        '''forgets social bits'''
        tm = self.tm
        sadd = self.nsoc_added
        if tt >= tm:
            self.soc_memory = self.soc_memory[sadd[tt-tm]:,:]

    def observe_state(self):
        '''observes the current bitstring choice by everyone'''
        self.current_state = self.nature.current_state.copy()
        self.current_perf = self.nature.current_perf


    def report_state(self):
        '''reports state to nature'''
        n = self.n
        i = self.id
        self.nature.current_state[i*n:(i+1)*n] = self.current_state[i*n:(i+1)*n].copy()
        self.nature.current_soc[i] = self.phi_soc

class Nature:
    '''Defines the performances, inputs state, outputs performance; a hidden class.'''
    def __init__(self,p,n,k,c,s,t,rho,lazy):
        self.p = p
        self.n = n
        self.k = k
        self.c = c
        self.s = s
        self.t = t
        self.rho = rho
        self.lazy = lazy
        self.inmat = None
        self.landscape = None
        self.argmax = None
        self.globalmax = None
        self.current_state = np.zeros(n*p,dtype=int)
        self.current_perf = np.zeros(p,dtype=float)
        self.current_soc = np.zeros(p,dtype=float)
        self.past_state = []
        self.past_perf = []
        self.past_soc = []

    def set_interactions(self):
        '''sets interaction matrices'''
        p = self.p
        n = self.n
        k = self.k
        c = self.c
        s = self.s
        tmp = np.zeros((n*p, n*p),dtype=int)
        if s>(p-1):
            print("Error: S is too large")
            return
        couples = nk.generate_couples(p,s)
        
        # the idea is to have similar interaction for rho=1 to work
        internal = nk.interaction_matrix(n,k,"random")
        external = nk.random_binary_matrix(n,c)
        # internal coupling
        for i in range(p):
            tmp[i*n:(i+1)*n, i*n:(i+1)*n] = internal

        # external coupling
        for i,qples in zip(range(p),couples):
            for qple in qples:
                tmp[i*n:(i+1)*n, qple*n:(qple+1)*n] = external

        self.inmat = tmp

    def set_landscapes(self):
        '''sets landscapes; set lazy=True for memory-efficient but CPU-heavy version'''
        p = self.p
        n = self.n
        k = self.k
        c = self.c
        s = self.s
        rho = self.rho
        lazy = self.lazy
        # if lazy just set contrib matrix; else create dictionary
        contrib = nk.contrib_define(p,n,k,c,s,rho)
        if lazy is False:
            tmp1,tmp3 = nk.contrib_full(self.inmat,contrib,n,p) # !!! processing heavy !!!
            self.landscape = tmp1
            self.globalmax = tmp3
        else:
            self.landscape = contrib
            self.globalmax = nk.get_globalmax(self.inmat,contrib,n,p) # !!! processing heavy !!!
            
    def phi(self,myid,x,eps=0.0):
        '''inputs bitstring, outputs performance; set lazy=True for memory-efficient but CPU-heavy version'''
        n = self.n
        p = self.p
        n_p = n*p
        imat = self.inmat
        cmat = self.landscape
        globalmax = self.globalmax
        lazy = self.lazy
        if len(x) != n_p:
            print("Error: Please enter the full bitstring")
            return
        # if lazy calculate phi, else get it from the dictionary
        if lazy is False:
            tmp = cmat[nk.dec2bin(x,n_p),:]
        else:
            tmp = nk.contrib_solve(x,imat,cmat,n,p) / globalmax

        tmp = tmp + np.random.normal(0,eps) # imperfect information
        output = tmp # if no index is given, return all perfs
        if myid is not None: # given index, return own perf and mean of others
            tmp1 = tmp[myid]
            tmp2 = np.delete(tmp,myid).mean() # mean of others perfs
            output = (tmp1,tmp2)
        return output

    def calculate_perf(self):
        '''uses phi to calculate current performance'''
        tmp = self.phi(myid=None,x=self.current_state.copy())
        self.current_perf = tmp
        output = tmp
        self.past_perf.append(output)
        return output

    def archive_state(self):
        '''archives state'''
        self.past_state.append(self.current_state.copy())
        self.past_soc.append(self.current_soc.copy())

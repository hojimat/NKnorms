"""
The file containing the architecture for the organization
operating on NK framework with multiple interacting agents.

The architecture features 3 objects:
1. Organization: allocates tasks, observes outcomes.
2. Agent: makes decisions on interdependent tasks,
    interacts with colleagues, shares information in networks.
3. Nature: a main object that owns agents, does the processing:
    observes reported states, calculates the performances,
    shares the results.

The code heavily relies on the satelite NKPackage for required utilities.

Created by Ravshan S.K.
I'm on Twitter @ravshansk
"""
from time import time, sleep
import numpy as np
from typing import List
from numpy.typing import NDArray
import nkpack as nk

class Nature:
    """Defines the performances, inputs state, outputs performance; a hidden class."""
    def __init__(self, **kwargs):
        # task environment (user input)
        self.p = kwargs['p'] # population / number of agents
        self.n = kwargs['n'] # number of tasks per agent
        self.k, self.c, self.s = kwargs['kcs'] # number of coupled bits
        self.rho = kwargs['rho'] # correlation coefficient among individual landscapes
        self.lazy = kwargs['lazy'] # normalize or not by the global maximum; CPU-heavy 
        # task environment (generated)
        self.globalmax : float = 1.0
        self.interaction_matrix : NDArray[np.int8] = None
        self.landscape : NDArray[np.float32] = None
        # social interactions
        self.nsoc = kwargs['nsoc'] # number of social bits
        self.degree = kwargs['deg'] # degree of network of agents (analog and digital)
        self.xi = kwargs['xi'] # probability of connecting through channel
        self.net = kwargs['net'] # network topology
        self.w = kwargs['w'] # weights for performance and social norms
        self.tm = kwargs['tm'] # memory span of agents
        # search behavior
        self.wf = kwargs['wf'] # weights for individual vs. collective incentive system
        # history
        self.t = kwargs['t'] # lifespan of the organization
        # players
        self.organization = None # reference to the Organization
        self.agents = None # reference to the Agents

    def create_environment(self):
        """Creates the environment"""
        self._generate_interaction_matrix()
        self._generate_landscape()
        if not self.lazy:
            self._calculate_global_maximum()

    def create_players(self):
        """Spawn main players: 1 organization and P agents"""
        self.organization = Organization(nature=self)
        self.agents = [Agent(nature=self) for i in range(self.p)]

    def _phi(self, bstring: NDArray[np.int8]) -> NDArray[np.float32]:
        """
        Calculates individual performances of all agents
        for a given bitstring.

        Args:
            bstring: an input bitstring

        Returns:
            P-sized vector with performances of a bitstring for P agents.
        """

        if len(bstring) != self.n * self.p:
            raise nk.InvalidBitstringError("Please enter the full bitstring.")

        perfs = nk.calculate_performances(bstring, self.interaction_matrix, self.landscape, self.n, self.p)
        normalized_performances = perfs / self.globalmax

        return normalized_performances

    def _calculate_current_performances(self):
        """
        I THINK I CAN DELETE THIS
        uses _phi to calculate current performances for all agents"""
        # get latest state
        current_state = self.states[-1]
        tmp = self._phi(current_state)
        self.current_perf = tmp # append or put to the current state
        output = tmp
        self.past_perf.append(output)

    def archive_state(self):
        """MOVE THIS FUNCTIONALITY SOMEWHEREarchives state"""
        self.past_state.append(self.current_state.copy())
        self.past_sim.append(nk.similarity(self.current_state, self.p, self.n, self.nsoc))
        self.past_simb.append(nk.similarbits(self.current_state, self.p, self.n, self.nsoc))


    def _generate_interaction_matrix(self):
        """generates the NKCS interaction matrix"""
        inmat = np.zeros((self.n * self.p, self.n * self.p), dtype=np.int8)
        
        if self.s > (self.p - 1):
            raise nk.InvalidParameterError("The value of S cannot exceed P-1.")

        # internal coupling
        # the idea is to randomly draw only once
        # and have the same internal interaction for
        # all agents. This allows rho=1 to work.
        internal = nk.interaction_matrix(self.n, self.k, "random")
        for i in range(self.p):
            inmat[i*self.n : (i+1)*self.n, i*self.n : (i+1)*self.n] = internal

        # external coupling
        external = nk.random_binary_matrix(self.n, self.c)
        peers_list = nk.generate_couples(self.p, self.s)
        for i, peers in zip(range(self.p), peers_list):
            for peer in peers:
                inmat[i*self.n : (i+1)*self.n, peer*self.n : (peer+1)*self.n] = external

        # save the interaction matrix
        self.interaction_matrix = inmat

    def _generate_landscape(self):
        """generates the landscape given by the interaction matrix."""
        self.landscape = nk.generate_landscape(self.p, self.n, self.k, self.c, self.s, self.rho)

    def _calculate_global_maximum(self):
        """
        !!! WARNING: The most processing-heavy part of the project !!!

        Calculates global maximum for normalization. Set lazy=True to skip.
        """
        self.globalmax = nk.get_globalmax(self.interaction_matrix, self.landscape, self.n, self.p)

class Organization:
    """ Defines tasks, hires people; aggregation relation with Agent class."""
    def __init__(self, nature):
        # environment and user-input params:
        self.nature = nature
        self.p = nature.p
        self.agents = nature.agents # "hire" all people from environment
        # histories:
        self.states = np.empty((nature.t, nature.n*nature.p), dtype=np.int8) # bitstrings history
        self.performances = np.empty((nature.t, nature.p), dtype=np.float32) # agents' performances 
        self.synchronies = np.empty(nature.t, dtype=np.float32) # synchrony measures history

    def form_networks(self):
        """Generates the network structure for agents to communicate;
        it can be argued that a firm has the means to do that,
        e.g. through hiring, defining interfaces etc."""
        peers_list = nk.generate_network(self.p, self.nature.degree, self.nature.xi, self.nature.net)
        for peers, agent in zip(peers_list, self.agents):
            agent.peers = peers

    def observe_outcomes(self,tt):
        """Receives performance report from the Nature"""
        self.perf_hist[tt] = self.nature.current_perf.mean()

    def initialize(self):
        """Initializes the simulation"""
        for agent in self.agents:
            agent.initialize()
            agent.report_state()
            agent.nsoc_added = np.zeros(self.t,dtype=np.int8)
            agent.nsoc_added[0] = agent.soc_memory.shape[0]

        self.nature.calculate_perf()
        self.observe_outcomes(0)
        self.nature.archive_state()

    def play(self):
        """The central method. Runs the lifetime simulation of the organization."""
        self.initialize()
        for t in range(1,self.t):
            # check if the period is active under schism (ignore for goal programing):
            social = True if t in [z for z in range(self.t) if int(z/self.ts)%2==1] else False
            # at exactly t==TM, the memory fills (training ends) and climbing is done from scratch
            if t==self.tm:
                for agent in self.agents:
                    agent.current_state = np.random.choice(2,agent.n*agent.p)
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
    """ Decides on tasks, interacts with peers; aggregation relation with Organization class."""
    def __init__(self, nature):
        # adopt variables from the organization; not an inheritance.
        self.id = len(nature.agents)
        self.nature = nature
        # current status
        self.current_state = np.random.choice(2,self.n*self.p)
        #self.current_betas = np.ones((2**self.n,2),dtype=np.int8)
        self.phi_soc = 0.0
        self.current_util = 0.0
        self.current_perf = 0.0
        self.current_soc = np.repeat(-1, nature.nsoc)
        # information about social interactions
        self.soc_memory = np.repeat(-1, 2*nature.nsoc).reshape(2, nature.nsoc) # social storage matrix
        self.peers = [] # agents this agent talks with in a network

    def initialize(self):
        """Initializes agent after creation"""
        self.current_perf = self.nature.phi(None, self.current_state)
        self.current_util = self.current_perf
        #self.current_betas[0,0] += 1

    def perform_climb(self,lrn=False,soc=False):
        """The central method. Contains the main decision process of the agent"""
        # get attributes as local variables
        w = self.w.copy()
        wf = self.wf.copy()

        # get "before" parameters
        bit0 = self.current_state.copy() # current bitstring
        idx0 = nk.get_index(bit0,self.id,self.n) # location of ^
        all_phis = list(self.current_perf) # vector of performances of everybody
        my_phi = all_phis.pop(self.id) # get own perf
        other_phis = np.mean(all_phis) # get rest perfs
        phi0 = wf[0] * my_phi + wf[1] * other_phis # calculate earnings
        #beta0 = self.current_betas[idx0,:] # current beliefs
        soc0 = self.current_soc # current social bits (subset of bit0) 

        # get "after" parameters
        bit1 = nk.random_neighbour(bit0,self.id,self.n) # candidate bitstring
        idx1 = nk.get_index(bit1,self.id,self.n) # location of ^
        my_phi, other_phis = self.nature.phi(self.id,bit1,self.eps) # tuple of own perf and mean of others
        phi1 = wf[0] * my_phi + wf[1] * other_phis # calc potential earnings
        #beta1 = self.current_betas[idx1,:] # calc potential updated beliefs
        soc1 = nk.extract_soc(bit1,self.id,self.n,self.nsoc) # potential social bits (subset of bit1)

        # calculate mean betas
        #mbeta0 = nk.beta_mean(*beta0)
        #mbeta1 = nk.beta_mean(*beta1)

        # calculate soc frequency
        fsoc0 = nk.calculate_frequency(soc0,self.soc_memory)
        fsoc1 = nk.calculate_frequency(soc1,self.soc_memory)

        # calculate utility 
        util0 = w[0] * phi0 + w[1] * fsoc0
        util1 = w[0] * phi1 + w[1] * fsoc1

        # the central decision to climb or stay
        if util1 > util0:
            self.current_state = bit1
            self.phi_soc = fsoc1
        else:
            self.phi_soc = fsoc0

        # update beliefs (betas) 
        #self.current_betas[idx1,int(phi1<phi0)] += 1

    def share_soc(self,tt):
        """shares social bits with agents in a clique"""
        # get own social bits
        idd = self.id
        n = self.n
        p = self.p
        nsoc = self.nsoc
        clique = self.peers
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
        """forgets social bits"""
        tm = self.tm
        sadd = self.nsoc_added
        if tt >= tm:
            self.soc_memory = self.soc_memory[sadd[tt-tm]:,:]

    def observe_state(self):
        """observes the current bitstring choice by everyone"""
        self.current_state = self.nature.current_state.copy()
        self.current_perf = self.nature.current_perf


    def report_state(self):
        """reports state to nature"""
        n = self.n
        i = self.id
        self.nature.current_state[i*n:(i+1)*n] = self.current_state[i*n:(i+1)*n].copy()
        self.nature.current_soc[i] = self.phi_soc


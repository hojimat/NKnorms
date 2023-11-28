'''Nature definition'''
from typing import Optional
import numpy as np
from numpy.typing import NDArray
import nkpack as nk
from .agent import Agent
from .organization import Organization

class Nature:
    '''
    Nature class defines the performances,
    for a given state can output performance;
    
    It does not store any history, because being a nature,
    it can define how the environment behaves, and can 
    give an immediate feedback for every set of actions,
    however, it does not have a responsiblity to remember
    the past.

    '''
    def __init__(self, **kwargs):
        # task environment (user input)
        self.p = kwargs['p'] # population / number of agents
        self.n = kwargs['n'] # number of tasks per agent
        self.k, self.c, self.s = kwargs['kcs'] # number of coupled bits
        self.rho = kwargs['rho'] # correlation coefficient among individual landscapes
        self.normalize = kwargs['normalize'] # normalize or not by the global maximum; CPU-heavy 
        self.precompute = kwargs['precompute'] # normalize or not by the global maximum; CPU-heavy 
        # task environment (generated)
        self.globalmax : float = 1.0
        self.interaction_matrix : NDArray[np.int8] = None
        self.landscape : NDArray[np.float32] = None
        self.performance_lookup: NDArray[np.float32] = None # Px2^(n*p) lookup table of bstring perfs
        # social interactions
        self.nsoc = kwargs['nsoc'] # number of social bits
        self.degree = kwargs['deg'] # degree of network of agents (analog and digital)
        self.xi = kwargs['xi'] # probability of connecting through channel
        self.net = kwargs['net'] # network topology
        self.w = kwargs['w'] # weights for performance and social norms
        self.tm = kwargs['tm'] # memory span of agents
        # search behavior
        self.alt, self.prop, self.comp = kwargs['apc'] # number of alternatives, proposals, composites
        self.wf = kwargs['wf'] # weights for individual vs. collective incentive system
        # history
        self.t = kwargs['t'] # lifespan of the organization
        # players
        self.organization: Optional[Organization] = None # reference to the Organization
        self.agents: Optional[list[Agent]] = None # reference to the Agents

    def create_environment(self):
        """Creates the environment"""
        
        self._generate_interaction_matrix()
        self._generate_landscape()

        # TODO: move landscape to a separate class that can handle
        # precomputing, normalizing etc. This tree here looks ugly.
        if self.normalize and self.precompute:
            self._calculate_all_performances()
        elif self.normalize and not self.precompute:
            self._calculate_global_maximum()
        elif not self.normalize and self.precompute:
            raise nk.InvalidParameterError('It is inefficient to precompute if normalization is not needed.')

    def create_players(self):
        '''Spawn main players: 1 organization and P agents'''
        self.organization = Organization(n=self.n, p=self.p, nature=self)
        self.agents = [Agent(id_=i,n=self.n,p=self.p,nsoc=self.nsoc,degree=self.degree,tm=self.tm,nature=self) for i in range(self.p)]
        self.organization.agents = self.agents

    def initialize(self):
        '''
        Initializes the parameters for the 0th step of the simulation
        
        '''
        
        # check for errors
        if (self.agents is None) or (self.organization is None):
            raise nk.UninitializedError("Initialize Agents and Organization before using them.")

        # form networks through which agents will each have peers to communicate with
        self.organization.form_networks()

        # Set the initial firm-wide random bitstring (length=N*P)
        initial_bstring = np.random.choice(2, self.n*self.p)
        self.organization.states[0,:] = initial_bstring

        # set states and calculate performances of P agents
        performances = self.phi(self.organization.states[0,:])
        for agent, perf in zip(self.agents, performances):
            agent._current_performance = perf
            agent.current_state = initial_bstring


        for agent in self.agents:
            agent.report_state()
            agent.nsoc_added = np.zeros(self.t,dtype=np.int8)
            agent.nsoc_added[0] = agent._received_bits_memory.shape[0]

        self.nature.calculate_perf()
        self.observe_outcomes(0)
        self.nature.archive_state()


    def phi(self, bstring: NDArray[np.int8]) -> NDArray[np.float32]:
        '''
        Calculates individual performances of all agents for a given bitstring.
        If precompute=True, it just looks up the performances in the lookup_table,
        else it calculates the performance of the given bitstring.

        Args:
            bstring: an input bitstring

        Returns:
            P-sized vector with performances of a bitstring for P agents.
        '''

        if len(bstring) != self.n * self.p:
            raise nk.InvalidBitstringError("Please enter the full bitstring.")

        if self.precompute:
            return self.performance_lookup[nk.bin2dec(bstring),:] / self.globalmax

        return nk.calculate_performances(bstring, self.interaction_matrix, self.landscape, self.n, self.p) / self.globalmax


    def _generate_interaction_matrix(self):
        '''generates the NKCS interaction matrix'''
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
        '''generates the landscape given by the interaction matrix.'''
        self.landscape = nk.generate_landscape(self.p, self.n, self.k, self.c, self.s, self.rho)

    def _calculate_global_maximum(self):
        '''
        !!! WARNING: The most processing-heavy part of the project !!!

        Calculates global maximum for normalization. Set normalize=False to skip.
        
        '''

        self.globalmax = nk.get_globalmax(self.interaction_matrix, self.landscape, self.n, self.p)

    def _calculate_all_performances(self):
        '''
        !!! WARNING: The most processing-heavy part of the project !!!

        Pre-computes all performances before simulation starts, and saves a 
        giant lookup table. Set precompute=False to skip.
        
        '''

        self.performance_lookup, self.globalmax = nk.calculate_all_performances(self.interaction_matrix, self.landscape, self.n, self.p)


    def _calculate_current_performances(self):
        '''
        TODO: I THINK I CAN DELETE THIS
        uses _phi to calculate current performances for all agents'''
        # get latest state
        current_state = self.states[-1]
        tmp = self.phi(current_state)
        self.current_perf = tmp # append or put to the current state
        output = tmp
        self.past_perf.append(output)

    def _archive_state(self):
        '''MOVE THIS FUNCTIONALITY SOMEWHEREarchives state'''
        self.past_state.append(self.current_state.copy())
        self.past_sim.append(nk.similarity(self.current_state, self.p, self.n, self.nsoc))
        self.past_simb.append(nk.similarbits(self.current_state, self.p, self.n, self.nsoc))


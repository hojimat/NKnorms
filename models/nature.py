"""Nature definition"""
from typing import Optional
import numpy as np
from numpy.typing import NDArray
import nkpack as nk
from .agent import Agent
from .organization import Organization
from .landscape import Landscape

class Nature:
    """
    Nature class defines the performances,
    for a given state can output performance;
    
    It does not store any history, because being a nature,
    it can define how the environment behaves, and can 
    give an immediate feedback for every set of actions,
    however, it does not have a responsiblity to remember
    the past.

    """
    def __init__(self, **kwargs):
        # task environment
        self.p = kwargs['p'] # population / number of agents
        self.n = kwargs['n'] # number of tasks per agent
        self.k, self.c, self.s = kwargs['kcs'] # number of coupled bits
        self.rho = kwargs['rho'] # correlation coefficient among individual landscapes
        self.normalize = kwargs['normalize'] # normalize or not by the global maximum; CPU-heavy 
        self.precompute = kwargs['precompute'] # normalize or not by the global maximum; CPU-heavy 
        # social interactions
        self.nsoc = kwargs['nsoc'] # number of social bits
        self.degree = kwargs['deg'] # degree of network of agents (analog and digital)
        self.xi = kwargs['xi'] # probability of connecting through channel
        self.net = kwargs['net'] # network topology
        self.tm = kwargs['tm'] # memory span of agents
        # search behavior
        self.alt, self.prop, self.comp = kwargs['apc'] # number of alternatives, proposals, composites
        self.wf = kwargs['wf'] # weights for individual vs. collective incentive system
        self.w = kwargs['w'] # weights for performance and conformity
        self.goals = kwargs['goals']
        # history
        self.t = kwargs['t'] # lifespan of the organization
        # players
        self.organization: Organization = None # reference to the Organization
        self.agents: list[Agent] = None # reference to the Agents
        self.landscape: Landscape = None # reference to the Landscape

    def create_environment(self):
        """Creates the task environment."""
        self.landscape = Landscape(self.p, self.n, self.k, self.c, self.s, self.rho, self.normalize, self.precompute)

    def create_players(self):
        """Spawn main players: 1 organization and P agents"""
        
        self.organization = Organization(n=self.n, p=self.p, goals=self.goals, nature=self)
        params = (self.n, self.p, self.nsoc, self.degree, self.tm, self.w, self.wf, self)
        self.agents = [Agent(i, *params) for i in range(self.p)]

    def initialize(self):
        """Initializes the parameters for the 0th step of the simulation."""
        
        # check for errors
        if (self.agents is None) or (self.organization is None) or (self.landscape is None):
            raise nk.UninitializedError("Instantiate Agents, Organization, Landscape before using them.")

        # associate free-roaming agents with the organization
        self.organization.hire_agents(self.agents)

        # form networks through which agents will each have peers to communicate with
        self.organization.form_networks()

        # Set the initial firm-wide random bitstring (length=N*P)
        initial_bstring = np.random.choice(2, self.n*self.p)
        self.organization.states[0,:] = initial_bstring

        # set states and calculate performances of P agents
        performances = self.landscape.phi(self.organization.states[0,:])
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

    def play(self):
        """
        This function contains the main loop of the world. It iterates
        the world from one time step to the next. 
        TODO: heavy changes here
        """
        self.initialize()
        for t in range(1,self.t):
            # at exactly t==TM, the memory fills (training ends) and climbing is done from scratch
            # NOTE: that is stupid
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
                agent.publish_social_bits(t)
                agent.observe_state()
            # nature archives the state 
            self.archive_state()


    def _archive_state(self):
        """MOVE THIS FUNCTIONALITY SOMEWHEREarchives state"""
        self.past_state.append(self.current_state.copy())
        self.past_sim.append(nk.similarity(self.current_state, self.p, self.n, self.nsoc))
        self.past_simb.append(nk.similarbits(self.current_state, self.p, self.n, self.nsoc))


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
        self.coord = kwargs['coord'] # coordination mode
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


    def initialize(self):
        """
        Initializes the parameters for the 0th step of the simulation.
        TODO: Don't Repeat Yourself!
        """
        
        # create landscape (task environment)
        self._create_environment()
        
        # create agent and organizations
        self._create_players()       

        # associate free agents with the organization
        self.organization.hire_agents(self.agents)

        # form networks through which agents will each have peers to communicate with
        self.organization.form_networks()

        # Set the initial firm-wide random bitstring (length=N*P)
        initial_bstring = np.random.choice(2, self.n*self.p)
        self.organization.states[0,:] = initial_bstring
        
        # set states and calculate performances of P agents
        for agent in self.agents:
            agent.current_state = initial_bstring
            agent.current_utility = agent.calculate_utility(initial_bstring.reshape(1,-1))[0]

        # save performances and synchronies to the organization archive
        self.organization.performances[0, :] = self.landscape.phi(self.organization.states[0,:])
        self.organization.synchronies[0] = nk.similarity(initial_bstring, self.p, self.n, self.nsoc)
        self.organization.current_gp_score = self.organization.calculate_gp_score(initial_bstring)

    def play(self):
        """
        This function contains the main loop of the world. It iterates
        the world from one time step to the next. 
        TODO: fix the case when no climbing happens
        TODO: fix multiple computations of synchrony and performance
        """

        for t in range(1,self.t):
            # Run the meeting and get the outcome
            meeting = self.organization.meeting(self.n, self.p, self.alt, self.prop, self.comp, self)
            meeting.run()

            # Update archives to include the latest information
            self.organization.states[t, :] = meeting.outcome
            self.organization.performances[t, :] = self.landscape.phi(meeting.outcome)
            self.organization.synchronies[t] = nk.similarity(meeting.outcome, self.p, self.n, self.nsoc)
            
            # Update current values
            self.organization.current_gp_score = self.organization.calculate_gp_score(meeting.outcome)
            for agent in self.agents:
                agent.current_state = meeting.outcome
                agent.current_utility = agent.calculate_utility(meeting.outcome.reshape(1,-1))[0]

            # Agents talk in a network with each other
            for agent in self.agents:
                agent.publish_social_bits()
            

    def _create_environment(self):
        """Creates the task environment."""
        self.landscape = Landscape(self.p, self.n, self.k, self.c, self.s, self.rho, self.normalize, self.precompute)
        self.landscape.generate()


    def _create_players(self):
        """Spawn main players: 1 organization and P agents"""
        
        params = (self.n, self.p, self.t, self.goals, self.net, self.degree, self.xi, self.coord, self)
        self.organization = Organization(*params)

        params = (self.n, self.p, self.nsoc, self.degree, self.tm, self.w, self.wf, self)
        self.agents = [Agent(i, *params) for i in range(self.p)]
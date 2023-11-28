'''CEO defintion'''
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
import nkpack as nk
if TYPE_CHECKING:
    from .nature import Nature
    from .agent import Agent

class Organization:
    '''
    Organization class defines tasks, hires people;
    has aggregation relation with Agent class;
    organizes meetings;

    It stores performance and synchrony histories obtained from Nature.
    '''
    def __init__(self, n:int, p:int, nature: Nature):
        # environment and user-input params:
        self.nature = nature
        self.n = n
        self.p = p
        self.agents: list[Agent] = None # "hire" all people from environment
        # histories:
        self.states = np.empty((nature.t, nature.n*nature.p), dtype=np.int8) # bitstrings history
        self.performances = np.empty((nature.t, nature.p), dtype=np.float32) # agents' performances 
        self.synchronies = np.empty(nature.t, dtype=np.float32) # synchrony measures history

    def form_networks(self) -> None:
        '''
        Generates the network structure for agents to communicate;
        it can be argued that a firm has the means to do that,
        e.g. through hiring, defining interfaces etc.
        
        '''
        if self.agents is None:
            raise nk.UninitializedError("Agents are not initialized yet.")

        peers_list = nk.generate_network(self.p, self.nature.degree, self.nature.xi, self.nature.net)

        for peers, agent in zip(peers_list, self.agents):
            agent.peers = peers

    def play(self):
        '''THIS MOVES TO NATURE The central method. Runs the lifetime simulation of the organization.'''
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
                agent.publish_social_bits(t)
                agent.observe_state()
            # nature archives the state 
            self.nature.archive_state()

    def observe_outcomes(self,tt):
        '''Receives performance report from the Nature'''
        self.perf_hist[tt] = self.nature.current_perf.mean()
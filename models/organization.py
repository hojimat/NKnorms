"""CEO defintion"""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
import nkpack as nk
if TYPE_CHECKING:
    from .nature import Nature
    from .agent import Agent

class Organization:
    """
    Organization class defines tasks, hires people;
    has aggregation relation with Agent class;
    organizes meetings;

    It stores performance and synchrony histories obtained from Nature.
    """
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

    def hire_agents(self, agents: list[Agent]) -> None:
        """Hires people by storing a reference to them"""
        self.agents = agents
        for agent in agents:
            agent.organization = self

    def form_networks(self) -> None:
        """
        Generates the network structure for agents to communicate;
        it can be argued that a firm has the means to do that,
        e.g. through hiring, defining interfaces etc.
        
        """
        if self.agents is None:
            raise nk.UninitializedError("Agents are not initialized yet.")

        peers_list = nk.generate_network(self.p, self.nature.degree, self.nature.xi, self.nature.net)

        for peers, agent in zip(peers_list, self.agents):
            agent.peers = peers

    
    def calculate_goals(self, bstring: NDArray[np.int8]):
        """
        Calculates the satisfaction of two goals: 
        overall performance and synchrony.

        Args:

        Returns:
            Saves to self.goals

        """

    def plan_meetings(self) -> None:
        """
        Generates the meeting structure to make decisions
        at each step; it can be argued that a firm
        has the means to do that, e.g. through organization design etc.
        
        """
        if self.agents is None:
            raise nk.UninitializedError("Agents are not initialized yet.")

    def observe_outcomes(self,tt):
        """Receives performance report from the Nature"""
        self.perf_hist[tt] = self.nature.current_perf.mean()
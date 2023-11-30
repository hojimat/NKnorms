"""CEO defintion"""
from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
import nkpack as nk
from .meeting import Meeting, DecentralizedMeeting, LateralMeeting, HierarchicalMeeting
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
    def __init__(self, n:int, p:int, t:int, goals:tuple[float], net:int, deg:int, xi:float, coord:int, nature: Nature):
        # environment and user-input params:
        self.nature = nature
        self.n = n
        self.p = p
        self.t = t
        self.goals = goals
        self.agents: list[Agent] = None # hire people from environment
        # search behavior
        self.degree = deg
        self.xi = xi
        self.network: str = ('random', 'line', 'cycle', 'ring', 'star')[net]
        self.meeting: type[Meeting] = (DecentralizedMeeting, LateralMeeting, HierarchicalMeeting)[coord]
        # histories:
        self.current_gp_score = 0.0
        self.states = np.empty((t, n*p), dtype=np.int8) # bitstrings history
        self.performances = np.empty((t, p), dtype=np.float32) # agents' performances 
        self.synchronies = np.empty(t, dtype=np.float32) # synchrony measures history

    def hire_agents(self, agents: list[Agent]) -> None:
        """Hires people by storing a reference to them and in them"""
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

        peers_list = nk.generate_network(self.p, self.degree, self.xi, self.network)

        for peers, agent in zip(peers_list, self.agents):
            agent.peers = peers

    def _archive_state(self):
        """MOVE THIS FUNCTIONALITY SOMEWHEREarchives state"""
        self.past_state.append(self.current_state.copy())
        self.past_sim.append(nk.similarity(self.current_state, self.p, self.n, self.nsoc))
        self.past_simb.append(nk.similarbits(self.current_state, self.p, self.n, self.nsoc))

    
    def calculate_gp_score(self, bstring: NDArray[np.int8]) -> float:
        """
        Uses goal programming to calculate the satisfaction
        of two goals: overall performance and synchrony.

        Args:
            bstring: 
        Returns:
            float
        """
        # get overall organizational performance
        # which is a mean of individual performances
        # of P agents
        performance = np.mean(self.nature.landscape.phi(bstring))
        # get synchrony measure
        synchrony = nk.similarity(bstring, self.p, self.n, self.n)
        # get the goal programming score
        gp_score = nk.gp_score(np.array([performance, synchrony]), np.array(self.goals), np.array([1,1]))
        
        return gp_score

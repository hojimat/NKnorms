from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

class Meeting(ABC):
    """
    This class handles the decision making in coordination
    This is a generalization of a simple climb up-or-down mechanism from earlier versions.
    It relates to an Organization class in a one-to-one manner.

    """
    
    def __init__(self, n:int, p:int, prop:int, comp:int):
        self.n = n
        self.p = p
        self.prop = prop
        self.comp = comp
        self.proposals: list[NDArray[np.int8]] = None
        self.composites: NDArray[np.int8] = None
        self.outcome: NDArray[np.int8] = None

    def screen(self, agents) -> None:
        """
        For every agent run agent.screen(by='utility') option
        """
        for agent in agents:
            agent.screen()

    def compose(self) -> None:
        """
        Once the proposals are made, the meeting host creates
        COMP number of compositions or N-sized bitstring combinations
        to get N*P-sized full bitstrings

        Args:
            proposals: numpy array of size PxPROPxN
        
        Returns (saves to self.composites):
            Numpy array of size COMPx(N*P), randomly combined from the input list's elements

        Examples:
                [[1,0],[0,0]], --->  [1,0,1,1],
                [[1,1],[0,1]]  --->  [0,0,1,1]

        """
        proposals = np.array(self.proposals, dtype=np.int8)

        # pick COMP random integers
        random_picks = np.random.choice(self.prop**self.p, self.comp)
        # convert the integers into triplet indices (x,y,z)
        picked_indices = [np.unravel_index(i, [prop]*p) for i in random_picks]
        # composite:
        composites = [ proposals[np.arange(self.p),index_,:].reshape(-1) for index_ in picked_indices ]
        
        self.composites = np.array(composites, dtype=np.int8)

    @abstractmethod
    def decide(self):
        """
        A virtual method in which agents come together with the meeting host and decide what to do.
        The decision is made differently depending on the meeting type. This method needs to be
        overloaded in the child classes.
        """

class HierarchicalMeeting(Meeting):
    """
    The hierarchical coordination:
    1) agents screen their proposals
    2) meeting host creates composites of their proposals
    3) organization CEO chooses the best solution according to goal programming,
    and output is written to self.outcome

    """

    def decide(self):
        pass

class LateralMeeting(Meeting):
    """
    The lateral communication:
    1) agents randomly come up with the proposals
    2) meeting host creates composites of their proposals
    3) agents vote/veto the solutions in a random order,
    and output is written to self.outcome

    """
    def screen(self, agents);
        for agent in agents:
            agent.screen(random=True)

    def decide(self):
        pass

class DecentralizedMeeting(Meeting):
    """
    The decentralized structure:
    1) agents screen their proposals and propose 1 bistring
    2) meeting host creates composites of their proposals
    3) agents vote for their own 1 bitstring (kinda redundant),
    and output is written to self.outcome

    """

    def __init__(self, n:int, p:int):
        super().__init__(n=n, p=p, prop=1, comp=1)

    def decide(self):
        self.outcome = self.composites[0, :]
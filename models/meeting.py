from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
if TYPE_CHECKING:
    from .nature import Nature

class Meeting(ABC):
    """
    This class handles the decision making in coordination
    This is a generalization of a simple climb up-or-down mechanism from earlier versions.
    It relates to an Organization class in a one-to-one manner.

    """
    
    def __init__(self, n:int, p:int, alt:int, prop:int, comp:int, nature:Nature):
        self.n = n
        self.p = p
        self.alt = alt
        self.prop = prop
        self.comp = comp
        self.nature = nature
        self.random = False
        self.final = False
        self.proposals: NDArray[np.int8] = None
        self.composites: NDArray[np.int8] = None
        self.outcome: NDArray[np.int8] = None

    def screen(self) -> None:
        """
        For every agent run agent.screen()
        """
        proposals = []
        for agent in self.nature.agents:
            proposal = agent.screen(self.alt, self.prop, self.random, self.final)
            proposals.append(proposal)

        self.proposals = np.array(proposals, dtype=np.int8)


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

        # pick COMP random integers
        random_picks = np.random.choice(self.prop**self.p, self.comp)
        # convert the integers into triplet indices (x,y,z)
        picked_indices = [np.unravel_index(i, [self.prop]*self.p) for i in random_picks]
        # composite:
        composites = [ self.proposals[np.arange(self.p),index_,:].reshape(-1) for index_ in picked_indices ]
        
        self.composites = np.array(composites, dtype=np.int8)

    @abstractmethod
    def decide(self):
        """
        A virtual method in which agents come together with the meeting host and decide what to do.
        The decision is made differently depending on the meeting type. This method needs to be
        overloaded in the child classes.
        """       

    def run(self) -> None:
        """
        Holds a meeting by running sequentially screen->compose->decide.
        then it exports the state to all agents and organization
        """
        self.screen()
        self.compose()
        self.decide()

class HierarchicalMeeting(Meeting):
    """
    The hierarchical coordination:
    1) agents screen their proposals
    2) meeting host creates composites of their proposals
    3) organization CEO chooses the best solution according to goal programming,
    and output is written to self.outcome

    """

    def decide(self):
        """
        Composites are put to check by organization one by one until
        found one better than status quo
        """
        
        for composite in self.composites:
            # utility of a composite and get current utility
            new_gp_score = self.nature.organization.calculate_gp_score(composite)
            old_gp_score = self.nature.organization.current_gp_score

            if new_gp_score >= old_gp_score:            
                self.outcome = composite
                break


class LateralMeeting(Meeting):
    """
    The lateral communication:
    1) agents randomly come up with the proposals
    2) meeting host creates composites of their proposals
    3) agents vote/veto the solutions in a random order,
    and output is written to self.outcome

    """
    def __init__(self, n:int, p:int, nature:Nature):
        super().__init__(n=n, p=p, alt=2, prop=1, comp=1, nature=nature)
        self.random = True

    def decide(self):
        """Composites are put to vote one by one until consensus is reached"""
        
        for composite in self.composites:
            # for each agent calculate utility of a composite and get current utility
            new_utilities = np.array([agent.calculate_utility(composite) for agent in self.nature.agents])
            old_utilities = np.array([agent.current_utility for agent in self.nature.agents])
            # vote True if decides to climb up
            votes = (new_utilities >= old_utilities)
            # if voted unanimously, climb up and end the loop
            if votes.all():            
                self.outcome = composite
                break



class DecentralizedMeeting(Meeting):
    """
    The decentralized structure:
    1) agents screen their proposals and propose 1 bistring
    already having compared it to the status quo and chosen
    the highest
    2) meeting host creates composites of their proposals
    3) agents vote for their own 1 bitstring (kinda redundant),
    and output is written to self.outcome

    """

    def __init__(self, n:int, p:int, alt:int, prop:int, comp:int, nature:Nature):
        super().__init__(n=n, p=p, alt=2, prop=1, comp=1, nature=nature)
        self.final = True

    def decide(self):
        """
        Every agent independently decides to climb up or not.
        No vote happens, everybody simply picks his own proposal
        """

        self.outcome = self.composites[0, :]
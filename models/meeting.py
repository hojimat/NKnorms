from abc import ABC, abstractclassmethod, abstractmethod
from typing import TYPE_CHECKING, Optional
import numpy as np
from numpy.typing import NDArray
if TYPE_CHECKING:
    from .nature import Nature

class Meeting(ABC):
    '''
    This class handles the decision making in coordination
    This is a generalization of a simple climb up-or-down mechanism from earlier versions.
    It relates to an Organization class in a one-to-one manner.

    '''
    
    def __init__(self, nature: Nature):
        self.nature = nature
        self.proposals: Optional[list[NDArray]] = None
        self.composites: Optional[list[NDArray]] = None
        self.outcome: Optional[NDArray] = None

    def screen(self, prop: int, method: str) -> NDArray:
        '''
        Ever agent must prepare to the meeting depending on the Meeting Type.
        By default, every agent screens ALT 1-bit deviations to their current bitstrings
        and picks top PROP proposals and brings them into the composition stage.
        
        In some meeting types this screening process may be random leaving the decision
        to the further stages. Then this method will be overloaded in those meetings.

        Returns:
            A list of P (one for each agent) numpy arrays of size N*PROP
        '''
        

    def compose(self, proposals: list[NDArray[np.int8]]) -> None:
        '''
        Once the proposals are made, the meeting host creates
        COMP number of compositions or N-sized bitstring combinations
        to get N*P-sized full bitstrings

        Args:
            proposals: list of P (one for each agent) numpy arrays of size NxPROP
        
        Returns (saves to self.composites):
            list of COMP numpy arrays of size N*P, randomly combined from the input list's elements

        '''
        
    @abstractmethod
    def decide(self):
        '''
        A virtual method in which agents come together with the meeting host and decide what to do.
        The decision is made differently depending on the meeting type. This method needs to be
        overloaded in the child classes.
        '''

class HierarchicalMeeting(Meeting):
    '''
    The hierarchical coordination:
    1) agents screen their proposals
    2) meeting host creates composites of their proposals
    3) organization CEO chooses the best solution according to goal programming
    4) output is written to self.outcome
    '''

class LateralMeeting(Meeting):
    '''
    The lateral communication:
    1) agents randomly come up with the proposals
    2) meeting host creates composites of their proposals
    3) agents vote/veto the solutions in a random order
    4) output is written to self.outcome
    '''

class DecentralizedMeeting(Meeting):
    '''
    The decentralized structure:
    1) agents screen their proposals and propose 1 bistring
    2) meeting host creates composites of their proposals
    3) agents vote for their own 1 bitstring (kinda redundant)
    4) output is written to self.outcome
    '''
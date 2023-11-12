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

    def screen(self) -> None:
        '''
        Ever agent must prepare to the meeting depending on the Meeting Type.
        By default, every agent screens ALT 1-bit deviations to their current bitstrings
        and picks top PROP proposals and brings them into the composition stage.
        
        In some meeting types this screening process may be random leaving the decision
        to the further stages. Then this method will be overloaded in those meetings.

        Returns (saves to self.proposals):
            list of P (one for each agent) numpy arrays of size N*PROP
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
        pass

class HierarchicalMeeting(Meeting):
    pass

class LateralMeeting(Meeting):
    pass

class DecentralizedMeeting(Meeting):
    pass
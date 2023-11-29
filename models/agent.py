"""Employee defintion"""
from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import numpy as np
from numpy.typing import NDArray
import nkpack as nk
if TYPE_CHECKING:
    from .nature import Nature

class Agent:
    """ Decides on tasks, interacts with peers; aggregation relation with Organization class."""
    def __init__(self, id_:int, n:int, p:int, nsoc:int, deg:int, tm:int, w:float, wf:float, nature:Nature):
        # adopt variables from the organization; not an inheritance.
        self.id_ = id_
        self.nature = nature
        self.organization = nature.organization
        self.n = n
        self.p = p
        self.nsoc = nsoc
        self.deg = deg
        self.tm = tm
        self.w = w
        self.wf = wf
        # current status
        self.current_state: NDArray[np.int8] = np.empty(self.n*self.p, dtype=np.int8)
        self.current_utility: float = 0.0
        self._current_performance: float = 0.0
        self._current_conformity: float = 0.0
        self._current_received_bits: list = [] # temporary storage
        # information about social interactions
        self._received_bits_memory: NDArray[np.int8] = np.zeros((tm, deg, nsoc), dtype=np.int8) # shared social bits memory
        self.peers : list[Agent] = None # agents, that this agent talks with in a network

    def publish_social_bits(self) -> None:
        """
        Shares social bits with agents connected in a network
        via invoking their receive() method

        """
        start = self.id_ * self.n
        end = start + self.nsoc
        current_social_bits = self.current_state[start:end]
        
        for peer in self.peers:
            peer.receive_social_bits(current_social_bits)

    def receive_social_bits(self, received_bits:NDArray[np.int8]) -> None:
        """
        Collects social bits to a temporary list, and if it contains
        DEG elements, appends it into the most recent row in the memory
        """
        # add received bits to the temporary buffer
        self._current_received_bits.append(received_bits)

        # if all DEG peers sent their bits, and the buffer is full
        if len(self._current_received_bits) == self.deg:
            # FIFO the received bits memory
            updated_memory = np.roll(self._received_bits_memory, shift=1, axis=0)
            updated_memory[0] = self._current_received_bits
            self._received_bits_memory = updated_memory
            # clear the buffer
            self._current_received_bits.clear()

    def observe_state(self):
        """observes the current bitstring choice by everyone"""
        self.current_state = self.nature.current_state.copy()
        self._current_performance = self.nature.current_perf

    def report_state(self):
        """reports state to nature"""
        n = self.n
        i = self.id
        self.nature.current_state[i*n:(i+1)*n] = self.current_state[i*n:(i+1)*n].copy()
        self.nature.current_soc[i] = self.phi_soc

    def calculate_utility(self, bstrings: NDArray[np.int8]) -> NDArray[np.float32]:
        """
        Calculate utility of a given array of bitstrings
        
        Args:
            bstrings: an array of bitstrings of interest; shape=(Any)x(N*P)
        Returns:
            a vector of utilities for each bitstring; shape=(Any)x1
        
        TODO: add an option to pass pre-computed performances for each bitstring,
        to have fewer lookups,
        """
        # define important indices
        start = self.id_ * self.n
        end_soc = start + self.nsoc

        # calculate performance for every alternative; shape=(Any)xP
        performances_all = np.apply_along_axis(self.nature.landscape.phi, axis=1, arr=bstrings)

        # calculate incentives (own performance + mean of other performances weighted sum)
        performances_own, performances_others = nk.decompose_performances(performances_all, self.id_)
        incentives: NDArray[np.float32] = self.wf * performances_own + (1-self.wf) * performances_others

        # calculate conformity for every alternative; shape=(Any)x1
        bstrings_social = bstrings[:, start:end_soc]
        conformities: NDArray[np.float32] = np.apply_along_axis(
            lambda bits: nk.calculate_frequency(bits, self._received_bits_memory), axis=1, arr=bstrings_social
        )
        
        # calculate utility for every alternative; shape=(Any)x1
        utilities = self.w * incentives + (1-self.w) * conformities

        return utilities


    def screen(self, alt:int, prop:int, random:bool=False, final:bool=False) -> NDArray[np.int8]:
        """
        Ever agent must prepare to the meeting depending on the Meeting Type.
        By default, every agent screens ALT 1-bit deviations to their current bitstrings
        and picks top PROP proposals and brings them into the composition stage.
        
        Args:
            alt:    number of alternatives to screen
            prop:   number of proposals to choose from the alternatives
            random: if true, the screening simply returns PROP random 1-bit deviations
            final:  final decision; if true, directly compares new utility of the highest
                    candidate to the status quo and returns whichever is highest
        Returns:
            numpy array of shape PROPxN
        """
        # define important indices
        start = self.id_ * self.n
        end = start + self.n

        # get alt 1bit deviations to the current bit string; shape=ALTx(N*P)
        alternatives = nk.get_1bit_deviations(self.current_state, self.n, self.id_, alt)

        # if screening method is random, don't proceed further
        # and just return the random PROPxN proposals
        if random:
            proposals = alternatives[:prop, start:end]
            return proposals
       
        # calculate utility for every alternative; shape=ALTx1
        utilities = self.calculate_utility(alternatives)
        
        # sort alternatives by utility (in descending order)
        sorted_indices = np.argsort(-utilities)
        alternatives = alternatives[sorted_indices]

        # identify proposals; shape=PROPxN
        proposals = alternatives[:prop, start:end]

        # if final option is on, make final screening decision
        # here and now: compare best alternative utility to 
        # the current utility, and if it is not better,
        # or equal to, simply return the current state
        # Note: reshape(1,-1) converts a 1d vector current_state
        # into a 2d array with a single row, to match the format
        # of proposals
        
        if final:
            if np.max(utilities) < self.current_utility:
                proposals = self.current_state.reshape(1, -1)[:, start:end]

        return proposals
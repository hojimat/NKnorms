'''Employee defintion'''
from __future__ import annotations
from typing import TYPE_CHECKING, Optional
import numpy as np
from numpy.typing import NDArray
import nkpack as nk
if TYPE_CHECKING:
    from .nature import Nature

class Agent:
    ''' Decides on tasks, interacts with peers; aggregation relation with Organization class.'''
    def __init__(self, id_: int, nature: Nature):
        # adopt variables from the organization; not an inheritance.
        self.id_ = id_
        self.nature = nature
        self.organization = nature.organization
        self.n = self.organization.n
        self.p = self.organization.p
        # current status
        self.current_state: NDArray = np.empty(self.n*self.p, dtype=np.int8)
        self.current_perf: float = 0.0
        self.current_soc: float = np.repeat(-1, nature.nsoc)
        # information about social interactions
        self.soc_memory = np.repeat(-1, 2*nature.nsoc).reshape(2, nature.nsoc) # social storage matrix
        self.peers : Optional[list] = None # agents this agent talks with in a network

    def perform_climb(self,lrn=False,soc=False):
        '''The central method. Contains the main decision process of the agent'''
        # get attributes as local variables
        w = self.w.copy()
        wf = self.wf.copy()

        # get "before" parameters
        bit0 = self.current_state.copy() # current bitstring
        idx0 = nk.get_index(bit0,self.id,self.n) # location of ^
        all_phis = list(self.current_perf) # vector of performances of everybody
        my_phi = all_phis.pop(self.id) # get own perf
        other_phis = np.mean(all_phis) # get rest perfs
        phi0 = wf[0] * my_phi + wf[1] * other_phis # calculate earnings
        #beta0 = self.current_betas[idx0,:] # current beliefs
        soc0 = self.current_soc # current social bits (subset of bit0) 

        # get "after" parameters
        bit1 = nk.random_neighbour(bit0,self.id,self.n) # candidate bitstring
        idx1 = nk.get_index(bit1,self.id,self.n) # location of ^
        my_phi, other_phis = self.nature.phi(self.id,bit1,self.eps) # tuple of own perf and mean of others
        phi1 = wf[0] * my_phi + wf[1] * other_phis # calc potential earnings
        #beta1 = self.current_betas[idx1,:] # calc potential updated beliefs
        soc1 = nk.extract_soc(bit1,self.id,self.n,self.nsoc) # potential social bits (subset of bit1)

        # calculate mean betas
        #mbeta0 = nk.beta_mean(*beta0)
        #mbeta1 = nk.beta_mean(*beta1)

        # calculate soc frequency
        fsoc0 = nk.calculate_freq(soc0,self.soc_memory)
        fsoc1 = nk.calculate_freq(soc1,self.soc_memory)

        # calculate utility 
        util0 = w[0] * phi0 + w[1] * fsoc0
        util1 = w[0] * phi1 + w[1] * fsoc1

        # the central decision to climb or stay
        if util1 > util0:
            self.current_state = bit1
            self.phi_soc = fsoc1
        else:
            self.phi_soc = fsoc0

        # update beliefs (betas) 
        #self.current_betas[idx1,int(phi1<phi0)] += 1

    def share_soc(self,tt):
        '''shares social bits with agents in a clique'''
        # get own social bits
        idd = self.id
        n = self.n
        p = self.p
        nsoc = self.nsoc
        clique = self.peers
        current = self.current_state.copy()
        current_soc = nk.extract_soc(current,idd,n,nsoc)
        noisy_soc = nk.with_noise(current_soc,self.eta)
        
        # share social bits with the clique
        for i in range(p):
            connect = np.random.choice(2,p=[1-clique[i], clique[i]])
            if connect:
                current_memory = self.employer.agents[i].soc_memory
                self.employer.agents[i].soc_memory = np.vstack((current_memory, noisy_soc))
                self.employer.agents[i].nsoc_added[tt] += 1
        
        # update own social bit attribute for future references
        self.current_soc = current_soc



    def forget_soc(self,tt):
        '''forgets social bits'''
        tm = self.tm
        sadd = self.nsoc_added
        if tt >= tm:
            self.soc_memory = self.soc_memory[sadd[tt-tm]:,:]

    def observe_state(self):
        '''observes the current bitstring choice by everyone'''
        self.current_state = self.nature.current_state.copy()
        self.current_perf = self.nature.current_perf


    def report_state(self):
        '''reports state to nature'''
        n = self.n
        i = self.id
        self.nature.current_state[i*n:(i+1)*n] = self.current_state[i*n:(i+1)*n].copy()
        self.nature.current_soc[i] = self.phi_soc


    def screen(self, alt: int, prop: int, method: str) -> NDArray:
        '''
        Ever agent must prepare to the meeting depending on the Meeting Type.
        By default, every agent screens ALT 1-bit deviations to their current bitstrings
        and picks top PROP proposals and brings them into the composition stage.
        
        Args:
            alt: number of alternatives to screen
            prop: number of proposals to choose from the alternatives
            method: screening method (by utility, by performance, randomly)
        Returns:
            numpy array of size N*PROP
        '''

        # get alt 1bit deviations to the current bit string
        alternatives = nk.get_1bit_deviations(self.current_state, self.n, self.id_, alt)

        # calculate utilities for all deviations
        
        # get "before" parameters
        idx0 = nk.get_index(self.current_state, self.id,s elf.n) # location of ^
        all_phis = list(self.current_perf) # vector of performances of everybody
        my_phi = all_phis.pop(self.id) # get own perf
        other_phis = np.mean(all_phis) # get rest perfs
        phi0 = wf[0] * my_phi + wf[1] * other_phis # calculate earnings
        #beta0 = self.current_betas[idx0,:] # current beliefs
        soc0 = self.current_soc # current social bits (subset of bit0) 

        # get "after" paarameters
        bit1 = nk.random_neighbour(self.current_state,self.id,self.n) # candidate bitstring
        idx1 = nk.get_index(bit1,self.id,self.n) # location of ^
        my_phi, other_phis = self.nature.phi(self.id,bit1,self.eps) # tuple of own perf and mean of others
        phi1 = wf[0] * my_phi + wf[1] * other_phis # calc potential earnings
        #beta1 = self.current_betas[idx1,:] # calc potential updated beliefs
        soc1 = nk.extract_soc(bit1,self.id,self.n,self.nsoc) # potential social bits (subset of bit1)

        # calculate mean betas
        #mbeta0 = nk.beta_mean(*beta0)
        #mbeta1 = nk.beta_mean(*beta1)

        # calculate soc frequency
        fsoc0 = nk.calculate_freq(soc0,self.soc_memory)
        fsoc1 = nk.calculate_freq(soc1,self.soc_memory)

        # calculate utility 
        util0 = w[0] * phi0 + w[1] * fsoc0
        util1 = w[0] * phi1 + w[1] * fsoc1

        # the central decision to climb or stay
        if util1 > util0:
            self.current_state = bit1
            self.phi_soc = fsoc1
        else:
            self.phi_soc = fsoc0

        # update beliefs (betas) 
        #self.current_betas[idx1,int(phi1<phi0)] += 1


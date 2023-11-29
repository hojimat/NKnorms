import numpy as np
import NKPackage as nk

class Organization:
	""" Defines tasks, hires people; aggregation relation with Agent class """
    def __init__(self,pop,n,npub,k,degree,t,th,tm):
        self.pop = pop
		self.n = n
		self.npub = npub
		self.k = k
		self.degree = degree
		self.t = t
		self.th = th
		self.tm = tm
		self.performance_history = np.zeros(t,dtype=float) # organizations call past history, individuals --- memory

	def hire_people(self):
		for i in range(self.pop):
			Agent(employer=self)

    def allocate_tasks(self):
		""" in the future, firm can allocate tasks, at the moment agents have separate landscapes """
        pass

	def form_cliques(self):
		clique = nk.generate_network(self.pop,self.degree,"cycle")
		return clique

	def play(self):
		""" single pass through a lifetime """
		for t in range(self.t):
			if t==0: # initialize at the beginning
				for agent in Agent.agents:
					agent.initialize()
			elif t>0 and t<TH: # adapt some time before accumulating social norms
				for agent in Agent.agents:
					agent.solo_climb()
					agent.share_pub()
			else: # full-on social climbing
				for agent in Agent.agents:
					agent.social_climb()
					agent.share_pub()
		pass


class Agent:
	""" Main player; works for Organization, adopts parameters from it """
	agents = [] # agents list, indexed from 0
    def __init__(self,employer):
		# adopt variables from the organization; NOT inherit.
        self.employer = employer
        self.n = employer.n
		self.npub = employer.npub
        self.k = employer.k
		self.degree = employer.degree
		self.t = employer.t
		self.th = employer.th
		# current status; realistic
        self.current_choice = np.zeros(self.n,dtype=int)
        self.current_betas = np.ones((2**self.n,2),dtype=int) # agents forget details, only remember the attitude
		self.current_util = 0.0
		# memory excludes the current period
		self.choice_memory = np.zeros((self.t,self.n),dtype=int)
		self.util_memory = np.zeros(self.t,dtype=float)
        self.pub_memory = [[] for _ in range(self.th)]
		# landscape architecture
        self.inmat = nk.interaction_matrix(self.n,self.k)
        self.conmat = nk.contrib_define(self.n,self.k)
        self.landscape = nk.perf_calc(self.inmat, self.conmat)
		self.global_max = np.max(self.landscape)
		Agent.agents.append(self) # add an agent to agents list

	def initialize(self):
		self.current_util = self.landscape[0]
		self.current_betas[0,0] += 1
		self.choice_memory[0] = self.current_choice
		self.util_memory[0] = self.current_util

	def solo_climb(self):
		# define status quo parameters
		idx0 = nk.binx(self.current_choice)
		phi0 = self.landscape[idx0]
		beta0 = self.current_betas[idx0,:]
		# define the candidate parameters
		idx1 = nk.get_neighbours(self.current_choice,1)[1] # nay=1 is set to enforce local climb
		phi1 = self.landscape[idx1]
		beta1 = self.current_betas[idx1,:]
		# calculate mean betas
		mbeta0 = nk.beta_mean(*beta0)
		mbeta1 = nk.beta_mean(*beta1)
		# climb decision here
		if phi1+mbeta1>=phi0+mbeta0:
			self.current_choice = 
			self.current_util = 
		else:
			pass
		# beta updating here
		if phi1>=phi0:
			pass
		else
			pass
		return	

	def social_climb(self):
		""" No modularity here is intentional, i.e. the code from solo_climb is repeated because writing a function for it does not save space --- all of the variables have to be assigned anyway. """

		pass

	def share_pub(self):
		pass


 
"""
            if t > self.td + 100 and t > trnd + 100: # no interference before social interactions
                if rnd is False: # active checking phase
                    if self.check_slope(t):
                        rnd = True # start RnD period
                        trnd = t # set starting time
                        self.define_artifacts() # set artifacts for jumps
                        for agent in Agent.lst:
                            agent.adopt_artifacts() # adopt artifacts for jumps
                            agent.report_state()
                        for agent in Agent.lst:
                            agent.observe_outcomes()

                else: # RnD phase
                    if t == trnd + self.tr:
                        before = np.mean(self.perf_hist[(trnd-10):trnd])
                        after = np.mean(self.perf_hist[(t-10):t])
                        if before > after: # if RnD did not improve
                            self.nature.current_state = self.nature.past_state[trnd-5].copy() # then jump back
                            self.nature.current_perf = self.nature.past_perf[trnd-5].copy()
                            for agent in Agent.lst:
                                agent.observe_outcomes()
                        rnd = False # end RnD period

            if t in self.tj:
                self.define_artifacts()
                for agent in self.agents:
                    print(agent.pub_memory)
                    agent.adopt_artifacts()
                    agent.report_state()
                for agent in self.agents:
                    agent.observe_state()
    def define_artifacts(self):
        n = self.n
        nrev = self.nrev
        nart = self.nart
        p = self.p
 
        artx = nk.artify(n,p,nart)

        # Random revelation model
        bitx = nk.pick(2**(n*p),nrev)
        bitx = [nk.binx(int(z),n*p) for z in bitx]
        self.reveals += bitx
        bitx = bitx + self.nature.past_state
        bitx = self.reveals + bitx
        
        tmp = [self.nature.phi(myid=None,x=z).mean() for z in bitx]
        tmp = np.argmax(tmp)
        reveal = bitx[tmp]

        artifact = [list(zip(artx[z],[reveal[i] for i in artx[z]])) for z in range(p)]
        
        # Neighbor search model
        #bitx = nk.get_neighbours(self.nature.current_state,n*p)[0]

        #reveal = [np.random.choice(2) for z in tmp]
        #artifact = [list(zip(artx[z],[(1-reveal[i]) for i in artx[z]])) for z in range(p)]
        #artifact = [list(zip(artx[z],[abs((np.random.uniform() > 1.0) - reveal[i]) for i in artx[z]])) for z in range(p)]

        # Explorative jumps model
        #tmp = self.nature.current_state.copy()
        #reveal = [(1-z) for z in tmp]
        #artifact = [list(zip(artx[z],[reveal[i] for i in artx[z]])) for z in range(p)]

        for agent in self.agents:
            agent.artifact = artifact[agent.id]
            
    def adopt_artifacts(self):
        artifact = self.artifact
        tmp = self.current_state.copy()
        for z in artifact:
            tmp[z[0]] = z[1]
        phi1 = self.nature.phi(self.id,tmp)
        if 1==1:#phi1 >= self.ubar*0.9:# and phi1 >= self.current_perf:
            self.current_state = tmp
        
"""

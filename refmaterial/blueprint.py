from setup import Organization, Agent
from matplotlib import pyplot as plt

"""
Syntax guidelines:
	- attributes and variables are lowercase
	- constants are UPPERCASE
	- don't use camelCase
"""

POP = 4
N = 6
NPUB = 3
K = 3
DEG = 1
T = 100
TH = 101
TM = 5

# create an organization with set parameters
firm = Organization(pop=POP,n=N,npub=NPUB,k=K,degree=DEG,t=T,th=TH,tm=TM)

# hire people = create agents with set parameters
firm.hire_people() # NOTE: maybe move POP to the arguments here.
# form cliques = set network structure (who shares with whom)
firm.form_cliques() # NOTE: maybe move DEG here to the arguments.
# 'play' the game, i.e. run the single lifetime
firm.play()

#print(firm.performance_history)
plt.plot(list(range(T)),Agent.registry[0].util_memory)
plt.show()


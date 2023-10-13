from setup import Organization, Agent
from matplotlib import pyplot as plt

'''
Syntax guidelines:
	- attributes and variables are lowercase
	- constants are UPPERCASE
	- don't use camelCase
'''

P = 5
N = 4
K = 1
C = 2
S = 2
NPUB = 4
DEG = 1
NART = 6
NREV = 30
T = 200
TD = 100
TM = 25
TJ = 1400
W = [0.33,0.33,0.33]

# create an organization with set parameters
firm = Organization(p=P,n=N,npub=NPUB,nart=NART,nrev=NREV,k=K,degree=DEG,c=C,s=S,t=T,td=TD,tm=TM,tj=TJ,w=W)

firm.define_tasks()
firm.hire_people() 
firm.form_cliques() 
firm.play()

plt.plot(list(range(T)),firm.performance_history/POP)
plt.show()
#print(firm.performance_history)

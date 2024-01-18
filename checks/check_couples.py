import sys
sys.path.append("/home/ravshan/Seafile/research/codes/NKnorms/")
import nkpack as nk

# set params
K=1
C=1
S=2

# generate couplings
couples = nk.generate_couples(5, 2, 'random')
peers = nk.generate_network(5, 2, shape='random')

print(couples)
import sys
sys.path.append("/home/ravshan/Seafile/research/codes/NKnorms/")
import models

# set params
K=3
C=2
S=3

# set up the world
landscape = models.Landscape(5,4,K,C,S,0.0,False,False)
landscape._generate_interaction_matrix()


# check different values
print(landscape.interaction_matrix)
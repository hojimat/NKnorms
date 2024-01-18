import sys
sys.path.append("/home/ravshan/Seafile/research/codes/NKnorms/")
import models

# set params
K=1
C=1
S=2

# set up the world
landscape = models.Landscape(5,4,K,C,S,0.0,False,False)
landscape._generate_interaction_matrix()


# check different values
print(landscape.interaction_matrix)
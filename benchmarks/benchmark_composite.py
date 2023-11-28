import numpy as np
import itertools

P = 5
PROP = 3
N = 4
COMP = 6

# get proposals
proposals = [np.random.choice(2, (PROP,N))]*P
proposals = np.array(proposals)

# randomly pick combination indices
all_indices = itertools.product(range(PROP), repeat=P)
random_picks = np.random.choice(PROP**P, COMP)
picked_indices = [indices for i,indices in enumerate(all_indices) if i in random_picks]

# composite:
composites = [proposals[range(P), idx, :].reshape(-1) for idx in picked_indices]
composites = np.array(composites)
print(composites)
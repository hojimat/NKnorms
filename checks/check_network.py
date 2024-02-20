import sys
sys.path.append('..')
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import nkpack as nk
from models import Landscape

P,N,K,C,S=(5, 4, 2, 1, 2)
landscape = Landscape(P,N,K,C,S,0.3, False, False)
landscape.generate()
adjacency_matrix = landscape.interaction_matrix - np.eye(N*P, dtype=int)


#adjacency_matrix = nk.generate_network(30, shape="star")

# Create a directed graph from the adjacency matrix
G = nx.DiGraph(adjacency_matrix)

# Draw the graph
#pos = nx.kamada_kawai_layout(G)  # positions for all nodes
#pos = nx.spring_layout(G)  # positions for all nodes
pos = nx.circular_layout(G)  # positions for all nodes
#pos = nx.spectral_layout(G)  # positions for all nodes
#pos = nx.fruchterman_reingold_layout(G)  # positions for all nodes

nx.draw(G, pos, with_labels=False, node_color='skyblue', node_size=700, edge_color='k', linewidths=1, font_size=15, arrows=True)
plt.show()


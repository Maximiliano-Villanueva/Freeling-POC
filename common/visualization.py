"""
This file contains all tools needed for visual representation
"""
# Standard imports
import os

from typing import Optional

# Third party imports
import matplotlib.pyplot as plt
import networkx as nx

# Local imports
from grap_builder.module import KnowledgeGraph


def plot_directed_graph(graph: KnowledgeGraph,
                        save_path: Optional[str]):
    """ Draw the graph using Matplotlib. """

    G = graph.G
    # If the graph isn't directed but you want it to be, convert it
    if not isinstance(G, nx.DiGraph):
        G = G.to_directed()

    # Create the layout for the graph
    pos = nx.spring_layout(G)

    # Draw the graph using the layout
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, 
            edge_cmap=plt.cm.Blues, font_size=10, arrows=True)

    # Attempt to save the figure
    if save_path:
        plt.savefig(save_path)
        # Close the figure no matter what
        plt.close()

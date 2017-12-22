from __future__ import absolute_import

import networkx as nx
import numpy as np
from matplotlib import pyplot
import seaborn as sns


class VisualizeHeatmap():

    @staticmethod
    def visualize_input_heatmap(x, y, cmap="Reds"):

        y_asInt = y.astype(int)  # cast y as int

        # multiplying the matrix with its transpose to get the co-occurrence matrix
        co_occurrence_matrix = y_asInt.T.dot(y_asInt)

        g = nx.from_scipy_sparse_matrix(co_occurrence_matrix)  # making graph from the sparse co-occurrence matrix

        adjacency_matrix = nx.to_numpy_matrix(g, dtype=np.bool, nodelist=None)  # converting to adjacency matrix

        # seaborn heatmap
        pyplot.figure()
        sns.heatmap(adjacency_matrix, cmap=cmap)

        # Statistical clustering with Seaborn - clustered heatmap
        pyplot.figure()

        sns.set(color_codes=True)
        sns.clustermap(adjacency_matrix, cmap=cmap, robust=True)

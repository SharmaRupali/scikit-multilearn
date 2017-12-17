from __future__ import absolute_import

import numpy as np
import networkx as nx
import scipy.sparse as sp
import matplotlib.pyplot as plt
import math


class VisualizeNetwork():

    @staticmethod
    def visualize_input_network(x, y, labels=None, k=2, iterations=50, n_size=25, n_e_ratiotype="linear",
                                linear_ratio_val=0.5, log_ratio_base_val=1.2, cmap_node="copper",
                                cmap_edge=plt.cm.BrBG):

        n_classes = len(y.toarray()[0])  # number of labels taken from the size of y

        y = sp.csr_matrix(y)
        y_asInt = y.astype(int)  # cast y as int
        co_occurrence_matrix = y_asInt.T.dot(y_asInt)  # multiplying the matrix with its transpose to get the co-occurrence matrix

        g = nx.from_scipy_sparse_matrix(co_occurrence_matrix)  # making graph from the sparse co-occurrence matrix

        adjacency_matrix = nx.to_numpy_matrix(g, dtype=np.bool, nodelist=None)  # converting to adjacency matrix

        H = nx.from_numpy_matrix(np.array(adjacency_matrix))  # creating network structure
        G = H.to_undirected()  # undirected graph - excluding edge repetitions

        spring_pos = nx.spring_layout(G, k, iterations)  # positioning nodes using Fruchterman-Reingold force-directed algorithm

        # taking the weights of the edges from the graph (list(G.edges_iter(data='weight')))
        weights = []
        for u, v, d in G.edges(data=True):
            weights.append(d['weight'])

        weights_normalized = [(i / max(weights)) for i in weights]  # normalizing the weights for better visualization

        # naming the labels if no list is provided
        if labels is None:
            labels = {}
            for i in range(0, n_classes):
                labels[i] = "Label " + str(i)

        #drawing the network depending on the ratio type selected (linear or logarithmic), main difference lies in the proportions (sizing) of
        #node-edge relationships
        if n_e_ratiotype is "log":
            if log_ratio_base_val is not 1:
                ratio_log = math.log(1 / min(weights_normalized), log_ratio_base_val)
                weights_normalized_log = [i * ratio_log for i in weights_normalized]
                nx.draw(G, labels=labels, pos=spring_pos,
                        node_size=[v * n_size for v in co_occurrence_matrix.diagonal()], node_color=range(n_classes),
                        cmap=cmap_node, edge_color=weights_normalized_log, width=weights_normalized_log,
                        edge_cmap=cmap_edge)
        else:
            ratio_li = linear_ratio_val / min(weights_normalized)
            weights_normalized_li = [i * ratio_li for i in weights_normalized]
            nx.draw(G, labels=labels, pos=spring_pos, node_size=[v * n_size for v in co_occurrence_matrix.diagonal()],
                    node_color=range(n_classes), cmap=cmap_node, edge_color=weights_normalized_li,
                    width=weights_normalized_li, edge_cmap=cmap_edge)

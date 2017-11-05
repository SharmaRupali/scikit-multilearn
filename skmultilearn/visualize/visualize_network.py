import numpy as np
import networkx as nx
import scipy.sparse as sp


class VisualizeNetwork():

    def visualize_input_label_network(x, y, n_classes, cmap_node, cmap_edge):

        y = sp.csr_matrix(y)
        y_asInt = y.astype(int)  # cast y as int
        co_occurrence_matrix = y_asInt.T.dot(y_asInt)  # multiplying the matrix with it's transpose to get the co-occurrence matrix

        g = nx.from_scipy_sparse_matrix(co_occurrence_matrix)  # making graph from the sparse co-occurrence matrix

        adjacency_matrix = nx.to_numpy_matrix(g, dtype=np.bool, nodelist=None)  # converting to adjacency matrix

        H = nx.from_numpy_matrix(np.array(adjacency_matrix))  # creating network structure
        G = H.to_undirected()  # undirected graph - excluding edge repetitions

        spring_pos = nx.spring_layout(G, k=2.0, iterations=50)  # spring_layput positions nodes using Fruchterman-Reingold force-directed algorithm

        # taking the weights of the edges from the graph (list(G.edges_iter(data='weight')) - see the list of edges-weights)
        weights = []
        for u, v, d in G.edges(data=True):
            weights.append(d['weight'])

        weights_normalized = [(i / max(weights)) * 20 for i in weights]  # normalizing the weights for better visualization

        nx.draw(G, pos=spring_pos, node_size=[v * 25 for v in co_occurrence_matrix.diagonal()], node_color=range(n_classes),
                cmap=cmap_node, edge_color=weights_normalized, width=weights_normalized, edge_cmap=cmap_edge)

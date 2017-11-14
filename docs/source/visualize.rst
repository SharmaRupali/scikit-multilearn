.. _visualize:

Developing a visualization method
=================================

In order to visualize the data, different methods of data visualization will be implemented. The first one to be worked on is the
Network Visualization, followed by the Cluster-Pie Visualization.

Network Visualization
---------------------

One of the selected approaches to data visualization is a graph like structure - network, more precisely.
In the simplest version of a network, there are 2 main aspects to take care of: nodes, edges. Going deeper for network
generation purposes, we may want to look into the different possibilities of representing the attributes. Here, we are considering
to represent 4 characteristics: color of the nodes, size of the nodes, width of the edges, and, color of the edges.

All these are represented in the following way:
    * color of the nodes (node_color): represented by the number of different labels.
    * size of the nodes (node_size): represented by the number of occurrences of a given label.
    * width of the edges (width): represented by the weights of the edges
    * color of the edges (edge_color): same as the width: represented by the weights of the edges

All the aforementioned characteristics will be defined on the basis of the adjacency matrix which we will be generated from the
co-occurrence matrix of labels from the given dataset.

Using Predefined Libraries
--------------------------

The basis of this Network Visualization module are a few already implemented python modules including
scikit-learn and scikit-multilearn modules. We are importing the following modules and how:

.. code-block:: python

    import numpy as np
    import networkx as nx
    import scipy.sparse as sp
    import matplotlib.pyplot as plt
    import math

Generating Data for Experimentation
-----------------------------------

Let's start by generating some exemplary data:
We'll generate a dataset with a defined number of classes, and sparse matrices

.. code-block:: python

    from sklearn.datasets import make_multilabel_classification

    x, y = make_multilabel_classification(sparse = True, n_classes = 7, return_indicator = 'sparse', allow_unlabeled= False)


Defining Parameters and Defaults
--------------------------------

The parameters that are needed for our function are:
    * x: {array-like, sparse matrix}, shape (n_samples, n_features)
    * y: numpy array, shape (n_samples, ) - Target values: labels
    * labels (Default = None): labels to be shown on the network, if not specified any, uses defaults
    * k (Default = 2): optimal distance between nodes
    * iterations (Default = 50): number of iterations of spring-force relaxation
    * n_size (Default = 25): ratio of node size (greater this value, larger the nodes, but always in proportion with other nodes)
    * n_e_ratiotype (Default = "linear"): type of calculation (linear or logarithmic) for ratios to be applied on the proportions of nodes and edges
    * linear_ratio_val (Default = 0.5): value for calculating the ratio when "n_e_ratiotype" is chosen to be "linear"
    * log_ratio_base_val (Default = 1.2): value of the logarithmic base for calculating the ratio when "n_e_ratiotype" is chosen to be "log"
    * cmap_node (Default = "copper"): set of predefined color palette values from the colormap of networkx (Node Colormap)
    * cmap_edge (Default = plt.cm.BrBG): set of predefined color palette values from the colormap of networkx (Edge Colormap)

Getting Started with the Data
-----------------------------

First of all, we need to see how many different classes of labels we are dealing with, which we take from the input data:

.. code-block:: python

    n_classes = len(y.toarray()[0])

In order to get our function working, we need to check if the data we are using is in the appropriate form that the function
accepts, here, we have it because in the exemplary data, we have set the sparse option to be true, but when we are not sure about it,
we need to transform the generated data into the appropriate form, i.e. sparse matrices and matrix elements as integers:

.. code-block:: python

    y = sp.csr_matrix(y)
    y_asInt = y.astype(int)


Function Description
--------------------

After having converted the data into the appropriate form, we need to get the co-occurrence matrix from the data, as it's the base
of our next steps. The co-occurrence matrix is a matrix containing the information about the labels occurring at the same time.

.. code-block:: python

    co_occurrence_matrix = y_asInt.T.dot(y_asInt)

Using the co-occurrence matrix that we just created, we'll from the base graph structure and the adjacency matrix to take the weights
from (for further use):

.. code-block:: python

    g = nx.from_scipy_sparse_matrix(co_occurrence_matrix)
    adjacency_matrix = nx.to_numpy_matrix(g, dtype=np.bool, nodelist=None)

From the adjacency matrix we will generate the network with all the nodes and edges, it will be an undirected graph as the adjacency
matrix is symmetric, so we will take an edge once:

.. code-block:: python

    H = nx.from_numpy_matrix(np.array(adjacency_matrix))
    G = H.to_undirected()

In order to draw the network nicely and not to overlap the nodes, we need to determine some layout for the positioning of the nodes, for which
we will see use one of the predefined layouts from netwrokx. There are different layouts implemented in networkx: circular_layout, random_layout,
spring_layout, spectral_layout, and a few more. We will be using the **spring_layout** that uses the Fruchterman-Reingold force-directed
algorithm to position the nodes. The reason behind using the spring_layout specifically is that no other layout implemented in networkx works as efficiently
for our purpose: the circular_layout just positions the nodes on a circle; the random_layout positions the nodes uniformly at random in the unit square;
and, spectral_layout positions nodes using the eigenvectors of the graph Laplacian

.. code-block:: python

    spring_pos = nx.spring_layout(G, k, iterations)

The edges are dependent on the weights, that we will take from the graph and save as list to further use them for drawing purposes:

.. code-block:: python

    weights = []
    for u,v,d in G.edges(data=True):
        weights.append(d['weight'])

We need to normalize the weights so that all the wights are in propotion and we won't have inconsistencies while drawing the network

.. code-block:: python

    weights_normalized = [(i/max(weights)) for i in weights]

In order to label the nodes in the network, if no list is provided, we need to generate a list of labels:

.. code-block:: python

    if labels is None:
        labels = {}
        for i in range(0, n_classes):
            labels[i] = "Label " + str(i)

After having completed all the aforementioned steps and having gotten the sizes of the nodes, and the normalized weights of the edges,
we need to define the ratios for the node and edges size proportions, for which we need to choose a method for the calculation of ratios.
Two methods have been defined to choose from: **linear** and **logarithmic**.

The linear method simply calculates the ratio by dividing the defined ratio value (parameter) by the minimum value of the normalized weights.
Whereas, the logarithmic method calculates the ratio by taking the logarithm of the base provided (parameter) of the inverse of the minimum value
of the normalized weights. This is followed by the application of the ratios to the list of normalized weights and drawing the network with all the
specified and calculated parameters. We have used the networkx.draw() method for drawing purposes:

.. code-block:: python

    if n_e_ratiotype is "log":
        if log_ratio_base_val is not 1:
            ratio_log = math.log(1 / min(weights_normalized), log_ratio_base_val)
            weights_normalized_log = [i * ratio_log for i in weights_normalized]
            nx.draw(G, labels=labels, pos=spring_pos, node_size=[v * n_size for v in co_occurrence_matrix.diagonal()], node_color=range(n_classes), cmap=cmap_node, edge_color=weights_normalized_log, width=weights_normalized_log, edge_cmap=cmap_edge)
    else:
        ratio_li = linear_ratio_val / min(weights_normalized)
        weights_normalized_li = [i * ratio_li for i in weights_normalized]
        nx.draw(G, labels=labels, pos=spring_pos, node_size=[v * n_size for v in co_occurrence_matrix.diagonal()], node_color=range(n_classes), cmap=cmap_node, edge_color=weights_normalized_li, width=weights_normalized_li, edge_cmap=cmap_edge)


We can notice, in the parameters of draw(), the specifications of the four characteristics that we had considered to be our representation essentials:
    * node_size: taken from the primary diagonal of the co-occurrence matrix
    * node_color: taken from the number of classes mentioned
    * width: taken from the weights of the edges after normalization
    * edge_color: same as the width, taken from the weights of the edges after normalization
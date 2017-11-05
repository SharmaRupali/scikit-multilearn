.. _visualize:

Developing a visualization method
=================================

In order to visualize the data, different forms of data visualization will be implemented.

Network Visualization
---------------------

One of the selected approaches do data visualization is a graph like structure - network, more precisely.
In the simplest version of a network, there are 2 main aspects to take care of: nodes, edges. Going deeper for network
generation purposes, we may want to look into the different possibilities of representing the attributes. Here, we are considering
to represent 4 characteristics: color of the nodes, size of the nodes, width of the edges, and, color of the edges. All these are
represented in the following way:
    - color of the nodes (node_color): represented by the number of different labels.
    - size of the nodes (node_size): represented by the number of occurrences of a given label.
    - width of the edges (width): represented by the weights of the edges
    - color of the edges (edge_color): same as the width: represented by the weights of the edges

All the aforementioned characteristics will be defined on the basis of the adjacency matrix which we will generate from the
co-occurrence matrix of labels from the given dataset.

Generating Data for Experimentation
-----------------------------------

Let's start by generating some exemplary data:
We'll generate a dataset with a defined number of classes, and sparse matrices

.. code-block:: python

    from sklearn.datasets import make_multilabel_classification
    n_classes = 7
    x, y = make_multilabel_classification(sparse=True, n_classes=n_classes, return_indicator='sparse', allow_unlabeled=False)

Defining Parameters
-------------------
The parameters that are needed for our function are:
    - x: {array-like, sparse matrix}, shape (n_samples, n_features)
    - y: numpy array, shape (n_samples, ) - Target values: labels
    - n_classes: number of labels from y
    - cmap_node: set of predefined color palette values from the colormap of networkx (Node Colormap)
    - cmap_edge: set of predefined color palette values from the colormap of networkx (Edge Colormap)

Getting Started with the Data
-----------------------------

In order to get our function working, we need to check if the data we are using is in the appropriate form that the function
accepts, here, we have it because it the exemplary data, we have set the sparse option to be true, but when we are not sure about it,
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

In order to draw the network nicely and not to overlap the nodes, we will use "spring_layout" that positions nodes using Fruchterman-Reingold
force-directed algorithm.

.. code-block:: python

    spring_pos = nx.spring_layout(G, k=2.0, iterations=50)

The edges are dependent on the weights, that we will take from the graph and save as list to further use them for drawing purposes:

.. code-block:: python

    weights = []
    for u, v, d in G.edges(data=True):
        weights.append(d['weight'])

We need to normalize the weights so that all the wights are related and can be compared

.. code-block:: python

    weights_normalized = [(i / max(weights)) * 20 for i in weights]

After having gotten the weights of the edges, we see that we have all the parameters for drawing the graph, so we will draw the graph
using the networkx draw() function:

.. code-block:: python

    nx.draw(G, pos=spring_pos, node_size=[v * 25 for v in co_occurrence_matrix.diagonal()], node_color=range(n_classes),
            cmap=cmap_node, edge_color=weights_normalized, width=weights_normalized, edge_cmap=cmap_edge)


We can see in the parameters of draw() the 4 characteristics that we had mentioned in the beginning:
    - node_size: taken from the primary diagonal of the co-occurrence matrix
    - node_color: taken from the number of classes mentioned
    - width: taken from the weights of the edges after normalization
    - edge_color: same as the width, taken from the weights of the edges after normalization
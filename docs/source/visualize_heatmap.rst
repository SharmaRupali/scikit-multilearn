.. _visualize_heatmap:

Developing a visualization method using Heatmap
================================================

In order to visualize the data, different methods of data visualization may be considered. One of the methods presented is the
Heatmap Visualization, developed in parallel with the Network Visualization, and the Cluster-Pie Visualization, which will be described
in parallel with the other modules, i.e. See "Developing a visualization method using Networks" OR "Developing a visualization method using Clustering".

Heatmap Visualization
---------------------

One of the selected approaches to visualization is a heatmap structure. Here, there are two types of representations given in a similar category, both
of them are based on the adjacency matrix calculated from the co-occurrence matrix of the labels from the initial data. One of them is a representation
of a heatmap, and the other one is a hierarchically-clustered heatmap (clustermap) that has clusters represented in rows and columns separately. Both
the representations can be used to see how the labels co-occur individually or as clusters.

Heatmap Details
---------------

Heatmaps visualize data through variations in coloring. When applied to a tabular format, heatmaps are useful for cross-examining multivariate data,
through placing variables in the rows and columns and coloring the cells within the table. In our case, heatmap is one of the selected methods because
of its basis for showing variance across multiple variables, revealing any patterns, displaying whether any variables are similar to each other, and for
detecting if any correlations exist in-between them.

Using Predefined Libraries
--------------------------

As the basis of these representations, we are using some predefined libraries. We import them in the following way:

.. code-block:: python

    import networkx as nx
    import numpy as np
    from matplotlib import pyplot
    import seaborn as sns

Generating Data for Experimentation
-----------------------------------

Let's start by generating some exemplary data. We'll generate a dataset with a defined number of classes, and sparse matrices:

.. code-block:: python

    from sklearn.datasets import make_multilabel_classification

    x, y = make_multilabel_classification(sparse = True, n_classes = 7, return_indicator = 'sparse', allow_unlabeled= False)


Defining Parameters and Defaults
--------------------------------

The parameters that are needed for our function are:
    * x: {array-like, sparse matrix}, shape (n_samples, n_features)
    * y: numpy array, shape (n_samples, ) - Target values: labels
    * cmap (Default = "Reds"): set of predefined color palette values

Getting Started with the Data
-----------------------------
First of all we need to check if the data we are using is in the appropriate form that the function
accepts, here, we have it because in the exemplary data, we have set the sparse option to be true, but when we are not sure about it,
we need to transform the generated data into the appropriate form, i.e. matrix elements as integers:

.. code-block:: python

    y_asInt = y.astype(int)

Function Description
--------------------

After having converted the data into the appropriate form, we need to get the co-occurrence matrix from the data, as it's the base
of the calculation of our adjacency matrix which will be the basis of our representations. The co-occurrence matrix is a matrix containing
the information about the labels occurring at the same time.

.. code-block:: python

    co_occurrence_matrix = y_asInt.T.dot(y_asInt)

Using the co-occurrence matrix that we just created, we'll calculate the adjacency matrix that we need for the representations, but before that
we need to convert the co-occurrence matrix into a graph like structure where it make the adjacency matrix from:

.. code-block:: python

    g = nx.from_scipy_sparse_matrix(co_occurrence_matrix)
    adjacency_matrix = nx.to_numpy_matrix(g, dtype=np.bool, nodelist=None)

After having the adjacency matrix, we are ready to use the "seaborn" library that we have imported for the representation of our heatmaps. As
mentioned earlier, we have two types of heatmap representations: heatmap and hierarchically-clustered heatmap. The first one to be shown is the
heatmap, for which we are using the "heatmap" function from the seaborn library itself:

.. code-block:: python

    pyplot.figure()
    sns.heatmap(adjacency_matrix, cmap=cmap)

Second one to be shown is the hierarchically-clustered heatmap, for which, again, we are using a preexisting function "clustermap" from the seaborn
library:

.. code-block:: python

    pyplot.figure()
    sns.set(color_codes=True)
    sns.clustermap(adjacency_matrix, cmap=cmap, robust=True)


This method is the simplest of all the other visualization methods implemented in parallel, and also commonly used to visualize relationship, co-occurrences,
patterns, comparisons, etc.

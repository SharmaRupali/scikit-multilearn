.. _visualize_cluster_pie:

Developing a visualization method using Clustering
==================================================

In order to visualize the data, different methods of data visualization may be considered. One of the methods presented is
Visualization using Clustering, developed in parallel with the Network Visualization, which will be described in parallel with the other module,
i.e. See "Developing a visualization method using Networks".

Cluster Visualization
---------------------

One of the selected methods to data visualization is using Clusters. K-means clustering has been used in order to find the centroids of the clusters
which have been operated on, followed by the proportioning of cluster sizes, their placement, their content, and many more factor that will be
discussed on board.
As the core of the implementation, the plots generated are focused on the labels, their proportion, their grouping - clustering according to where
they fall depending on their features taken from samples generated from the input dataset (assuming the input data to be already in form of {samples, features}),
their positioning on the plot in proportion to other cluster (See "Cluster Details"), and their size in proportion to other clusters.

All these are represented in the following way:
    * centroids of clusters: represented by the placing of the pies
    * centers for plotting markers (pies): represented as the center point to the centroids, and marked as the point where the center of the pie will be
    * sizes of clusters: taken from the sum of all labels present in a given cluster, and represent as the size of the pies in general
    * number of pie slices: solely dependent on the number of labels taken from the dataset
    * size of pie slices: dependent on the proportion of labels (number) in a given cluster independent from other clusters

All the aforementioned characteristics will be defined on the basis of centroids generating by clustering performed using K-means method, followed
by the process of generating the markers taking in consideration the labels, size proportions, and plot sizes.

Cluster Details
---------------

Being the initial step in this development, clustering will be considered the most important part of the same as this is where we get the centers
for our markers.
Clustering here is performed using the K-means method. K-means clustering aims to partition 'n' observations into 'k' clusters in which each observation
belongs to the cluster with the nearest mean, serving as a prototype of the cluster.

Using Predefined Libraries
--------------------------

Some already implemented python modules will be used as basis for the development. Following modules will be imported:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn import preprocessing
    from sklearn.cluster import KMeans

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
    * n_clusters: number of clusters the user wants to see the data split in
    * size_proportion: for better proportions of pie sizes for better visualization

Getting Started with the Data
-----------------------------

First of all, we need to see how many different classes of labels we are dealing with, which we take from the input data:

.. code-block:: python

    n_classes = len(y.toarray()[0])

Function Description
--------------------

In order to get started with the clustering, we need to standardize our dataset, for which we will use preprocessing form the sklearn module:

.. code-block:: python

    x_scaled = preprocessing.scale(x.toarray())

After getting a standardized array of our input samples, we may get started with the clustering process. We will simply call the Kmeans fucntion
from the sklearn.cluster module and predict the clusters depending on the number specified by the user (n_clusters as parameter):

.. code-block:: python

    kmeans = KMeans(n_clusters=n_clusters)
    k = kmeans.fit_predict(x_scaled)

This gives us an array of labels predicted for each sample of our dataset. We'll use this array for the generation of the centroids where all these
samples may fall:

.. code-block:: python

    labels = k
    centroids = kmeans.cluster_centers_

Now, we need to find the indices for the labels, given my kmeans clustering, in order to for further calculations of cluster and slice
size proportions respectively.

.. code-block:: python
.. _visualize_cluster_pie:

Developing a visualization method using Clustering
==================================================

In order to visualize the data, different methods of data visualization may be considered. One of the methods presented is
Visualization using Clustering, developed in parallel with the Network Visualization, and the Heatmap Visualization, which will be described
in parallel with the other modules, i.e. See "Developing a visualization method using Networks" OR "Developing a visualization method using Heatmap".

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
    * sizes of clusters: taken from the sum of all labels present in a given cluster, and represented as the size of the pies in general
    * number of pie slices: solely dependent on the number of labels taken from the dataset
    * size of pie slices: dependent on the proportion of labels (number) in a given cluster independent from other clusters

All the aforementioned characteristics will be defined on the basis of centroids generating by clustering performed using K-means method, followed
by the process of generating the markers taking in consideration the labels, size proportions, and plot sizes.

Cluster Details - K-means Clustering
------------------------------------

Being the initial step in this development, clustering will be considered the most important part of the same as this is where we get the centers
for our markers.
Clustering here is performed using the K-means method. K-means clustering aims to partition 'n' observations into 'k' clusters in which each observation
belongs to the cluster with the nearest mean, serving as a prototype of the cluster.

Using Predefined Libraries
--------------------------

Some already implemented python modules will be used as basis for development. Following modules will be imported:

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn import preprocessing
    from sklearn.cluster import KMeans

Generating Data for Experimentation
-----------------------------------

Let's start by generating some exemplary data. We'll generate a dataset with a defined number of classes, and sparse matrices:

.. code-block:: python

    from sklearn.datasets import make_multilabel_classification

    x, y = make_multilabel_classification(sparse=True, n_classes=7, return_indicator='sparse', allow_unlabeled=False)


Defining Parameters and Defaults
--------------------------------

The parameters that are needed for our function are:
    * x: {array-like, sparse matrix}, shape (n_samples, n_features)
    * y: numpy array, shape (n_samples, ) - Target values: labels
    * labels (Default = None): labels to be shown on the network, if not specified any, uses defaults
    * n_clusters (Default = 3): number of clusters the user wants to see the data split in
    * size_proportion (Default = 50): for better proportions of pie sizes for better visualization

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

Now, we need to find the indices for the labels, given by kmeans clustering, in order to use them for further calculations of cluster and slice
size proportions respectively. For achieving the aforementioned, we've written a function to find the elements grouped in given clusters - labels
that form part of the cluster. The function gives the indices of the labels defined initially in 'y':

.. code-block:: python

    def cluster_indices_numpy(clustNum, labels_array):
    return np.where(labels_array == clustNum)[0]

Using the defined function:

.. code-block:: python

    arr={}
    for i in range(0, n_clusters):
        arr[i] = cluster_indices_numpy(i, kmeans.labels_)

Now that we have the indices of the labels that we have in a cluster, we want to get the respective rows of labels from 'y'

.. code-block:: python

    lab = {}
    for key, values in arr.items():
        lab[key] = y[values].toarray()

After getting all the details related to individual labels present in clusters and having gotten their respective array, now
we want to see the proportion of each label in a given cluster, for which, we will find the total frequency of each label in
each cluster. This will serve us for determining the size of each label and also the clusters while plotting them:

.. code-block:: python

    clusters = {}
    for key, values in lab.items():
        clusters[key] = values.sum(axis=0)

As we already mentioned, that the previous step will help us determining the size of the clusters as well, we will calculate it
right away. We will calculate the total size of each cluster by summing up the label frequencies in them:

.. code-block:: python

    sizes = {}
    for key in clusters.keys():
        sizes[key] = np.sum(clusters[key]) * size_proportion

In order to label the slices of clusters, if no list is provided, we need to generate a list of labels:

.. code-block:: python

    if labels is None:
        labels = {}
        for i in range(0, n_classes):
            labels[i] = "Label " + str(i)

Next step in the pipeline is to determine the exact centers of the cluster centroids, in order to plot them
on the coordinate axes:

.. code-block:: python

    centers = {}
    for i in range(0, len(centroids)):
        centers[i] = (centroids[i,0], centroids[i,1])

Now we have to perform some calculations for the actual plotting of the results. We will begin with determining the size of each slice - label - in each cluster.
We take each label frequency as the radius for calculating the circumference of the circle and then divide it by the sum of all label frequencies in order to get
appropriate portions of pie slices. In order to determine the exact starting and ending points of the slices we are using the 'numpy.linspace' function which returns
evenly spaced numbers over a specified interval and the maximum of these numbers is used as our points:

.. code-block:: python

    sl = {}
    for key in clusters.keys():
        sl[key] = {}
        sl[key][0] = 2 * np.pi * clusters[key][0]/float(np.sum(clusters[key]))
        for i in range(1, len(clusters[key])):
            sl[key][i] = sl[key][i-1] + 2 * np.pi * clusters[key][i]/float(np.sum(clusters[key]))

    cl_xy = {}
    sl_max = {}
    for key in sl.keys():
        cl_xy[key] = {}
        x = [0] + np.cos(np.linspace(0, sl[key][0], 10)).tolist()
        y = [0] + np.sin(np.linspace(0, sl[key][0], 10)).tolist()
        cl_xy[key][0] = list(zip(x,y))
        sl_max[key] = np.max(cl_xy[key][0])
        for i in range(1, len(sl[key])):
            x = [0] + np.cos(np.linspace(sl[key][i-1], sl[key][i], 10)).tolist()
            y = [0] + np.sin(np.linspace(sl[key][i-1], sl[key][i], 10)).tolist()
            cl_xy[key][i] = list(zip(x,y))
            sl_max[key] = np.max(cl_xy[key][i])

After having calculated the sizes, proportions, etc. We also need to calculate the range of colors to use in order to distinguish one label from another, for
which we have written a function that generates colors using 'hsv' depending on the number needed:

.. code-block:: python

    def get_cmap(n, name='hsv'):
    return plt.cm.get_cmap(name, n+1)

Calling the function, we determine the number of colors needed:

.. code-block:: python

    cmap = get_cmap(n_classes)

After having found all the values needed for plotting the pies, we can do so now. First, we'll plot the centers of the centroids, as those are the fundamental
positions where we have the clusters present:

.. code-block:: python

    plt.figure()
    for key in centers:
        plt.scatter(centers[key][0], centers[key][1], marker="x", color='r')

And now we plot the rest:

.. code-block:: python

    fig, ax = plt.subplots()
    for key in cl_xy.keys():
        for i in cl_xy[key]:
            ax.scatter(centers[key][0], centers[key][1], marker=(cl_xy[key][i], 0), s=sl_max[key] ** 2 * sizes[key], c=cmap(i))
    ax.legend()
    plt.show()

We can notice in the parameters of scatter(), the specifications of the characteristics that we had considered to be our representation essentials:
    * centroids of clusters: represented as the whole pie marker
    * centers for plotting pie markers: taken from the central points of the centroids as coordinate values
    * sizes of clusters: taken from the sum of all labels present in a given cluster
    * number of pie slices: solely dependent on the number of labels taken from the dataset
    * size of pie slices: calculated basing on the frequency of labels in each cluster
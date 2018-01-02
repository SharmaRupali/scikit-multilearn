from __future__ import absolute_import

import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.cluster import KMeans


class VisualizeClusterPie():

    @staticmethod
    def visualize_input_cluster(x, y, labels=None, n_clusters=3, size_proportion=50, colors="hsv"):

        n_classes = len(y.toarray()[0]) # number of labels taken from the size of y

        x_scaled = preprocessing.scale(x.toarray()) # standardization of data

        # using k-means clustering algorithm to find the centroids of the clusters
        kmeans = KMeans(n_clusters=n_clusters)
        k = kmeans.fit_predict(x_scaled)  # fit_predict gives the array of labels (check formal definition)

        labels = k
        centroids = kmeans.cluster_centers_

        # indices of the labels using the function ClusterIndicesNumpy
        arr = {}
        for i in range(0, n_clusters):
            arr[i] = VisualizeClusterPie.cluster_indices_numpy(i, kmeans.labels_)

        # getting the rows from y having the indices
        lab = {}
        for key, values in arr.items():
            lab[key] = y[values].toarray()

        # get the total frequency of each label in the clusters with diff centroids
        clusters = {}
        for key, values in lab.items():
            clusters[key] = values.sum(axis=0)

        # calculating the total size of clusters by summing up all the label counts in them
        sizes = {}
        for key in clusters.keys():
            sizes[key] = np.sum(clusters[key]) * size_proportion

        # for naming the labels
        if labels is None:
            labels = {}
            for i in range(0, n_classes):
                labels[i] = "Label " + str(i)

        # find the centres of the cluster of centroids
        centers = {}
        for i in range(0, len(centroids)):
            centers[i] = (centroids[i, 0], centroids[i, 1])

        # calculating the size of the slices as starting point to ending point
        sl = {}
        for key in clusters.keys():
            sl[key] = {}
            sl[key][0] = 2 * np.pi * clusters[key][0] / float(np.sum(clusters[key]))
            for i in range(1, len(clusters[key])):
                sl[key][i] = sl[key][i - 1] + 2 * np.pi * clusters[key][i] / float(np.sum(clusters[key]))

        cl_xy = {}
        sl_max = {}
        for key in sl.keys():
            cl_xy[key] = {}
            x = [0] + np.cos(np.linspace(0, sl[key][0], 10)).tolist()
            y = [0] + np.sin(np.linspace(0, sl[key][0], 10)).tolist()
            cl_xy[key][0] = list(zip(x, y))
            sl_max[key] = np.max(cl_xy[key][0])
            for i in range(1, len(sl[key])):
                x = [0] + np.cos(np.linspace(sl[key][i - 1], sl[key][i], 10)).tolist()
                y = [0] + np.sin(np.linspace(sl[key][i - 1], sl[key][i], 10)).tolist()
                cl_xy[key][i] = list(zip(x, y))
                sl_max[key] = np.max(cl_xy[key][i])

        # getting the colors with the help of the function get_cmap
        cmap = VisualizeClusterPie.get_cmap(n_classes, colors)

        # plot the centers
        plt.figure()
        for key in centers:
            plt.scatter(centers[key][0], centers[key][1], marker="x", color='r')

        # plot the pies
        fig, ax = plt.subplots()
        for key in cl_xy.keys():
            for i in cl_xy[key]:
                ax.scatter(centers[key][0], centers[key][1], marker=(cl_xy[key][i], 0),
                           s=sl_max[key] ** 2 * sizes[key], c=cmap(i))
        ax.legend()
        plt.show()

    # to find the elements grouped in given clusters (values that form part of the cluster)
    # gives the indices of the labels (defined initially as rows in y)
    @staticmethod
    def cluster_indices_numpy(clustNum, labels_array):
        return np.where(labels_array == clustNum)[0]

    # generating random set of colors according to the number of classes/labels
    @staticmethod
    def get_cmap(n, colors):
        return plt.cm.get_cmap(colors, n + 1)



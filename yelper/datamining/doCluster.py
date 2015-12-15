#!/usr/bin/env python


__authors__ 	= [
    '"Prabhat Kumar" <prabhat.genome@gmail.com>',
    '"Aiswarya Thomas" <aiswaryathomas15@gmail.com>',
    '"Sequømics Corporation" <admin@sequomics.com>'
    ]
__company__ 	= 'Sequømics Corporation'
__homepage__ 	= 'http://sequomics.com/'
__account__ 	= 'SequomicsCorporation'
__githubURL__  = 'https://github.com/SequomicsCorporation'
__license__     = 'Apache License'

# ------------------------------------------------------------------------
# Copyright © 2015, Sequømics Corporation. All rights reserved.
#
# Licensed under the Apache License, Version 2.0, (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#       	http://www.apache.org/licenses/LICENSE-2.0
#                           	or
#   https://github.com/SequomicsCorporation/Yelper/blob/master/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================
import nltk
import numpy
import sklearn.cluster as skcluster
import sklearn.metrics as skmetrics
import scipy.cluster as scicluster
import scipy.spatial.distance as distance
import matplotlib.pyplot as plt
import time
# ========================================================================
# Python’s class mechanism [doCluster]: A: /yelper/datamining
# ========================================================================
class doCluster:
    # Defining Function and __init__ method as a constructor [private].
    def __init__(self):
        pass
    
    # @staticmethod declarations.
    # A way to write a method inside a class without reference to the object it is being called on.
    @staticmethod
    # A: k-means clustering.
    def k_means_scikit(matrix):
        k_means = skcluster.KMeans(n_clusters=50, init='k-means++', n_init=1, verbose=1)
        k_means.fit(matrix)
        # it will return output of k_means_scikit.
        return k_means.labels_
    
    # A: k-means clustering [nltk].
    # With cosine distance the algorithm doesn't converge.
    @staticmethod
    def k_means_nltk(matrix):
        doCluster = nltk.KMeansClusterer(50, nltk.euclidean_distance, avoid_empty_clusters=True, conv_test=10)
        labels = numpy.array(doCluster.cluster(matrix, True, trace=True))
        return labels
    
    # A: k-means clustering [scipy].
    # The vq module only supports vector quantization and the k-means algorithms.
    @staticmethod
    def k_means_scipy(matrix):
        centroids, distortion = scicluster.vq.kmeans(matrix, 50, thresh=0.1)
        print('Yelper® - Centroids:', centroids)
        print('Yelper® - Distortion:', distortion)
    
    # A: GAAC [nltk].
    # Group-average agglomerative clustering or GAAC.
    @staticmethod
    def gaac(matrix):
        doCluster = nltk.GAAClusterer()
        labels = numpy.array(doCluster.cluster(matrix, False, trace=True))
        dendrogram = doCluster.dendrogram()
        # to show dendrogram
        dendrogram.show()
        return labels
        
    # A: hierarchical/agglomerative [scipy].
    # Performs hierarchical/agglomerative clustering on the condensed distance matrix y.
    @staticmethod
    def linkage(matrix):
        linkage_matrix = scicluster.hierarchy.linkage(matrix)
        print('Yelper® - Linkage matrix:', linkage_matrix)
        dendrogram = scicluster.hierarchy.dendrogram(linkage_matrix)
        ax = plt.gca()
        xlbls = ax.get_xmajorticklabels()
        plt.show()
        leaves = dendrogram['leaves']
        print(leaves)
    
    # A: Mean shift [sklearn].
    # Mean shift clustering using a flat kernel.
    @staticmethod
    def mean_shift(matrix):
        mean_shift = skcluster.MeanShift()
        mean_shift.fit(matrix)
        labels = mean_shift.labels_
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print('Yelper® - Estimated number of clusters:', n_clusters_)
        return labels
    
    # A: Ward's method [sklearn].
    # Ward's method is a criterion applied in hierarchical cluster analysis.
    @staticmethod
    def ward(matrix):
        ward = skcluster.Ward(n_clusters=50, compute_full_tree=False)
        ward.fit(matrix)
        return ward.labels_

    # A: DBSCAN
    # Density-based spatial clustering of applications with noise (DBSCAN).
    @staticmethod
    def dbscan(matrix):
    	dbscan = skcluster.DBSCAN(eps=0.3, min_samples=50, metric='euclidean')
    	dbscan.fit(matrix)
    	labels = dbscan.labels_
    	# Number of clusters in labels, ignoring noise if present.
    	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    	print('Yelper® - Estimated number of clusters:', n_clusters_)
    	return labels

    # A: Silhouette Score
    # Silhouette refers to a method of interpretation and validation of consistency within clusters of data.
    # The technique provides a succinct graphical representation of how well each object lies within its cluster.
    @staticmethod
    def evaluate_performance(data, labels, metric='euclidean'):
    	score = skmetrics.silhouette_score(data, labels, metric=metric)
    	print('Yelper® - Labels:', labels)
    	print('Yelper® - Score:', score)
    	return score
    
    # A: To show "Cluster Data".
    @staticmethod
    def cluster_data(matrix, algorithm):
        if algorithm == 'k-means-scikit':
            return doCluster.k_means_scikit(matrix)

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

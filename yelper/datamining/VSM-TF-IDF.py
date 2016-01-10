#!/usr/bin/env python
#  -*- coding: utf-8 -*-

"""
============================
Vector Space Model - TF-IDF
============================
Machine Learning :: Text feature extraction (tf-idf).
------------------------------------------------------------------
In information retrieval or text mining,
the term frequency – inverse document frequency (also called tf-idf),
is a well know method to evaluate how important is a word in a document.
tf-idf are is a very interesting way to convert the textual representation
of information into a Vector Space Model (VSM), or into sparse features.
"""
__author__ = 'Aiswarya'

import math

itemData = (
    "Health and Medical Cosmetics & Beauty Supply",
    "Department Stores Grocery Mobile Phones",
    "Coffee & Tea Art Galleries Venues & Event Spaces",
    "Drugstores Convenience Stores Cosmetics & Beauty Supply",
    "Women's Clothing Costumes Accessories"
)
## [1]
print ("Yelper - Vector Space Model - TF-IDF.")
print ("=====================================")
# the Sklearn TF-IDF Vectorizer and transform my data for items(shopping) into the TF-IDF matrix:--
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(itemData)

print("Yelper - TF-IDF Result:")
print tfidf_matrix.shape
# result -- (5, 24)

## [2] The Cosine Similarity.
from sklearn.metrics.pairwise import cosine_similarity
print("Yelper - Cosine Similarity Result:")
## The tfidf_matrix[0:1] is the Scipy operation to get the first row of the sparse matrix and
## the resulting array is the Cosine Similarity between the first document with all items documents in the set.
## Note: that the first value of the array is 1.0 because it is the Cosine Similarity between
## the first document with itself.
print cosine_similarity(tfidf_matrix[0:1], tfidf_matrix)
# array([[ 1.          0.          0.          0.40895103  0.        ]])

## [3]
# It's already calculated on the previous step, so I'm just using the value.
cos_sim = 0.40895103
angle_in_radians = math.acos(cos_sim)
print("Yelper - The angle between the first and third items:")
print math.degrees(angle_in_radians)
## Note: The angle of ~65.8 is the angle between the first and
## the third document of my items document set.
# 65.861042836

# - EOF.

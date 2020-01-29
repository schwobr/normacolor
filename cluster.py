"""
Cluster pixels of an image and create corresponding masks.

**********************************************************
"""
# coding: utf8
from sklearn.cluster import KMeans
import pandas as pd
from data import get_generators
import numpy as np
from keras import Model


def get_centroids(itemlist, model, n_clusters=8, batch_size=16, patch_size=16):
    """
    Performs KMeans on input image features computed from input model.

    ******************************************************************
    """
    gen, _ = get_generators(itemlist, batch_size, 1., patch_size, 1.)
    extractor = Model(inputs=model.input,
                      outputs=model.get_layer('encoding').output)
    fts = extractor.predict_generator(gen, steps=len(gen))
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(fts)

"""
Cluster pixels of an image and create corresponding masks.

**********************************************************
"""
# coding: utf8
from sklearn.cluster import KMeans
from data import get_generators
from histogram import hist_cv
import numpy as np
from keras import Model
import cv2
from joblib import dump, load


def get_centroids(itemlist, extractor, save_path, n_clusters=8, batch_size=16, patch_size=16):
    """
    Performs KMeans on input image features computed from input model.

    ******************************************************************
    """
    gen, _ = get_generators(itemlist, batch_size, 1., patch_size, 1.)
    fts = extractor.predict_generator(gen, steps=len(gen))
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(fts)
    dump(kmeans, save_path)
    return kmeans

def get_histograms(itemlist, extractor, kmeans, save_path, batch_size=16, patch_size=16, img_size=256):
    """
    Compute a cumulated histogram for each cluster and each channel.

    ****************************************************************
    """
    n_clusters = kmeans.cluster_centers_.shape[0]
    hist = np.zeros((n_clusters, 256, 3))
    for item in item_list:
        gen, _ = get_generators([item], batch_size, 1., patch_size, 1.)
        fts = extractor.predict_generator(gen, steps=len(gen))
        clusters = kmeans.predict(fts)
        mask = clusters.reshape((img_size, img_size))
        img = cv2.imread(item)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(n_clusters):
            hist[i] += hist_cv(img, mask=(mask == i))
    np.save(save_path, hist)
    return hist

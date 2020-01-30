"""
Cluster pixels of an image and create corresponding masks.

**********************************************************
"""
# coding: utf8
from sklearn.cluster import MiniBatchKMeans
from data import get_generators, patchify
from histogram import hist_cv, get_lut
from model import make_autoencoder_model
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
    kmeans = MiniBatchKMeans(n_clusters=n_clusters)
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
    hist = hist.reshape(256, -1)
    np.save(save_path, hist)
    return hist

def load_batches(patches, batch_size=16):
    batch = []
    for patch in patches:
        if len(batch) < batch_size:
            batch.append(patch)
        else:
            x = np.stack(batch)
            yield x
            batch = []
    if batch != []:
        x = np.stack(batch)
        yield x        
    

def get_transform(weights_path, kmeans_path, hist_path, width=16, depth=3, patch_size=16, batch_size=16):
    """
    Get transform function that normalizers an image according to the reference histograms and clusters.

    ****************************************************************************************************
    """
    model = make_autoencoder_model(width, depth, patch_size)
    model.load_weights(weights_path)
    extractor = Model(inputs=model.inpu, outputs=model.get_layer('encoding').output)
    kmeans = load(kmeans_path)
    n_clusters = kmeans.cluster_centers_.shape[0]
    ref_hist = np.load(hist_path)
    src_hist = np.zeros((n_clusters, 256, 3))

    def _transform(x):
        """
        TODO: refactor
        """
        patches = patchify(x)
        fts = extractor.predict_generator(load_batches(patches, batch_size=batch_size))
        clusters = kmeans.predict(fts)
        mask = clusters.reshape((img_size, img_size))
        dtype = x.dtype
        div = dtype in [np.float16, np.float32, np.float64]
        if div:
            x = (x * 255)
        x = x.astype(np.uint8)
        stacked_x = []
        stacked_mask = []
        for i in range(n_clusters):
            bin_mask = (mask == i)
            src_hist[i] += hist_cv(x, mask=bin_mask)
            bin_mask = bin_mask[..., None]
            stacked_x.append(bin_mask * x)
            stacked_mask.append(bin_mask)
        stacked_x = np.stack(stacked_x).reshape(*x.shape[:2], -1)
        stacked_mask = np.stack(stacked_mask)
        src_hist = src_hist.reshape(256, -1)
        lut = get_lut(src_hist, ref_hist)
        stacked_x = cv2.LUT(stacked_x, lut[:, None]).reshape(n_clusters, *x.shape[:2], 3)
        stacked_x = (stacked_mask * stacked_x).astype(dtype)
        if div:
            stacked_x /= 255
        return stacked_x.sum(0)
    return _transform

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
    print('Computing centroids...')
    gen, _ = get_generators(itemlist, batch_size, 0.1, patch_size, 1.)
    fts = extractor.predict_generator(gen, steps=len(gen))
    fts = fts.reshape((fts.shape[0], -1))
    kmeans = MiniBatchKMeans(n_clusters=n_clusters)
    kmeans.fit(fts)
    dump(kmeans, save_path)
    return kmeans


def get_histograms(itemlist, extractor, kmeans, save_path, batch_size=16, patch_size=16, img_size=256):
    """
    Compute a cumulated histogram for each cluster and each channel.

    ****************************************************************
    """
    print('Computing histograms...')
    n_clusters = kmeans.cluster_centers_.shape[0]
    hist = np.zeros((n_clusters, 256, 3))
    for item in itemlist:
        gen, _ = get_generators([item], batch_size, 1., patch_size, 1.)
        fts = extractor.predict_generator(gen, steps=len(gen))
        fts = fts.reshape((fts.shape[0], -1))
        clusters = kmeans.predict(fts)
        mask = clusters.reshape((img_size, img_size))
        img = cv2.imread(str(item))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        for i in range(n_clusters):
            bin_mask = mask == i
            hist[i] += hist_cv(img, mask=bin_mask)
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


def get_mask(x, extractor, kmeans, batch_size=16):
    """
    Get mask and x as a byte array.

    *******************************
    """
    patches = patchify(x)
    fts = extractor.predict_generator(
        load_batches(patches, batch_size=batch_size))
    clusters = kmeans.predict(fts)
    mask = clusters.reshape(x.shape[:2])
    x = (x * 255)
    x = x.astype(np.uint8)
    return x, mask


def get_src_hist(x, mask, n_clusters):
    """
    Get histogram of x according to cluster mask. 

    *********************************************
    """
    src_hist = np.zeros((n_clusters, 256, 3))
    stacked_x = []
    stacked_mask = []
    for i in range(n_clusters):
        bin_mask = mask == i
        src_hist[i] += hist_cv(x, mask=bin_mask)
        bin_mask = bin_mask[..., None]
        stacked_x.append(bin_mask * x)
        stacked_mask.append(bin_mask)

    stacked_x = np.stack(stacked_x).reshape(*x.shape[:2], -1)
    stacked_mask = np.stack(stacked_mask)
    src_hist = src_hist.reshape(256, -1)
    return src_hist, stacked_x, stacked_mask


def get_transform(weights_path, kmeans_path, hist_path, width=16, depth=3, patch_size=16, batch_size=16):
    """
    Get transform function that normalizers an image according to the reference histograms and clusters.

    ****************************************************************************************************
    """
    model = make_autoencoder_model(width, depth, patch_size)
    model.load_weights(weights_path)
    extractor = Model(inputs=model.input,
                      outputs=model.get_layer('encoding').output)
    kmeans = load(kmeans_path)
    n_clusters = kmeans.cluster_centers_.shape[0]
    ref_hist = np.load(hist_path)

    def _transform(x):
        dtype = x.dtype
        div = dtype not in [np.float16, np.float32, np.float64]
        x = x.astype(np.float32)
        if div:
            x /= 255

        x, mask = get_mask(x, extractor, kmeans, batch_size=batch_size)
        src_hist, stacked_x, stacked_mask = get_src_hist(x, mask, n_clusters)

        lut = get_lut(src_hist, ref_hist)
        stacked_x = cv2.LUT(stacked_x, lut[:, None]).reshape(
            n_clusters, *x.shape[:2], 3)
        stacked_x = (stacked_mask * stacked_x).astype(dtype)
        if not div:
            stacked_x /= 255
        return stacked_x.sum(0)

    return _transform

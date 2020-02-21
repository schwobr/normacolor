"""
Cluster pixels of an image and create corresponding masks.

**********************************************************
"""
# coding: utf8
from sklearn.cluster import MiniBatchKMeans
from data import get_generators, patchify
from histogram import hist_cv, get_lut, match_hists
from model import make_autoencoder_model
import numpy as np
from keras import Model
import cv2
from joblib import dump, load
from math import ceil
import time


_format_configs = {
    'HSV': {
        'max_values': [359., 1., 1.],
        'n_bins': 360,
        'dtype': np.float32,
        'bgr_cvt': cv2.COLOR_BGR2HSV,
        'rgb_cvt': cv2.COLOR_RGB2HSV,
        'inv_cvt': cv2.COLOR_HSV2RGB
    },
    'RGB': {
        'max_values': [255, 255, 255],
        'n_bins': 256,
        'dtype': np.uint8,
        'bgr_cvt': cv2.COLOR_BGR2RGB,
        'rgb_cvt': None,
        'inv_cvt': None
    }
}


def get_centroids(itemlist, extractor, save_path,
                  n_clusters=8, batch_size=16, patch_size=16):
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


def get_histograms(itemlist, extractor, kmeans, save_path,
                   batch_size=16, patch_size=16, img_size=256, format='RGB'):
    """
    Compute a cumulated histogram for each cluster and each channel.

    ****************************************************************
    """
    print('Computing histograms...')
    n_clusters = kmeans.cluster_centers_.shape[0]
    conf = _format_configs[format]
    hist = np.zeros((n_clusters, conf['n_bins'], 3))
    for item in itemlist:
        gen, _ = get_generators([item], batch_size, 1., patch_size, 1.)
        fts = extractor.predict_generator(gen, steps=len(gen))
        fts = fts.reshape((fts.shape[0], -1))
        clusters = kmeans.predict(fts)
        mask = clusters.reshape((img_size, img_size))

        img = cv2.imread(str(item)).astype(conf['dtype'])
        if img.dtype == np.float32:
            img /= 255
        img = cv2.cvtColor(img, conf['bgr_cvt'])
        max_vals = np.array(conf['max_values'])[None, None]
        img *= (conf['n_bins'] - 1) / max_vals
        if format == 'HSV':
            img[..., 0] -= 250
            img[..., 0] %= 360
        img = img.astype(np.uint16)
        if format == 'HSV':
            img[..., 0] %= 360

        for i in range(n_clusters):
            bin_mask = mask == i
            hist[i] += hist_cv(img, mask=bin_mask,
                               n_bins=conf['n_bins'],
                               max_vals=conf['max_values'])
    hist = hist.reshape(conf['n_bins'], -1)
    np.save(save_path, hist)
    return hist


def load_batches(patches, batch_size=16):
    batch = []
    k = 0
    for k in range(ceil(len(patches)/batch_size)):
        yield patches[k*batch_size: (k+1)*batch_size]


def get_mask(x, extractor, kmeans, patch_size=16, batch_size=16):
    """
    Get mask and x as a byte array.

    *******************************
    """
    patches = patchify(x, patch_size)
    t = time.time()
    fts = extractor.predict_generator(
        load_batches(patches, batch_size=batch_size),
        steps=ceil(patches.shape[0]/batch_size))
    print(time.time()-t)
    fts = fts.reshape((fts.shape[0], -1))
    t = time.time()
    clusters = kmeans.predict(fts)
    print(time.time()-t)
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


def get_transform(weights_path, kmeans_path, hist_path,
                  width=16, depth=3, patch_size=16, batch_size=16):
    """
    Get transform function that normalizers an image according to the reference
    histograms and clusters.

    ***************************************************************************
    """
    model = make_autoencoder_model(width, depth, patch_size)
    model.load_weights(weights_path)
    extractor = Model(inputs=model.input,
                      outputs=model.get_layer('encoding').output)
    kmeans = load(kmeans_path)
    centers = kmeans.cluster_centers_
    n_clusters = centers.shape[0]
    ref_hist = np.load(hist_path)

    def _transform(x):
        dtype = x.dtype
        div = dtype not in [np.float16, np.float32, np.float64]
        x = x.astype(np.float32)
        if div:
            x /= 255
        x, mask = get_mask(x, extractor, kmeans,
                           patch_size=patch_size, batch_size=batch_size)
        src_hist, stacked_x, stacked_mask = get_src_hist(x, mask, n_clusters)

        lut = get_lut(src_hist, ref_hist)
        stacked_x = cv2.LUT(stacked_x, lut[:, None]).reshape(
            n_clusters, *x.shape[:2], 3).clip(0, 255)
        stacked_x = (stacked_mask * stacked_x).astype(dtype)
        if not div:
            stacked_x /= 255
        return stacked_x.sum(0)

    return _transform


def get_transform1(weights_path, kmeans_path, hist_path,
                   width=16, depth=3, patch_size=16,
                   batch_size=16, format='RGB', chans=[0, 1, 2]):
    """
    Get transform function that normalizers an image according to the reference
    histograms and clusters.

    ***************************************************************************
    """
    model = make_autoencoder_model(width, depth, patch_size)
    model.load_weights(weights_path)
    extractor = Model(inputs=model.input,
                      outputs=model.get_layer('encoding').output)

    kmeans = load(kmeans_path)
    centers = kmeans.cluster_centers_
    closest = ((centers[None]-centers[:, None]) **
               2).sum(-1).argsort(axis=-1)[:, 1:]
    n_clusters = centers.shape[0]

    conf = _format_configs[format]
    hist_ref = np.load(hist_path)
    hist_ref = hist_ref.reshape((n_clusters, conf['n_bins'], -1))

    def _transform(x):
        dtype = x.dtype
        div = dtype not in [np.float16, np.float32, np.float64]
        x = x.astype(np.float32)
        if div:
            x /= 255
        x, mask = get_mask(x, extractor, kmeans,
                           patch_size=patch_size, batch_size=batch_size)
        x = x.astype(conf['dtype'])
        if x.dtype == np.float32:
            x /= 255
        if conf['rgb_cvt'] is not None:
            x = cv2.cvtColor(x, conf['rgb_cvt'])
        max_vals = np.array(conf['max_values'])[None, None]
        if format == 'HSV':
            x[..., 0] -= 250
            x[..., 0] %= 360
        x *= (conf['n_bins'] - 1) / max_vals
        x = x.astype(np.uint16)
        if format == 'HSV':
            x[..., 0] %= 360
        tfmed = np.copy(x)
        tfmed[..., chans] = 0

        # for k in range(n_clusters):
        #     bin_mask = mask == k
        #     if bin_mask.sum() < 0.01 * np.prod(mask.shape):
        #         for j in range(n_clusters - 1):
        #             if (mask == j).sum() >= 0.01 * np.prod(mask.shape):
        #                 mask[bin_mask] = closest[k, j]
        #                 break
        for k in range(n_clusters):
            bin_mask = mask == k
            hist_src = hist_cv(x, mask=bin_mask,
                               n_bins=conf['n_bins'],
                               chans=chans,
                               max_vals=conf['max_values'])
            lut = get_lut(hist_src, hist_ref[k][:, chans])
            tfmed += (match_hists(x * bin_mask[..., None], lut) *
                      bin_mask[..., None])
        if format == 'HSV':
            tfmed[..., 0] += 250
            tfmed[..., 0] %= 360
        tfmed = tfmed.astype(conf['dtype'])
        tfmed *= max_vals / (conf['n_bins'] - 1)
        if conf['inv_cvt'] is not None:
            tfmed = cv2.cvtColor(tfmed, conf['inv_cvt'])
        if not div and format == 'RGB':
            tfmed /= 255
        elif div and format == 'HSV':
            tfmed *= 255
        tfmed = tfmed.astype(dtype)
        print(cv2.cvtColor(tfmed.astype(np.float32)/255, conf['rgb_cvt'])[10, 173, 0])
        return tfmed

    return _transform

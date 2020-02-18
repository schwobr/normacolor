import numpy as np
import cv2


def hist_cv(img, mask=None, n_bins=256, chans=[0, 1, 2]):
    """
    Get a histogram of the input image for each of its channels.

    ************************************************************
    """
    res = []
    if mask is not None:
        mask = mask.astype(np.uint8)
    for i in chans:
        res.append(cv2.calcHist([img], [i], mask, [n_bins], [0, n_bins]))
    return np.concatenate(res, axis=-1)


def get_lut(h_src, h_ref):
    """
    Get a lookup table from a source histogram to a reference histogram.

    ********************************************************************
    """
    h_src = h_src.cumsum(0)
    h_src /= (h_src.max(0) + 1e-7)
    h_ref = h_ref.cumsum(0)
    h_ref /= (h_ref.max(0) + 1e-7)

    lut = np.argmax((h_ref[:, None]+1e-7 >= h_src[None]), axis=0)
    return lut


def hist_matching(src, ref):
    """
    Matches the histogram of a source image with respect to a reference image.

    **************************************************************************
    """
    h_src, h_ref = hist_cv(src), hist_cv(ref)
    lut = get_lut(h_src, h_ref)
    return cv2.LUT(src, lut[:, None])


def match_hists(src, lut, n_bins=256, chans=[0, 1, 2]):
    res = np.zeros_like(src)
    for k in range(n_bins):
        for c in chans:
            res[..., c] = np.where((src == k)[..., c], lut[k, c], res[..., c])
    return res

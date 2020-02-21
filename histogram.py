import numpy as np
import cv2


def hist_cv(img, mask=None, n_bins=256, chans=[0, 1, 2], max_vals=None):
    """
    Get a histogram of the input image for each of its channels.

    ************************************************************
    """
    res = []
    if mask is not None:
        mask = mask.astype(np.uint8)
    for i in chans:
        res.append(cv2.calcHist([img], [i], mask, [n_bins], [
                   0, n_bins - 1 if max_vals is None else max_vals[i]]))
    return np.concatenate(res, axis=-1)


def get_lut(h_src, h_ref):
    """
    Get a lookup table from a source histogram to a reference histogram.

    ********************************************************************
    """
    h_src = h_src.cumsum(0).astype(np.float128)
    h_src /= (h_src.max(0) + 1e-7)
    h_ref = h_ref.cumsum(0).astype(np.float128)
    h_ref /= (h_ref.max(0) + 1e-7)
    lut = np.argmax((h_ref[:, None]+1e-20 >= h_src[None]), axis=0)
    return lut


def hist_matching(src, ref):
    """
    Matches the histogram of a source image with respect to a reference image.

    **************************************************************************
    """
    h_src, h_ref = hist_cv(src), hist_cv(ref)
    lut = get_lut(h_src, h_ref)
    return cv2.LUT(src, lut[:, None])


def match_hists(src, lut):
    res = np.zeros_like(src)
    for c in range(lut.shape[1]):
        new_vals = lut[:, c][src[..., c]]
        # new_vals = np.where(new_vals > src[..., c],
        #                     np.minimum(new_vals, src[..., c] + 60),
        #                     np.maximum(new_vals, src[..., c] - 60))
        res[..., c] = new_vals
    return res

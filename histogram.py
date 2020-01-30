import numpy as np
import cv2


def hist_cv(img, mask=None):
    """
    Get a histogram of the input image for each of its channels.

    ************************************************************
    """
    res = []
    for i in range(img.shape[-1]):
        res.append(cv2.calcHist([img], [i], mask, [256], [0, 256]))
    return np.concatenate(res, axis=-1)


def get_lut(h_src, h_ref):
    """
    Get a lookup table from a source histogram to a reference histogram.

    ********************************************************************
    """
    h_src = h_src.cumsum(axis=0)
    h_src /= h_src.max()
    h_ref = h_ref.cumsum(axis=0)
    h_ref /= h_ref.max()

    lut = np.argmax((h_ref[:, None] >= h_src[None]), axis=0)
    return lut


def hist_matching(src, ref):
    """
    Matches the histogram of a source image with respect to a reference image.

    **************************************************************************
    """
    h_src, h_ref = hist_cv(src), hist_cv(ref)
    lut = get_lut(h_src, h_ref)
    return cv2.LUT(src, lut[:, None])

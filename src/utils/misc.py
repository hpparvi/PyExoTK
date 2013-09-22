import numpy as np

def fold(x, period, origo=0.0, shift=0.0, normalize=True,  clip_range=None):
    """
    Folds the given data over a given period.
    """
    xf = ((x - origo)/period + shift) % 1.

    if not normalize:
        xf *= period
        
    if clip_range is not None:
        mask = np.logical_and(clip_range[0]<xf, xf<clip_range[1])
        xf = xf[mask], mask
    return xf

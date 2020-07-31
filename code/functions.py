import numpy as np

from utils import jackknife


def get_mag(cfgs: np.ndarray):
    """Return mean and error of magnetization."""
    return jackknife(cfgs.mean(axis=(1,2)))

def get_abs_mag(cfgs: np.ndarray):
    """Return mean and error of absolute magnetization."""
    return jackknife(np.abs(cfgs.mean(axis=(1,2))))

def get_chi2(cfgs: np.ndarray):
    """Return mean and error of suceptibility."""
    V = cfgs.shape[1] * cfgs.shape[2]
    mags = cfgs.mean(axis=(1,2))

    return jackknife(V * (mags**2 - mags.mean()**2))

def get_corr_func(cfgs: np.ndarray):
    """Return connected two-point correlation function with errors."""
    mag_sq = np.mean(cfgs)**2
    corr_func = []

    for i in range(1, cfgs.shape[1], 1):
        corrs = np.mean(0.5 * cfgs * (np.roll(cfgs, i, 1) + np.roll(cfgs, i, 2)), axis=(1,2)) - mag_sq
        corr_mean, corr_err = jackknife(corrs)
        corr_func.append([i, corr_mean, corr_err])

    return np.array(corr_func)
# File: chigrad/src/chigrad/statistics/jackknife.py

import numpy as np

def jackknife_resample(x, weights=None, f=None):
    N = len(x)
    w = np.ones(N) if weights is None else weights
    if len(w) != N:
        raise ValueError(f"jackknife.resample: Weights length {len(w)} != sample length {N}")
    mean = np.average(x, axis=0, weights=w)
    N_w = np.sum(w)
    if f is not None:
        return np.array([ f( mean + w[j] * (mean - x[j]) / (N_w - w[j]) ) for j in range(N)])
    w_col = w.reshape((-1,) + (1,) * (x.ndim - 1))
    return mean + w_col * (mean - x) / (N_w - w_col)

def jackknife_variance(jks, mean=None):
    if mean is None: mean = np.mean(jks, axis=0)
    N = len(jks)
    return np.sum((jks - mean)**2, axis=0) * (N-1) / N

def jackknife_covariance(jks, mean=None):
    if mean is None: mean = np.mean(jks, axis=0)
    N = len(jks)
    d = (jks - mean).reshape(N, -1)   # np.outer flattens its inputs
    return np.sum(d[:, :, None] * d[:, None, :], axis=0) * (N - 1) / N

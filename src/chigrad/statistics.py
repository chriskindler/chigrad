import numpy as np
from typing import Literal

def inverse_covariance(
    resamples:       np.ndarray, # shape (nres, ntime)
    resample_method: Literal["bootstrap", "jackknife"] = "jackknife"
):
    N    = len(resamples)
    mean = np.mean(resamples, axis=0)
    d    = (resamples - mean).reshape(N, -1)

    if resample_method == "jackknife":
        norm = (N - 1) / N
    elif resample_method == "bootstrap":
        norm = 1 / (N - 1)
    else:
        raise ValueError(f"Unknown resample method {resample_method!r}")
    cov     = norm * np.einsum("ni,nj->ij", d, d)
    cov_inv = np.linalg.inv(cov)
    return cov_inv 

def inverse_variance(
    resamples:       np.ndarray, # shape (nres, ntime)
    resample_method: Literal["bootstrap", "jackknife"] = "jackknife"
):
    N    = len(resamples)
    mean = np.mean(resamples, axis=0)
    d    = (resamples - mean).reshape(N, -1)

    if resample_method == "jackknife":
        norm = (N - 1) / N
    elif resample_method == "bootstrap":
        norm = 1 / (N - 1)
    else:
        raise ValueError(f"Unknown resample method {resample_method!r}")
    cov     = norm * np.einsum("ni,nj->ij", d, d)
    var_inv = np.diag(1.0 / np.diag(cov))
    return var_inv 

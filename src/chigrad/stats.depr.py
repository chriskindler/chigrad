from __future__ import annotations

import numpy as np

from scipy.special import gammaincc
from typing import Literal

# ================================================================================ 
# COMPUTE (INVERSE) COVARIANCE MATRIX FROM RESAMPLE DATA
# ================================================================================ 

def compute_cov(y: np.ndarray, resample_method: Literal["bootstrap", "jackknife"]) -> np.ndarray:
    n_res = y.shape[0]
    if resample_method == "jackknife":
        return (n_res - 1) * np.cov(y, rowvar=False, bias=True)
    if resample_method == "bootstrap":
        return np.cov(y, rowvar=False, bias=True)
    raise ValueError(f"Unknown resample_method: {resample_method!r}. Either bootstrap or jackknife.")

def compute_cov_inv(cov: np.ndarray, svd: bool = False) -> np.ndarray:
    if not svd:
        return np.linalg.inv(cov)
    else:
        # TODO: Implement singular-value-decomposition (SVD)
        raise NotImplementedError("SVD not yet implemented.")

def compute_sdev(y: np.ndarray, resample_method: Literal["bootstrap", "jackknife"]) -> np.ndarray:
    n_res = y.shape[0]
    if resample_method == "jackknife":
        return np.std(y, axis=0, ddof=0) * np.sqrt(n_res - 1)
    if resample_method == "bootstrap":
        return np.std(y, axis=0, ddof=1)
    else:
        raise ValueError(f"Unknown resample_method: {resample_method!r}. Either bootstrap or jackknife.")

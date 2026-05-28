from __future__ import annotations

import numpy as np
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
    raise ValueError(f"Unknown resample_method: {resample_method!r}. Either bootstrap or jackknife.")

SCALAR_KEYS = (
    # identification
    "timestamp",
    "hash",
    
    "bin_size",
    "nsquare",

    "fit_id",
    "model_id",
    "resample_method",
    "correlation_type",

    # window
    "t_min",
    "t_max",

    # counts
    "n_data", # number of points included in the fit = len(np.arange(t_min, t_max + 1)) 
    "n_par",
    "n_dof",
    "n_res",
    "n_valid_res",

    # quality
    "chi2",
    "pvalue",
    "aic",
    "aicc",

    # Migrad flags
    "valid",
    "converged",
    "accurate", # return True if the covariance matrix is accuarate
    "fmin_edm",

    # Migrad inputs
    "tolerance",
    "strategy",
    "ncall",
)

ARRAY_KEYS = (
    "fit_range",       # (n_pts,) timeslices fit
    "fit_avg",         # (n_pts,) fit data central values
    "fit_err",         # (n_pts,) fit data errors
    "cov",             # (n_pts, n_pts)
    "cov_inv",         # (n_pts, n_pts)
    "residuals",       # (n_pts,) y_data - y_model at central values
    "valid_res",       # (n_res,) bool — convergence per resample
    "fit_range_ext",   # (n_ext,) extended grid for plotting
    "model_eval_ext",  # (n_res, n_dense) fit evaluated per resample on fine grid
)

DICT_KEYS = (
    "params_start",       # {name: float} - starting values
    "params_limits",      # {name: (lo, hi)} - bounds
    "params_res",         # {name: (n_res,) array} - resample values
    "params_cen",         # {name: float} - central values
    "params_err",         # {name: float} - jackknife/bootstrap error
    "params_err_hesse",   # {name: float} - Hesse error from migrad (cross-check with jackknife/bootstrap error)
)
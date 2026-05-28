from __future__ import annotations

import iminuit
import numpy as np

from scipy.special import gammaincc
from typing import Callable, Literal, Optional

from chigrad.result import FitResult, FitRunResult

# ======================================================================================
# COST FUNCTIONS ALIAS CHI2 DISTRIBUTION FUNCTIONS FOR (UN)CORRELATED LEAST SQUARE FITS
# ======================================================================================

def _compute_correlated_cost(
    t:          np.ndarray,
    y:          np.ndarray,
    cov_inv:    np.ndarray,
    model:      Callable,
    param_keys: dict
):
    # TODO: Implement priors
    def correlated_cost(*param_values):
        params   = dict(zip(param_keys, param_values))
        residual = y - model(t, **params)
        chi2     = residual @ cov_inv @ residual
        return chi2

    # default errordef = iminuit.Minuit.LEAST_SQAURES = 1.0
    correlated_cost.errordef = iminuit.Minuit.LEAST_SQUARES
    correlated_cost.ndata    = len(t)

    return correlated_cost

def _compute_uncorrelated_cost(
    t:          np.ndarray,
    y:          np.ndarray,
    sdev_inv:   np.ndarray,
    model:      Callable,
    param_keys: dict
):
    # TODO: Implement priors
    def uncorrelated_cost(*param_values):
        params   = dict(zip(param_keys, param_values))
        residual = y - model(t, **params)
        chi2     = float(np.sum((residual * sdev_inv) ** 2))
        return chi2

    # default errordef = iminuit.Minuit.LEAST_SQAURES = 1.0
    uncorrelated_cost.errordef = iminuit.Minuit.LEAST_SQUARES
    uncorrelated_cost.ndata    = len(t)

    return uncorrelated_cost

def _fit(
    t:                np.ndarray,
    t_ext:            np.ndarray, # finer timeslice array for plotting fit function with errorbands later 
    y:                np.ndarray, # central or single resample row, shape (ndata,)

    correlation_type: Literal["correlated", "uncorrelated"],
    cov_inv:          Optional[np.ndarray],  # shape (ndata, ndata), correlated
    sdev_inv:         Optional[np.ndarray],  # shape (ndata,),       uncorrelated

    model:            Callable,
    params_start:     dict[str, float],
    params_limit:     Optional[dict[str, tuple[float | None, float | None]]],

    tolerance:        float,
    strategy:         int,
    ncall:            int

) -> FitResult:

    params_key = list(params_start.keys())
    
    if correlation_type == "correlated":
        cost = _compute_correlated_cost(t, y, cov_inv, model, params_key)
    else:
        cost = _compute_uncorrelated_cost(t, y, sdev_inv, model, params_key)

    # minuit constructor
    m = iminuit.Minuit(cost, *params_start.values(), name=tuple(params_start.keys()))

    m.tol      = tolerance
    m.strategy = strategy

    if params_limit:
        for name, lim in params_limit.items():
            m.limits[name] = lim

    m.migrad(ncall=ncall)
    m.hesse()

    params_est = dict(zip(m.parameters, m.values))
    y_fit      = model(t, **params_est)
    y_fit_ext  = model(t_ext, **params_est)
    residuals  = y - y_fit

    return FitResult(
        t               = t,
        t_ext          = t_ext,
        params_est      = dict(zip(m.parameters, m.values)),
        params_err_hess = dict(zip(m.parameters, m.errors)),

        chi2            = m.fval,
        ndata           = cost.ndata,
        npar            = m.nfit,
        
        tolerance       = tolerance,
        strategy        = strategy,
        ncall           = ncall,

        valid           = m.valid,
        converged       = m.fmin.is_valid and not m.fmin.is_above_max_edm,
        accurate_cov    = m.accurate,
        edm             = m.fmin.edm,

        y_fit           = y_fit,
        y_fit_ext       = y_fit_ext,
        residuals       = residuals
    )

def execute_fits(
    t:                    np.ndarray,
    t_ext:                np.ndarray,
    y:                    np.ndarray, # (nres, ndata)

    correlation_type:     Literal["correlated", "uncorrelated"],
    resample_type:        Literal["bootstrap", "jackknife"],
    cov_inv:              Optional[np.ndarray], # precomputed (ndata, ndata)
    sdev_inv:             Optional[np.ndarray], # precomputed (ndata,)

    execute_central_fit:  bool,
    execute_resample_fit: bool,

    model:                Callable,
    params_start:         dict[str, float],
    params_limit:         Optional[dict[str, tuple[float | None, float | None]]],

    tolerance:            float,
    strategy:             int,
    ncall:                int,
   
) -> tuple[FitResult, Optional[list[FitResult]]]:

    if not (execute_central_fit or execute_resample_fit):
        raise ValueError("Either central value or resample fit must be executed.")

    params_shared = dict(
        t                = t,
        t_ext            = t_ext,
        correlation_type = correlation_type,
        cov_inv          = cov_inv,
        sdev_inv         = sdev_inv,
        model            = model,
        params_start     = params_start,
        params_limit     = params_limit,
        tolerance        = tolerance,
        strategy         = strategy,
        ncall            = ncall,
    )

    if execute_central_fit:
        y_cen = np.mean(y, axis=0)
        central_result = _fit(y=y_cen, **params_shared)

    if not execute_resample_fit:
        return FitRunResult(
            central       = central_result,
            resample      = None,
            resample_type = resample_type
        )

    if execute_resample_fit:
        nres = y.shape[0]
        resample_result = [_fit(y=y[j], **params_shared) for j in range(nres)]

    return FitRunResult(
        central       = central_result,
        resample      = resample_result,
        resample_type = resample_type if execute_resample_fit else None
    ) 

"""
run.central.chi2        # central fit chi2
run.central.pvalue      # central fit pvalue
run.params_err          # {"E0": 0.003, "A0": 1.2e-7} — resample errors
run.y_fit_ext_err       # error band array on dense grid
"""

class ConstantFitResult:
    """FitResult-compatible container for closed-form correlated weighted means."""

    def __init__(self, B0: float, sigma: float, chi2: float, ndof: int):
        self.values  = {"B0": B0}
        self.errors  = {"B0": sigma}
        self.minos_errors = None

        self.chi2     = chi2
        self.npar     = 1
        self.ndof     = ndof
        self.chi2_red = chi2 / ndof if ndof > 0 else float("inf")

        # Closed-form has no convergence issues by construction.
        self.valid               = True
        self.accurate_covariance = True
        self.converged           = True

        self.covariance  = np.array([[sigma**2]])
        self.correlation = np.array([[1.0]])

def fit_constant_correlated(
    t_data:       np.ndarray,                                 # (ndata,) — unused, kept for API symmetry
    y_data:       np.ndarray,                                 # (n_res, ndata)
    cov_inv:      np.ndarray,                                 # (ndata, ndata) — required, precomputed
    resample_fit: bool = False,
) -> tuple[ConstantFitResult, Optional[list[ConstantFitResult]]]:
    """
    Closed-form correlated weighted mean for the constant model f(t) = B_0:
        B_0      = (1^T C^-1 y) / (1^T C^-1 1)
        sigma^2  = 1 / (1^T C^-1 1)
        chi^2    = (y - B_0)^T C^-1 (y - B_0)
        ndof     = ndata - 1

    The error sigma is the parameter error from the inverse covariance, used
    for the central FitResult. Per-resample fits return their own central B_0
    values; jackknife/bootstrap errors should be computed from the spread of
    resample B_0 values, not from sigma.
    """
    n_res, ndata = y_data.shape
    ones  = np.ones(ndata, dtype=float)
    cinv1 = cov_inv @ ones                                    # (ndata,)
    denom = float(ones @ cinv1)                               # scalar
    sigma = float(np.sqrt(1.0 / denom))
    ndof  = ndata - 1

    def _fit_one(y: np.ndarray) -> ConstantFitResult:
        B0    = float((cinv1 @ y) / denom)
        resid = y - B0
        chi2  = float(resid @ cov_inv @ resid)
        return ConstantFitResult(B0=B0, sigma=sigma, chi2=chi2, ndof=ndof)

    # Central fit
    y_avg          = np.mean(y_data, axis=0)
    central_result = _fit_one(y_avg)

    if not resample_fit:
        return central_result, None

    # Per-resample fits
    resample_results = [_fit_one(y_data[i]) for i in range(n_res)]
    return central_result, resample_results

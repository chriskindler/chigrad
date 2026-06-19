# File: chigrad/src/chigrad/minimise.py
import iminuit
import numpy as np

from dataclasses import replace
from typing import Callable, Literal, Optional

from chigrad.log import message
from chigrad.fit.config import FitConfig
from chigrad.fit.result import FitResult, FitRunResult

class ConvergenceError(Exception):
    pass

def _compute_correlated_chi2(
    t:          np.ndarray, # dim = (ndata, )
    y:          np.ndarray, # dim = (nres, ) 
    f:          Callable,
    cov_inv:    np.ndarray, # dim = (ndata, ndata)
    param_start:         dict[str, float],
    priors:     Optional[dict[str, tuple[float, float]]] = None,
):
    param_keys = list(param_start.keys())

    def cost(*vals):
        p = dict(zip(param_keys, vals))
        r = y - f(t, **p)
        chi2 = np.einsum("i, ij, j->", r, cov_inv, r) 
        if priors is not None:
            return chi2 + _prior_term(p, priors)
        else:
            return chi2

    cost.errordef = iminuit.Minuit.LEAST_SQUARES # = 1.0 means 1sigma
    cost.ndata    = len(t) + (len(priors) if priors else 0)

    return cost

def _compute_uncorrelated_chi2(
    t:          np.ndarray, # dim = (ndata, )
    y:          np.ndarray, # dim = (nres, )
    f:          Callable,
    var_inv:    np.ndarray, # dim = (ndata, )
    param_start:         dict[str, float],
    priors:     Optional[dict[str, tuple[float, float]]] = None,
):
    param_keys = list(param_start.keys())

    def cost(*vals):
        p = dict(zip(param_keys, vals))
        r = y - f(t, **p)
        chi2 = np.einsum("i, i->", var_inv, r ** 2)
        if priors is not None:
            return chi2 + _prior_term(p, priors)
        else:
            return chi2

    cost.errordef = iminuit.Minuit.LEAST_SQUARES # = 1.0 means 1sigma
    cost.ndata    = len(t) + (len(priors) if priors else 0)

    return cost

def _prior_term(params, priors):
    """
        priors: Optional[dict[str, tuple[float, float]]] = None
        priors = {
            "p": (mu, sigma),
            "E1": (1.2, 0.3), # E1 = 1.2 ± 0.3  (Gaussian prior)
            "A1": (0.3, 0.1), # A1 = 0.3 ± 0.1
        }

    """
    if not priors:
        return 0.0
    return sum(((params[k] - mu) / sig) ** 2 for k, (mu, sig) in priors.items())

def minimise(config: FitConfig,t: np.ndarray, y: np.ndarray, f: Callable, correlation_type: Literal["correlated", "uncorrelated"] ,cov_inv = None, var_inv = None):
    # TODO: Documentation

    if correlation_type == "correlated": 
        if cov_inv is None:
            raise ValueError("Correlated fits require inverse covariance matrix.")
        cost = _compute_correlated_chi2(t, y, f, cov_inv, config.param_start, config.priors)
    else:
        if var_inv is None:
            raise ValueError("Uncorrelated fits require inverse variance.")
        cost = _compute_uncorrelated_chi2(t, y, f, var_inv, config.param_start, config.priors)

    param_keys = list(config.param_start.keys())
    param_vals = list(config.param_start.values())

    # Construct Minuit object
    m = iminuit.Minuit(cost, *param_vals, name = tuple(param_keys))
    if config.tolerance is not None:
        m.tol = config.tolerance
    m.strategy = config.strategy
    if config.param_limit:
        message(f"Parameter limits imposed: {config.param_limit}")
        for name, lim in config.param_limit.items():
            m.limits[name] = lim
    
    # minimisation: simplex -> param_start_est -> migrad -> param_start_final
    if config.enable_simplex:
        message(f"SIMPLEX algorithm initialised: Perform Nelder-Mead optimisation.")
        m.simplex()

    message(f"MIGRAD algorithm initialised: Perform optimisation. Strategy {config.strategy}, tolerance {config.tolerance}, and maximum of {config.iterations} function calls.")
    m.migrad(ncall=config.iterations)
    message(f"Optimisation performed with {config.iterations} function calls.")

    if config.raise_failure and not m.valid:
        raise ConvergenceError("Optimisation terminated. Failed convergence.")

    if m.valid:
        message(f"Valid optimisation.")
        if m.accurate:
            message(f"Accurate covariance: {m.accurate}.")
        else:
            message(f"Accurate covariance: {m.accurate}.")
            message("Optimisation terminated, but with inaccurate covariance.")

        message("Computing Hessian matrix.")
        m.hesse()

    else:
        msg = "Invalid optimisation."
        fmin = m.fmin
        if fmin.has_reached_call_limit:
            msg += " Call limit was reached."
        if fmin.is_above_max_edm:
            msg += " Estimated distance to minimum too large."

        message(msg)

    return m

def _build_result(m, t, y, f) -> FitResult:
    param_est  = dict(zip(m.parameters, m.values))
    param_hess = dict(zip(m.parameters, m.errors)) if m.valid else None
    return FitResult(
        t=t, y=y,
        param_est=param_est,
        param_hess=param_hess,
        chi2=float(m.fval),
        ndat=int(m.ndof + m.nfit),
        npar=m.nfit,
        nfcn=m.nfcn,
        residuals=y - f(t, **param_est),
    )

def execute(config: FitConfig, t, y, f, correlation_type: Literal["correlated", "uncorrelated"], cov_inv=None, var_inv=None) -> FitRunResult:
    # Estimate params with simplex on, output serves es input for migrad
    y_cen = np.mean(y, axis=0)
    m = minimise(config, t, y_cen, f, correlation_type, cov_inv=cov_inv, var_inv=var_inv)
    central = _build_result(m, t, y_cen, f)

    resample = None
    # execute resample fits, if desired 
    if config.execute_resample:
        message(f"Running {y.shape[0]} resample fits.", silent=config.silent_output)
        res_config = replace(
            config,
            param_start    = central.param_est,
            enable_simplex = False,
            raise_failure  = False,
            silent_output  = True,
        )

        resample = []
        for k in range(y.shape[0]):
            m = minimise(res_config, t, y[k], f, correlation_type, cov_inv=cov_inv, var_inv=var_inv)
            resample.append(_build_result(m, t, y[k], f))

        message(f"Resample fits complete: {len(resample)} done.", silent=config.silent_output)

    return FitRunResult(
        central       = central,
        resample      = resample,
        resample_type = config.resample_type,
    )

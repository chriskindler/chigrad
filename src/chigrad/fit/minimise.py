# File: chigrad/src/chigrad/minimise.py
import iminuit
import numpy as np

from typing import Callable, Optional
from chigrad.log import message

from chigrad.statistics.jackknife import jackknife_variance, jackknife_covariance 

class ConvergenceError(Exception):
    pass

def _compute_correlated_chi2(
    t:          np.ndarray, # dim = (ndata, )
    y:          np.ndarray, # dim = (nres, ) 
    f:          Callable,
    cov_inv:    np.ndarray, # dim = (ndata, ndata)
    p0:         dict[str, float],
    priors:     Optional[dict[str, tuple[float, float]]] = None,

):
    param_keys = list(p0.keys())

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
    p0:         dict[str, float],
    priors:     Optional[dict[str, tuple[float, float]]] = None,
):
    param_keys = list(p0.keys())

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
            "E1": (1.2, 0.3), # E1 ~ 1.2 ± 0.3   (Gaussian prior)
            "A1": (0.3, 0.1), # A1 ~ 0.3 ± 0.1
        }

    """
    if not priors:
        return 0.0
    return sum(((params[k] - mu) / sig) ** 2 for k, (mu, sig) in priors.items())

def minimise(
    t:               np.ndarray, # dim = (ndata, )
    y:               np.ndarray, # dim = (nres, )
    f:               Callable,
    cov_inv =        None, # if passed, dim = (ndata, ndata)
    var_inv =        None, # if passed, dim = (ndata, )
    *,
    correlated:      bool,
    p0:              dict[str, float],
    p0_limits:       Optional[dict[str, tuple]] = None,
    priors:          Optional[dict[str, tuple[float, float]]] = None,
    tolerance:       float = 0.1,
    strategy:        int = 1,
    iterations:      int = 10000,
    use_simplex:     bool = True, # Always True
    raise_failure:   bool = True, # True for central value fits, False for resample fits
):
    # TODO: Documentation

    if correlated:
        if cov_inv is None:
            raise ValueError("Correlated fits require inverse covariance matrix.")
        cost = _compute_correlated_chi2(t, y, f, cov_inv, p0, priors)
    else:
        if var_inv is None:
            raise ValueError("Uncorrelated fits require inverse variance.")
        cost = _compute_uncorrelated_chi2(t, y, f, var_inv, p0, priors)

    param_keys = list(p0.keys())
    param_vals = list(p0.values())

    # Construct Minuit object
    m = iminuit.Minuit(cost, *param_vals, name = tuple(param_keys))
    if tolerance is not None:
        m.tol = tolerance
    m.strategy = strategy
    if p0_limits:
        message(f"Parameter limits imposed: {p0_limits}")
        for name, lim in p0_limits.items():
            m.limits[name] = lim
    
    # minimisation: simplex -> p0_est -> migrad -> p0_final
    if use_simplex:
        message(f"SIMPLEX initialised: Perform Nelder-Mead optimisation.")
        m.simplex()

    message(f"MIGRAD initialised: Perform optimisation. Strategy {strategy}, tolerance {tolerance}, and maximum of {iterations} function calls.")
    m.migrad(ncall=iterations)
    message(f"Optimisation performed with {iterations} function calls.")

    if raise_failure and not m.valid:
        raise ConvergenceError("Optimisation terminated. Failed convergence.")

    if m.valid:
        message(f"Valid optimisation: {m.valid}. Terminated successfully.")
        if m.accurate:
            message(f"Accurate covariance: {m.accurate}.")
        else:
            message(f"Accurate covariance: {m.accurate}.")
            message("Optimisation terminated successfully, but with inaccurate covariance.")

        message("Computing Hessian matrix.")
        m.hesse()
        message("Hessian matrix computed successfully.")
        # values and errors → length-npar vectors
        vals = np.asarray(m.values)             # shape (4,)  [A0, E0, A1, E1]
        errs = np.asarray(m.errors)             # shape (4,)
        print(vals.shape, errs.shape)
        print(vals, errs)

        # covariance → npar × npar matrix
        cov = np.asarray(m.covariance)          # shape (4, 4)
        print(cov.shape)
        print(cov)

        # scalars
        print(m.fval, m.valid, m.nfit, m.ndof)

    return m

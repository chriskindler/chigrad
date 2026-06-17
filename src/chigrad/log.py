from time import time
from chigrad.fit.result import FitRunResult

_t0 = None

def message(s="", silent=False):
    """Print a chirgrad-prefixed log line with seconds since the first call.

    ``t0`` is captured lazily on the first invocation so the timestamp
    reflects "seconds since first log message", not import time.
    """
    global _t0
    if _t0 is None:
        _t0 = time()
    if not silent:
        print(f"CHIGRAD: {time()-_t0:.6f}s: {s}")


def log_fit_summary(run: FitRunResult):
    message("="*50)
    message("FIT SUMMARY")
    message("="*50)

    # central fit quality
    message(f"chi2/dof = {run.chi2dof:.3f}  (chi2={run.chi2:.2f}, ndof={run.ndof})")
    message(f"p-value  = {run.pvalue:.3f}")
    message(f"AIC      = {run.aic:.2f}   AICc = {run.aicc:.2f}")

    # parameters with quoted errors
    message("-"*50)
    message("FINAL PARAMETERS (central value ± jackknife error):")
    if run.param_final:
        for name, (val, err) in run.param_final.items():
            hess = run.param_hess[name] if run.param_hess else float("nan")
            message(f"  {name:>4} = {val:.6f} ± {err:.6f}   (hess: {hess:.6f})")
    else:
        for name, val in run.param_est.items():
            message(f"  {name:>4} = {val:.6f}")

    # resample stability
    if run.resample:
        message("-"*50)
        message(f"RESAMPLES: {run.nres_valid}/{run.nres} valid")
        c2d = run.resample_chi2dof
        if c2d is not None and len(c2d) > 0:
            message(f"  chi2/dof spread: {c2d.mean():.3f} ± {c2d.std():.3f}")
        if run.nres_valid < run.nres:
            message(f"  WARNING: {run.nres - run.nres_valid} resample fits failed")

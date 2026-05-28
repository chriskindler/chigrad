# chigrad/results.py
from __future__ import annotations
import iminuit
import numpy as np

from scipy.special import gammaincc
from typing import Callable, Literal, Optional

# The fit-result dict has three categories of values, each goes to a different
# place during serialization:
#   - SCALAR_KEYS  → manifest (parquet) and HDF5 attributes
#   - ARRAY_KEYS   → HDF5 datasets (too bulky for parquet)
#   - DICT_KEYS    → flattened into manifest columns; saved as nested groups in HDF5

# ================================================================================ 
# FIT RESULT
# ================================================================================ 

class FitResult:
    # probably should used @dataclass for class FitResult to avoid huge __init__
    def __init__(
        self,
        t:      np.ndarray,
        t_ext: np.ndarray,
        params_est:      dict[str, float],
        params_err_hess: dict[str, float],
        chi2:            float,
        ndata:           int,
        npar:            int,
        # migrad inputs
        tolerance:       float,
        strategy:        int,
        ncall:           int,
        # migrad outputs
        valid:           bool,
        converged:       bool,
        accurate_cov:    bool,
        edm:             float,
        # model evaluation
        y_fit:           np.ndarray,
        y_fit_ext:       np.ndarray,
        residuals:       np.ndarray
    ):
        self.t       = t
        self.t_ext       = t_ext,
        self.params_est       = params_est
        self.params_err_hess  = params_err_hess 
        self.chi2             = chi2
        self.ndata            = ndata
        self.npar             = npar
        
        self.tolerance        = tolerance
        self.strategy         = strategy
        self.ncall            = ncall

        self.valid            = valid
        self.converged        = converged
        self.accurate_cov     = accurate_cov
        self.edm              = edm

        self.y_fit            = y_fit
        self.y_fit_ext        = y_fit_ext
        self.residuals        = residuals

    # ================================================================================ 
    # DERIVED PROPERTIES: NUMBER OF DEGREES OF FREEDOM, P-VALUE AND (CORRECTED) AIC 
    # ================================================================================ 

    @property
    def ndof(self) -> int:
        return self.ndata - self.npar

    @property
    def pvalue(self) -> float:
        """
            compute pvalue from upper/lower incomplete gamma functions
            https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.gammaincc.html

            lower incomplete gamma function: gammainc(a,x)
            upper incomplete gamma function: gammaincc(a,x)

            they are related via
            gammaincc(a,x) = 1 - gammainc(a,x)

            in our case
            a = n_dof / 2
            x = chi2 / 2
        """
        return float(gammaincc(self.ndof / 2.0, self.chi2 / 2.0))

    @property
    def aic(self) -> float:
        """
            E.T. Neil and J.W. Sitison
            Model averaging approaches to data subset selection (2023)
            https://arxiv.org/pdf/2305.19417

            AIC = chi2 + 2 * (npar - ndata)

            npar  = number of fit parameters
            ndata = number of data points, i.e. number of y(x)-values
        """
        return self.chi2 + 2.0 * (self.npar - self.ndata)

    @property
    def aicc(self) -> float:
        # corrected AIC (AICc) accounting for small sample sizes
        # AICc = AIC + 2 * (npar ** 2 + npar ) / (ndata - npar - 1)
        aic = self.chi2 + 2.0 * (self.npar - self.ndata)
        return aic + 2.0 * (self.npar ** 2 + self.npar) / (self.ndata - self.npar - 1)


    # TODO: Representation
    def __repr__(self) -> str:
        pass

class FitRunResult:
    def __init__(
        self,
        central:       FitResult,
        resample:      Optional[list[FitResult]],
        resample_type: Literal["bootstrap", "jackknife"]
    ):
        self.central       = central
        self.resample      = resample
        self.resample_type = resample_type

    @property
    def params_err(self) -> Optional[dict[str, float]]:
        if not self.resample:
            return None
        nres    = len(self.resample)
        params  = {k: np.array([r.params_est[k] for r in self.resample])
                   for k in self.central.params_est}
        if self.resample_type == "jackknife":
            return {
                k: float(np.sqrt((nres - 1) / nres * np.sum((v - v.mean())**2)))
                for k, v in params.items()
            }
        else: # bootstrap
            return {
                k: float(np.std(v, ddof=1))
                for k, v in params.items()
            }

    @property
    def y_fit_ext_err(self) -> Optional[np.ndarray]:
        """Error band on the fit function evaluated on t_ext."""
        if not self.resample:
            return None
        nres  = len(self.resample)
        bands = np.array([r.y_fit_ext for r in self.resample])  # (nres, ndense)
        if self.resample_type == "jackknife":
            mean = bands.mean(axis=0)
            return np.sqrt((nres - 1) / nres * np.sum((bands - mean)**2, axis=0))
        else:
            return np.std(bands, ddof=1, axis=0)
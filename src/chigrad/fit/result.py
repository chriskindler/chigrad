import numpy as np

from dataclasses import dataclass
from scipy.special import gammaincc
from typing import Literal, Optional

from chigrad.fit.config import FitConfig

@dataclass
class FitResult:
    t:          np.ndarray
    y:          np.ndarray
    
    param_est:  dict[str, float]
    param_hess: dict[str, float] | None

    chi2:      float
    ndat:      int
    npar:      int
    nfcn:      int
    residuals: np.ndarray 

    @property
    def ndof(self) -> int:
        return self.ndat - self.npar

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
        return self.chi2 + 2.0 * (self.npar - self.ndat)

    @property
    def aicc(self) -> float:
        # corrected AIC (AICc) accounting for small sample sizes
        # AICc = AIC + 2 * (npar ** 2 + npar ) / (ndata - npar - 1)
        aic = self.chi2 + 2.0 * (self.npar - self.ndat)
        return aic + 2.0 * (self.npar ** 2 + self.npar) / (self.ndat - self.npar - 1)

@dataclass
class FitRunResult:
    central:          FitResult
    resample:         Optional[list[FitResult]]
    resample_type:    Literal["bootstrap", "jackknife"] = "jackknife"

    @property
    def param_est(self) -> dict[str, float]:
        return self.central.param_est

    @property
    def param_hess(self) -> Optional[dict[str, float]]:
        return self.central.param_hess

    @property
    def param_err(self) -> Optional[dict[str, float]]:
        if not self.resample:
            return None
        
        # collect all valid resample fits
        valid = [r for r in self.resample if r.param_hess is not None]
        n = len(valid)
        keys = self.central.param_est.keys()
        stack = {k: np.array([r.param_est[k] for r in valid]) for k in keys}
        if self.resample_type == "jackknife":
            return {k: float(np.sqrt((n - 1) / n * np.sum((v - v.mean()) ** 2))) for k, v in stack.items()}
        else: # bootstrap
            return {k: float(np.std(v, ddof=1)) for k, v in stack.items()}

    @property
    def param_final(self) -> Optional[dict[str, tuple[float, float]]]:
        """{p: (val, err)}."""

        if self.param_err is None:
            return None
        
        val, err = self.param_est, self.param_err
        return {k: (val[k], err[k]) for k in val}

    # ============================================================
    # GOODNESS OF FIT — central fit's quoted quality
    # ============================================================

    @property
    def chi2(self) -> float:
        return self.central.chi2

    @property
    def ndof(self) -> int:
        return self.central.ndof

    @property
    def chi2dof(self) -> float:
        return self.central.chi2 / self.central.ndof if self.central.ndof > 0 else float("inf")

    @property
    def pvalue(self) -> float:
        return self.central.pvalue

    @property
    def aic(self) -> float:
        return self.central.aic

    @property
    def aicc(self) -> float:
        return self.central.aicc

    # ============================================================
    # RESAMPLE DIAGNOSTICS — distributions across resample fits
    # ============================================================

    @property
    def nres(self) -> Optional[int]:
        # number of resample fits
        return len(self.resample) if self.resample else None

    @property
    def nres_valid(self) -> Optional[int]:
        # number of valid resample fits
        if not self.resample:
            return None
        return sum(1 for r in self.resample if r.param_hess is not None)

    @property
    def resample_chi2(self) -> Optional[np.ndarray]:
        if not self.resample:
            return None
        return np.array([r.chi2 for r in self.resample])

    @property
    def resample_chi2dof(self) -> Optional[np.ndarray]:
        """chi2/dof for each resample — stability check."""
        if not self.resample:
            return None
        return np.array([r.chi2 / r.ndof for r in self.resample if r.ndof > 0])

    @property
    def resample_pvalue(self) -> Optional[np.ndarray]:
        if not self.resample:
            return None
        return np.array([r.pvalue for r in self.resample])

    @property
    def resample_aic(self) -> Optional[np.ndarray]:
        if not self.resample:
            return None
        return np.array([r.aic for r in self.resample])

    @property
    def resample_aicc(self) -> Optional[np.ndarray]:
        if not self.resample:
            return None
        return np.array([r.aicc for r in self.resample])

    # ============================================================
    # PARAMETER COVARIANCE — resample spread (for error bands)
    # ============================================================

    @property
    def param_cov(self) -> Optional[np.ndarray]:
        """Full resample parameter covariance (npar, npar) — for error bands."""
        if not self.resample:
            return None
        valid = [r for r in self.resample if r.param_hess is not None]
        keys  = list(self.central.param_est.keys())
        P = np.array([[r.param_est[k] for k in keys] for r in valid])  # (nres, npar)
        n = len(valid)
        d = P - P.mean(axis=0)
        if self.resample_type == "jackknife":
            return (n - 1) / n * (d.T @ d)
        return np.cov(P, rowvar=False, ddof=1)

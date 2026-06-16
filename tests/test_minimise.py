import numpy as np
import pytest

from chigrad.fit.minimise import (
    minimise,
    _compute_correlated_chi2,
    _compute_uncorrelated_chi2,
    _prior_term,
    ConvergenceError,
)

def model(t, A0, E0, A1, E1):
    """Two-exponential two-point function."""
    return A0 * np.exp(-E0 * t) + A1 * np.exp(-E1 * t)

P_TRUE = {"A0": 1.0, "E0": 0.5, "A1": 0.3, "E1": 1.2}

@pytest.fixture
def synthetic():
    """Noisy correlator + frozen inverse covariance/variance."""
    rng = np.random.default_rng(0)
    t = np.arange(0, 16)
    y_true = model(t, **P_TRUE)

    nres = 200
    sigma = 0.01 * y_true + 1e-4
    samples = y_true[None, :] + rng.normal(0, sigma, size=(nres, len(t)))
    y = samples.mean(axis=0)

    cov = np.cov(samples, rowvar=False, bias=True)
    cov_inv = np.linalg.inv(cov)
    var_inv = 1.0 / samples.var(axis=0, ddof=0)

    return dict(t=t, y=y, cov_inv=cov_inv, var_inv=var_inv)

# ---- prior term ----
def test_prior_term_none_is_zero():
    assert _prior_term({"A0": 1.0}, None) == 0.0
    assert _prior_term({"A0": 1.0}, {}) == 0.0

def test_prior_term_value():
    assert _prior_term({"E1": 1.2}, {"E1": (1.0, 0.1)}) == pytest.approx(4.0)

def test_prior_term_at_mean_is_zero():
    assert _prior_term({"E1": 1.0}, {"E1": (1.0, 0.1)}) == pytest.approx(0.0)

# ---- cost factory attributes ----
def test_cost_has_errordef_and_ndata(synthetic):
    s = synthetic
    cost = _compute_correlated_chi2(s["t"], s["y"], model, s["cov_inv"], P_TRUE)
    assert cost.errordef == 1.0
    assert cost.ndata == len(s["t"])

def test_cost_ndata_counts_priors(synthetic):
    s = synthetic
    priors = {"E1": (1.2, 0.3), "A1": (0.3, 0.1)}
    cost = _compute_correlated_chi2(s["t"], s["y"], model, s["cov_inv"], P_TRUE, priors)
    assert cost.ndata == len(s["t"]) + 2

def test_cost_at_truth_is_small(synthetic):
    s = synthetic
    cost = _compute_correlated_chi2(s["t"], s["y"], model, s["cov_inv"], P_TRUE)
    assert cost(*P_TRUE.values()) < 5 * len(s["t"])

# ---- fits recover truth ----
def test_correlated_recovers_truth(synthetic):
    s = synthetic
    p0 = {"A0": 1.1, "E0": 0.4, "A1": 0.2, "E1": 1.0}
    m = minimise(s["t"], s["y"], model, cov_inv=s["cov_inv"],
                 correlated=True, p0=p0, raise_failure=True)
    assert m.valid
    for key, truth in P_TRUE.items():
        assert m.values[key] == pytest.approx(truth, abs=0.05), f"{key} off"

def test_uncorrelated_recovers_truth(synthetic):
    s = synthetic
    p0 = {"A0": 1.1, "E0": 0.4, "A1": 0.2, "E1": 1.0}
    m = minimise(s["t"], s["y"], model, var_inv=s["var_inv"],
                 correlated=False, p0=p0, raise_failure=True)
    assert m.valid
    assert m.values["E0"] == pytest.approx(P_TRUE["E0"], abs=0.05)

# ---- input validation ----
def test_correlated_without_cov_raises(synthetic):
    s = synthetic
    with pytest.raises(ValueError, match="inverse covariance"):
        minimise(s["t"], s["y"], model, correlated=True, p0=P_TRUE)

def test_uncorrelated_without_var_raises(synthetic):
    s = synthetic
    with pytest.raises(ValueError, match="inverse variance"):
        minimise(s["t"], s["y"], model, correlated=False, p0=P_TRUE)

# ---- failure handling ----
def test_raise_failure_false_returns_object(synthetic):
    s = synthetic
    p0 = {"A0": 1e6, "E0": 50.0, "A1": -1e6, "E1": 80.0}
    m = minimise(s["t"], s["y"], model, cov_inv=s["cov_inv"],
                 correlated=True, p0=p0, iterations=1,
                 use_simplex=False, raise_failure=False)
    assert hasattr(m, "valid")

# ---- limits ----
def test_limits_applied(synthetic):
    s = synthetic
    p0 = {"A0": 1.1, "E0": 0.4, "A1": 0.2, "E1": 1.0}
    limits = {"E0": (0, 1), "E1": (0, 5)}
    m = minimise(s["t"], s["y"], model, cov_inv=s["cov_inv"],
                 correlated=True, p0=p0, p0_limits=limits)
    assert m.limits["E0"] == (0, 1)
    assert m.values["E0"] >= 0

import numpy as np
import pytest
from chigrad.fit.result import FitConfig, FitResult, FitRunResult
from chigrad.fit.minimise import execute, minimise, _build_result, ConvergenceError

def model(t, A0, E0, A1, E1):
    return A0 * np.exp(-E0 * t) + A1 * np.exp(-E1 * t)

P_TRUE = {"A0": 1.0, "E0": 0.5, "A1": 0.3, "E1": 1.2}

@pytest.fixture
def data():
    """Synthetic jackknife ensemble + frozen weight matrices."""
    rng = np.random.default_rng(0)
    t = np.arange(0, 16)
    y_true = model(t, **P_TRUE)
    samples = y_true[None, :] + rng.normal(0, 0.01*y_true + 1e-4, size=(100, len(t)))
    cov = np.cov(samples, rowvar=False, bias=True)
    return dict(t=t, samples=samples, cov_inv=np.linalg.inv(cov),
                var_inv=1.0/samples.var(axis=0))

START = {"A0": 1.1, "E0": 0.4, "A1": 0.2, "E1": 1.0}

# config validation
def test_config_rejects_bad_strategy():
    with pytest.raises(ValueError, match="strategy"):
        FitConfig(param_start=START, strategy=5)

def test_config_rejects_empty_params():
    with pytest.raises(ValueError, match="start parameters"):
        FitConfig(param_start={})

# single fit
def test_minimise_correlated_recovers_truth(data):
    cfg = FitConfig(param_start=START, correlated=True)
    m = minimise(cfg, data["t"], data["samples"].mean(0), model, cov_inv=data["cov_inv"])
    assert m.valid
    for k, truth in P_TRUE.items():
        assert m.values[k] == pytest.approx(truth, abs=0.05)

def test_minimise_uncorrelated_recovers_E0(data):
    cfg = FitConfig(param_start=START, correlated=False)
    m = minimise(cfg, data["t"], data["samples"].mean(0), model, var_inv=data["var_inv"])
    assert m.valid
    assert m.values["E0"] == pytest.approx(P_TRUE["E0"], abs=0.05)

def test_minimise_missing_cov_raises(data):
    cfg = FitConfig(param_start=START, correlated=True)
    with pytest.raises(ValueError, match="inverse covariance"):
        minimise(cfg, data["t"], data["samples"].mean(0), model)

# full run
def test_execute_central_only(data):
    cfg = FitConfig(param_start=START, correlated=True, execute_resample=False)
    run = execute(cfg, data["t"], data["samples"], model, cov_inv=data["cov_inv"])
    assert run.central is not None
    assert run.resample is None
    assert run.param_err is None
    assert run.param_final is None
    assert run.param_est["E0"] == pytest.approx(P_TRUE["E0"], abs=0.05)

def test_execute_with_resamples(data):
    cfg = FitConfig(param_start=START, correlated=True, execute_resample=True)
    run = execute(cfg, data["t"], data["samples"], model, cov_inv=data["cov_inv"])
    assert run.nres == 100
    assert run.nres_valid == 100
    val, err = run.param_final["E0"]
    assert val == pytest.approx(P_TRUE["E0"], abs=0.05)
    assert err > 0

def test_execute_param_err_keys_match(data):
    cfg = FitConfig(param_start=START, correlated=True, execute_resample=True)
    run = execute(cfg, data["t"], data["samples"], model, cov_inv=data["cov_inv"])
    assert set(run.param_err.keys()) == set(P_TRUE.keys())

def test_execute_goodness_of_fit(data):
    cfg = FitConfig(param_start=START, correlated=True, execute_resample=True)
    run = execute(cfg, data["t"], data["samples"], model, cov_inv=data["cov_inv"])
    assert 0.0 <= run.pvalue <= 1.0
    assert run.ndof == len(data["t"]) - 4
    assert np.isfinite(run.aic) and np.isfinite(run.aicc)

def test_per_resample_properties(data):
    cfg = FitConfig(param_start=START, correlated=True, execute_resample=True)
    run = execute(cfg, data["t"], data["samples"], model, cov_inv=data["cov_inv"])
    for r in run.resample:
        assert np.isfinite(r.pvalue)
        assert r.ndof == len(data["t"]) - 4
    assert run.resample_chi2dof.shape == (100,)


def test_central_value_resample_error_split(data):
    cfg = FitConfig(param_start=START, correlated=True, execute_resample=True)
    run = execute(cfg, data["t"], data["samples"], model, cov_inv=data["cov_inv"])
    # value == central fit's value (not resample mean)
    assert run.param_est["E0"] == run.central.param_est["E0"]
    # error == jackknife spread (not central's HESSE)
    assert run.param_err["E0"] != run.param_hess["E0"]   # different sources

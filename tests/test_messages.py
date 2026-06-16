"""Tests for the message() logging calls inside minimise."""
import numpy as np
import pytest
from unittest.mock import patch

from chigrad.fit.minimise import minimise        # ← adjust to your real path


def model(t, A0, E0, A1, E1):
    return A0 * np.exp(-E0 * t) + A1 * np.exp(-E1 * t)

P_TRUE = {"A0": 1.0, "E0": 0.5, "A1": 0.3, "E1": 1.2}


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(0)
    t = np.arange(0, 16)
    y_true = model(t, **P_TRUE)
    samples = y_true[None, :] + rng.normal(0, 0.01 * y_true + 1e-4, size=(200, len(t)))
    y = samples.mean(axis=0)
    cov_inv = np.linalg.inv(np.cov(samples, rowvar=False, bias=True))
    return dict(t=t, y=y, cov_inv=cov_inv)


# Patch message where it is USED, not where defined:
MSG = "chigrad.fit.minimise.message"             # ← adjust to your real path


def _run(s, **kw):
    p0 = {"A0": 1.1, "E0": 0.4, "A1": 0.2, "E1": 1.0}
    return minimise(s["t"], s["y"], model, cov_inv=s["cov_inv"],
                    correlated=True, p0=p0, **kw)


def test_simplex_message_emitted_when_used(synthetic):
    with patch(MSG) as msg:
        _run(synthetic, use_simplex=True)
    logged = " ".join(str(c.args[0]) for c in msg.call_args_list)
    assert "SIMPLEX" in logged

def test_simplex_message_absent_when_disabled(synthetic):
    with patch(MSG) as msg:
        _run(synthetic, use_simplex=False)
    logged = " ".join(str(c.args[0]) for c in msg.call_args_list)
    assert "SIMPLEX" not in logged

def test_migrad_message_includes_settings(synthetic):
    with patch(MSG) as msg:
        _run(synthetic, strategy=2, tolerance=0.05, iterations=5000)
    logged = " ".join(str(c.args[0]) for c in msg.call_args_list)
    assert "MIGRAD" in logged
    assert "Strategy 2" in logged
    assert "0.05" in logged
    assert "5000" in logged

def test_success_message_on_valid_fit(synthetic):
    with patch(MSG) as msg:
        _run(synthetic)
    logged = " ".join(str(c.args[0]) for c in msg.call_args_list)
    assert "terminated successfully" in logged

def test_hesse_messages_on_valid_fit(synthetic):
    with patch(MSG) as msg:
        _run(synthetic)
    logged = " ".join(str(c.args[0]) for c in msg.call_args_list)
    assert "Computing Hesse matrix." in logged
    assert "Hesse matrix successfully computed." in logged

def test_accuracy_message_branch(synthetic):
    with patch(MSG) as msg:
        _run(synthetic)
    logged = " ".join(str(c.args[0]) for c in msg.call_args_list)
    assert ("Accurate covariance achieved." in logged
            or "inaccurate covariance" in logged)

def test_limits_message_emitted(synthetic):
    with patch(MSG) as msg:
        _run(synthetic, p0_limits={"E0": (0, 1)})
    logged = " ".join(str(c.args[0]) for c in msg.call_args_list)
    assert "limits imposed" in logged

def test_limits_message_absent_without_limits(synthetic):
    with patch(MSG) as msg:
        _run(synthetic)
    logged = " ".join(str(c.args[0]) for c in msg.call_args_list)
    assert "limits imposed" not in logged

def test_message_called_at_least_once(synthetic):
    with patch(MSG) as msg:
        _run(synthetic)
    assert msg.call_count > 0

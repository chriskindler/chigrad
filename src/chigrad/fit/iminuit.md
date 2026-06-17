# iminuit Reference for χ² Correlator Fits

A practical reference for the iminuit calls used in the `chigrad` fitting
pipeline. Verified against **iminuit 2.32**. Organised by *when* you use each
call, following the order of a single fit.

---

## 1. Cost Function Setup

These are attributes you set on your **cost function** (not on `Minuit`)
before constructing the minimiser.

| Call | When | Notes |
|---|---|---|
| `cost.errordef = Minuit.LEAST_SQUARES` | always, for χ² | `= 1.0`. Δχ² = 1 defines 1σ. **Required** for correct errors. |
| `cost.errordef = Minuit.LIKELIHOOD` | only for −log L | `= 0.5`. Do **not** use for a plain χ². |
| `cost.ndata = len(t) + n_priors` | always | Lets iminuit compute `reduced_chi2` and `ndof`. Count priors as data. |

```python
def cost(*vals):
    p = dict(zip(param_keys, vals))
    r = y - f(t, **p)
    return r @ cov_inv @ r

cost.errordef = iminuit.Minuit.LEAST_SQUARES   # 1.0 → χ²
cost.ndata    = len(t)                         # for reduced_chi2
```

> **Pitfall.** Setting `errordef = 0.5` on a χ² cost shrinks all HESSE errors by
> √2. The `0.5` only belongs on a negative-log-likelihood. iminuit's own
> `minimize.py` uses `0.5` because it assumes scipy-style objectives — not your case.

---

## 2. Constructing the Minuit Object

| Call | When | Notes |
|---|---|---|
| `Minuit(cost, *values, name=names)` | always | Named params → access by name later. `*values` are starting points. |
| `Minuit(cost, **param_dict)` | alternative | Pass `A0=1.0, E0=0.5, ...` directly. |

```python
m = iminuit.Minuit(cost, *param_start.values(),
                   name=tuple(param_start.keys()))
```

The `name=` argument is what enables `m.values["E0"]`, `m.limits["E0"]`, etc.
Without it parameters are positional only.

---

## 3. Configuring the Fit (before minimising)

| Call | When | Notes |
|---|---|---|
| `m.tol = 0.1` | optional | Convergence tolerance (EDM threshold scaling). |
| `m.strategy = 0/1/2` | optional | 0 fast/rough, 1 balanced (default), 2 slow/accurate Hessian. |
| `m.limits["E0"] = (0, None)` | when bounding | Box constraints; `None` = unbounded on that side. |
| `m.limits["E1"] = (m0, None)` | ordering | Enforce `E1 > E0` to break exp-fit permutation symmetry. |
| `m.fixed["A1"] = True` | when fixing | Hold a parameter constant during the fit. |
| `m.values["E0"] = 0.5` | manual start | Override a starting value. |

```python
m.strategy = 1
m.tol      = 0.1
for name, lim in param_limit.items():
    m.limits[name] = lim
```

---

## 4. Minimisation

| Call | When | Returns | Notes |
|---|---|---|---|
| `m.simplex()` | cold start / robust | `m` | Gradient-free Nelder–Mead. Locates the basin. **Never the final step** — lax tolerance, no covariance. |
| `m.migrad(ncall=N)` | always | `m` | Gradient-based. The workhorse. `ncall` caps function evaluations. |
| `m.simplex().migrad()` | multi-exp / noisy | `m` | Recommended chain: simplex finds basin, migrad polishes. |
| `m.scan(ncall=N)` | rarely | `m` | Brute-force grid scan; only for ≤ few params. |

```python
if use_simplex:
    m.simplex()          # robust pre-stage (cold start)
m.migrad(ncall=10000)    # precise minimisation
```

> **Why chain.** Plain migrad from a poor start can converge to the *wrong*
> minimum on multi-exponential fits (it reports `valid=True` while being wrong).
> simplex first reaches the right basin. For **warm-started** resample fits
> (starting from converged central params) you can skip simplex.

---

## 5. Error Estimation (after minimising)

| Call | When | Returns | Notes |
|---|---|---|---|
| `m.hesse()` | for symmetric errors | `m` | Computes the Hessian → parameter covariance. Run **only if `m.valid`**. |
| `m.minos()` | for asymmetric errors | `m` | Profile-likelihood errors. Slower; important for masses with skewed χ². Fills `m.merrors`. |

```python
if m.valid:
    m.hesse()            # parabolic errors → m.covariance, m.errors
    # m.minos()          # asymmetric errors → m.merrors  (optional)
```

> **Order matters.** Check `m.valid` *before* `m.hesse()`. Running HESSE on a
> failed fit wastes effort and can emit a meaningless/forced covariance.

---

## 6. Extracting Results

### Parameters & errors

| Attribute | Type | Meaning |
|---|---|---|
| `m.values` | `ValueView` | Best-fit θ̂. `np.asarray(m.values)` or `m.values["E0"]`. |
| `m.errors` | `ErrorView` | Symmetric HESSE errors √V_aa. `m.errors["E0"]`. |
| `m.covariance` | `Matrix` or `None` | Full covariance V (npar×npar). `None` if invalid/no HESSE. |
| `m.merrors` | dict-like | MINOS asymmetric errors (only after `m.minos()`). |
| `m.parameters` | tuple | Parameter names, e.g. `("A0","E0","A1","E1")`. |

```python
params = dict(zip(m.parameters, m.values))          # {"A0":..., "E0":...}
hess   = dict(zip(m.parameters, m.errors))          # HESSE errors
cov    = np.asarray(m.covariance) if m.valid else None
```

### The minimum & goodness of fit

| Attribute | Type | Meaning |
|---|---|---|
| `m.fval` | float | χ² at the minimum. |
| `m.fmin.reduced_chi2` | float | χ²/ndof (needs `cost.ndata` set). |
| `m.ndof` | int | Degrees of freedom (`ndata − nfit`). |
| `m.npar` | int | Total number of parameters. |
| `m.nfit` | int | Number of **floating** parameters (excludes fixed). |
| `m.nfcn` | int | Function evaluations used. |
| `m.ngrad` | int | Gradient evaluations used. |

```python
chi2     = m.fval
ndof     = m.ndof
chi2_red = m.fmin.reduced_chi2
```

> For χ²/dof prefer deriving `m.fval / (your ndata − m.nfit)` yourself if you use
> priors, so the dof definition stays consistent with your `cost.ndata`.

---

## 7. Convergence & Quality Flags

### Top-level booleans

| Attribute | Meaning | Use |
|---|---|---|
| `m.valid` | Fit converged and is valid. | Gate HESSE; record per fit. |
| `m.accurate` | Covariance is accurate. | Cross-check covariance quality. |

### Fine-grained flags (`m.fmin.*`)

| Attribute | Meaning | Why it matters |
|---|---|---|
| `m.fmin.edm` | Estimated distance to minimum. | Small = truly converged. |
| `m.fmin.is_valid` | Same as `m.valid`. | — |
| `m.fmin.is_above_max_edm` | EDM too large. | Didn't really converge. |
| `m.fmin.has_reached_call_limit` | Hit `ncall`. | Ran out of iterations. |
| `m.fmin.hesse_failed` | HESSE computation failed. | Errors unavailable. |
| `m.fmin.has_posdef_covar` | Covariance is positive-definite. | Healthy. |
| `m.fmin.has_made_posdef_covar` | Covariance was **forced** posdef. | **Silent-failure flag** — errors unreliable even if `valid`. |
| `m.fmin.has_accurate_covar` | Same as `m.accurate`. | — |
| `m.fmin.has_parameters_at_limit` | A param sits on a bound. | Limit may be distorting the fit. |

```python
status = dict(
    valid           = bool(m.valid),
    accurate_cov    = bool(m.accurate),
    edm             = float(m.fmin.edm),
    nfcn            = int(m.nfcn),
    has_made_posdef = bool(m.fmin.has_made_posdef_covar),  # catch silent failures
)
```

> **The underrated flag.** `has_made_posdef_covar` is `True` when Minuit had to
> force the covariance positive-definite — a quiet sign the errors are
> untrustworthy even when `valid` is `True`. Worth recording.

---

## 8. Re-running / Adjusting

All minimisation methods return `m`, so they chain and can be re-invoked.

| Call | Effect |
|---|---|
| `m.migrad()` again | Re-minimise from current values (e.g. after changing limits). |
| `m.reset()` | Reset to initial state. |
| `m.fixed["E0"] = False` | Release a fixed parameter, then re-`migrad()`. |
| `m.values["E0"] = 0.6` | Nudge a value, then re-`migrad()`. |

---

## Typical Pipeline (chigrad)

```python
# 1. cost setup
cost.errordef = iminuit.Minuit.LEAST_SQUARES
cost.ndata    = len(t) + n_priors

# 2. construct
m = iminuit.Minuit(cost, *start.values(), name=tuple(start.keys()))

# 3. configure
m.tol      = tolerance
m.strategy = strategy
for name, lim in (limits or {}).items():
    m.limits[name] = lim

# 4. minimise (two-stage)
if use_simplex:
    m.simplex()
m.migrad(ncall=iterations)

# 5. errors — only if valid
if m.valid:
    m.hesse()

# 6. extract into a plain result container
result = dict(
    param_est   = dict(zip(m.parameters, m.values)),
    param_hess  = dict(zip(m.parameters, m.errors)) if m.valid else None,
    covariance  = np.asarray(m.covariance) if m.valid and m.covariance is not None else None,
    chi2        = float(m.fval),
    npar        = m.nfit,
    valid       = bool(m.valid),
    accurate    = bool(m.accurate),
    edm         = float(m.fmin.edm),
    nfcn        = int(m.nfcn),
    made_posdef = bool(m.fmin.has_made_posdef_covar),
)
```

---

## Decisions

| Situation | Calls |
|---|---|
| Single exp, good start | `m.migrad()` → `m.hesse()` |
| Multi-exp / noisy / cold start | `m.simplex()` → `m.migrad()` → `m.hesse()` |
| Warm-started resample fit | `m.migrad()` only (skip simplex), HESSE optional |
| Need asymmetric mass errors | add `m.minos()` after `m.hesse()` |
| Enforce E0 < E1 | `m.limits["E1"] = (E0_val, None)` before migrad |
| Fit won't converge | raise `m.strategy`, widen `ncall`, add `m.simplex()` first |
| Covariance looks wrong | check `m.fmin.has_made_posdef_covar` and `m.accurate` |

---

## Potential Risks

1. **`errordef`**: `1.0` for χ², `0.5` for −log L. Wrong value → errors off by √2.
2. **HESSE before validity check**: always `if m.valid: m.hesse()`.
3. **`m.covariance` can be `None`**: guard before `np.asarray`.
4. **`m.nfit` vs `m.npar`**: use `nfit` (floating params) for dof, not `npar`.
5. **simplex is never terminal**: always follow with migrad — its tolerance is lax.
6. **Don't store the `Minuit` object**: extract plain values; `Minuit` isn't serializable.

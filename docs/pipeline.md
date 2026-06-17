Reference for the correlator-fitting pipeline implemented in `chigrad`
(`minimise.py`, `result.py`). Covers the $\chi^2$ functional, two-stage
minimisation, resampling, and how fit quality is evaluated differently for
**correlated** vs **uncorrelated** fits.

---

## Notation

| Symbol | Meaning | Index | Code |
|---|---|---|---|
| $N_{\mathrm{dat}}$ | data points (time slices) | $i,j$ | `len(t)` |
| $N_{\mathrm{par}}$ | fit parameters | $a,b$ | `m.nfit` |
| $N_{\mathrm{res}}$ | resamples | $k$ | `nres` |
| $N_{\mathrm{dof}}$ | degrees of freedom | — | `ndof` |
| $y_i^{(k)}$ | observable $i$ on resample $k$ | — | `y[k,i]` |
| $\bar y_i$ | central (mean) data | — | `y_cen` |
| $f(t_i,\theta)$ | model | — | `f(t,**p)` |
| $\theta$ | parameter vector | — | `param_est` |
| $C_{ij}$ | data covariance | — | — |
| $W=C^{-1}$ | weight matrix | — | `cov_inv` |

---

## Inputs

**Central value** of the ensemble, computed as `y_cen = np.mean(y, axis=0)`:

$$\bar y_i = \frac{1}{N_{\mathrm{res}}}\sum_{k=1}^{N_{\mathrm{res}}} y_i^{(k)}$$

**Frozen weight matrix**, built once from the full ensemble and reused for
every fit (central and all resamples). Correlated uses `cov_inv`, uncorrelated
uses `var_inv`:

$$W = C^{-1} \qquad\text{or}\qquad w_i = \frac{1}{\sigma_i^2}$$

---

## 1. Cost Function

Residual vector:

$$r_i(\theta) = \bar y_i - f(t_i,\theta)$$

**Correlated** $\chi^2$, code `np.einsum("i,ij,j->", r, cov_inv, r)`:

$$\chi^2(\theta) = \sum_{i,j=1}^{N_{\mathrm{dat}}} r_i\,W_{ij}\,r_j + \chi^2_{\mathrm{prior}}$$

**Uncorrelated** $\chi^2$, code `np.einsum("i,i->", var_inv, r**2)`:

$$\tilde\chi^2(\theta) = \sum_{i=1}^{N_{\mathrm{dat}}} w_i\,r_i^2 + \chi^2_{\mathrm{prior}}$$

**Prior term** (augmented $\chi^2$, for parameters in the prior set $\mathcal{P}$):

$$\chi^2_{\mathrm{prior}} = \sum_{a\in\mathcal{P}}\left(\frac{\theta_a-\tilde\theta_a}{\tilde\sigma_a}\right)^2$$

Cost attributes: `errordef = 1` (so $\Delta\chi^2=1$ defines $1\sigma$) and
`ndata` $= N_{\mathrm{dat}}+|\mathcal{P}|$ (priors count as data points).

---

## 2. Minimisation

The optimisation problem:

$$\hat\theta = \arg\min_{\theta}\,\chi^2(\theta)$$

Two-stage (when `enable_simplex` is set), where `m.simplex()` locates the basin
and `m.migrad()` refines:

$$\theta_0 \;\longrightarrow\; \theta_{\mathrm{rough}} \;\longrightarrow\; \hat\theta$$

At the minimum the gradient vanishes:

$$\frac{\partial\chi^2}{\partial\theta_a}\bigg|_{\hat\theta} = -2\sum_{i,j} J_{ia}\,W_{ij}\,r_j = 0, \qquad J_{ia}=\frac{\partial f(t_i)}{\partial\theta_a}$$

---

## 3. HESSE Covariance (cross-check)

The Hessian of $\chi^2$ at the minimum (Gauss–Newton form, dropping the small
residual-weighted second-derivative term):

$$H_{ab} = \frac{1}{2}\frac{\partial^2\chi^2}{\partial\theta_a\,\partial\theta_b}\bigg|_{\hat\theta} \approx \sum_{i,j} J_{ia}\,W_{ij}\,J_{jb}$$

Parameter covariance and symmetric errors (`param_hess`):

$$V^{\mathrm{hess}} = H^{-1}, \qquad \sigma_a^{\mathrm{hess}} = \sqrt{V^{\mathrm{hess}}_{aa}}$$

Retained **only as a cross-check** — the quoted error comes from resampling.

---

## 4. Resampling

Each resample is refit with the **same frozen** $W$, warm-started from the
central result:

$$\hat\theta^{(k)} = \arg\min_{\theta} \sum_{i,j}\big[y_i^{(k)}-f(t_i,\theta)\big]\,W_{ij}\,\big[y_j^{(k)}-f(t_j,\theta)\big]$$

for $k=1,\dots,N_{\mathrm{res}}$. The data $y^{(k)}$ varies; $W$ does not.

---

## 5. Quoted Parameters and Errors

**Quoted value** — from the central fit (`param_est`):

$$\hat\theta_a = \hat\theta_a^{\mathrm{cen}}$$

**Quoted error** — from the resample spread.

Jackknife:

$$\sigma_a^{\mathrm{jkn}} = \sqrt{\frac{N_{\mathrm{res}}-1}{N_{\mathrm{res}}}\sum_{k}\big(\hat\theta_a^{(k)}-\overline{\hat\theta}_a\big)^2}, \qquad \overline{\hat\theta}_a=\frac{1}{N_{\mathrm{res}}}\sum_k\hat\theta_a^{(k)}$$

Bootstrap (std with `ddof=1`):

$$\sigma_a^{\mathrm{bst}} = \mathrm{std}\big(\hat\theta_a^{(k)}\big)$$

**Headline result** (`param_final`):

$$\theta_a = \hat\theta_a^{\mathrm{cen}} \pm \sigma_a^{\mathrm{resample}}$$

---

## 6. Resample Parameter Covariance (error bands)

With deviations $d_a^{(k)}=\hat\theta_a^{(k)}-\overline{\hat\theta}_a$:

Jackknife, code `(n-1)/n * d.T @ d`:

$$V^{\mathrm{res}}_{ab} = \frac{N_{\mathrm{res}}-1}{N_{\mathrm{res}}}\sum_k d_a^{(k)} d_b^{(k)}$$

Bootstrap, code `np.cov(..., ddof=1)`:

$$V^{\mathrm{res}}_{ab} = \frac{1}{N_{\mathrm{res}}-1}\sum_k d_a^{(k)} d_b^{(k)}$$

The quoted error is the diagonal, $\sigma_a^{\mathrm{resample}}=\sqrt{V^{\mathrm{res}}_{aa}}$.
The model error band on a fine grid propagates this through the Jacobian:

$$\sigma_f^2(t) = J(t)^\top\, V^{\mathrm{res}}\, J(t), \qquad J_a(t)=\frac{\partial f(t)}{\partial\theta_a}\bigg|_{\hat\theta^{\mathrm{cen}}}$$

---

# Judging Fit Quality

This is where correlated and uncorrelated fits **differ fundamentally**.

## Degrees of Freedom (both)

$$N_{\mathrm{dof}} = N_{\mathrm{dat}} - N_{\mathrm{par}}$$

## Correlated Fits — $\chi^2$ Is a True $\chi^2$

When the full covariance is used, the minimised statistic genuinely follows a
$\chi^2$ distribution with $N_{\mathrm{dof}}$ degrees of freedom (for a correct,
approximately linear model). All standard goodness-of-fit machinery applies.

Reduced $\chi^2$, where a value near 1 indicates a good fit:

$$\frac{\chi^2_{\min}}{N_{\mathrm{dof}}} \approx 1$$

p-value (regularised upper incomplete gamma $Q$, via `gammaincc`):

$$p = Q\!\left(\frac{N_{\mathrm{dof}}}{2},\,\frac{\chi^2_{\min}}{2}\right) = \int_{\chi^2_{\min}}^{\infty} f_{\chi^2}(x;N_{\mathrm{dof}})\,dx$$

Both $\chi^2/N_{\mathrm{dof}}$ and $p$ are **statistically valid** here.

## Uncorrelated Fits — $\chi^2$ Is NOT a True $\chi^2$

The uncorrelated statistic

$$\tilde\chi^2 = \sum_i \frac{r_i^2}{\sigma_i^2}$$

ignores the off-diagonal covariance, but the data points **are** correlated
(measured on the same configurations). Consequently:

$$\mathbb{E}[\tilde\chi^2] \neq N_{\mathrm{dof}}, \qquad \tilde\chi^2 \not\sim \chi^2_{N_{\mathrm{dof}}}$$

Therefore, for uncorrelated fits, **neither of the following is valid**:

$$\frac{\tilde\chi^2}{N_{\mathrm{dof}}}\approx 1 \quad\text{(not a valid GoF criterion)}$$

$$p = Q\!\left(\tfrac{N_{\mathrm{dof}}}{2},\tfrac{\tilde\chi^2}{2}\right) \quad\text{(not a valid p-value)}$$

The parameter estimates $\hat\theta$ remain **unbiased**, and the resample
errors remain **valid**, because they do not rely on interpreting $\tilde\chi^2$
as $\chi^2$-distributed. Only the *goodness-of-fit interpretation* of
$\tilde\chi^2$ breaks.

## Information Criteria (both, with caution)

$$\mathrm{AIC} = \chi^2_{\min} + 2(N_{\mathrm{par}}-N_{\mathrm{dat}})$$

$$\mathrm{AICc} = \mathrm{AIC} + \frac{2N_{\mathrm{par}}(N_{\mathrm{par}}+1)}{N_{\mathrm{dat}}-N_{\mathrm{par}}-1}$$

AIC differences are meaningful for **model comparison at fixed correlation
treatment**; comparing AIC between a correlated and an uncorrelated fit is not
meaningful, because the underlying $\chi^2$ statistics differ.

---

# Stability Diagnostics Across Resamples

Each resample fit yields a full set of statistics. Their **distributions**
diagnose stability — but *which* distribution to trust depends on the
correlation type.

## Parameter Distribution (valid for BOTH)

The most robust, correlation-agnostic diagnostic is the spread of the
parameters themselves. For each parameter, the resampled values
$\{\hat\theta_a^{(k)}\}_{k=1}^{N_{\mathrm{res}}}$ should, for a healthy fit, be
approximately Gaussian about the central value:

$$\hat\theta_a^{(k)} \sim \mathcal{N}\!\big(\hat\theta_a^{\mathrm{cen}},\,\sigma_a^2\big)$$

Diagnostics to compute per parameter:

| Quantity | Formula | Healthy value |
|---|---|---|
| mean | $\overline{\hat\theta}_a$ | $\approx \hat\theta_a^{\mathrm{cen}}$ |
| std | $\sigma_a^{\mathrm{resample}}$ | matches `param_err` |
| **bias** | $\overline{\hat\theta}_a - \hat\theta_a^{\mathrm{cen}}$ | $\approx 0$ |
| outliers | count of $k$ with $\lvert\hat\theta_a^{(k)}-\overline{\hat\theta}_a\rvert>3\sigma_a$ | few / none |

**Warning signs:**

- **Bimodal** — resamples landing in different minima (e.g. unbroken
  permutation symmetry, the $(A_0,E_0)$ and $(A_1,E_1)$ labels swapping). Fix
  with ordering limits $E_0 < E_1$.
- **Skewed** — parameter near a boundary or poorly constrained (often where
  $\sigma_a^{\mathrm{resample}} \gg \sigma_a^{\mathrm{hess}}$).
- **Large bias** — model nonlinearity; central value and resample mean disagree.

## $\chi^2$/dof Spread (correlated ONLY)

For correlated fits, the spread of the per-resample reduced $\chi^2$ is a valid
secondary stability check:

$$\mathrm{std}_k\!\left(\frac{\chi^2_{(k)}}{N_{\mathrm{dof}}}\right)$$

- **Small spread** — consistent fit quality across resamples, so stable.
- **Large spread** — fragile fit; small data fluctuations change the regime.

For **uncorrelated** fits this is **not** valid (the per-resample $\tilde\chi^2$
is not a true $\chi^2$); rely on the parameter distribution instead.

## Summary Table — What to Trust

| Diagnostic | Correlated | Uncorrelated |
|---|---|---|
| $\chi^2/N_{\mathrm{dof}}\approx 1$ | valid | invalid |
| p-value | valid | invalid |
| $\chi^2$/dof spread across resamples | valid | invalid |
| parameter distribution | valid | valid |
| resample errors | valid | valid |
| HESSE vs resample agreement | valid | valid |

---

# The Pipeline (ordered)

1. **Average** ensemble into central data $\bar y_i$.
2. **Build cost** from frozen $W$, with `errordef = 1` and `ndata` set.
3. **Central fit**: simplex, then migrad, then hesse, giving $\hat\theta^{\mathrm{cen}}$ and $V^{\mathrm{hess}}$.
4. **Extract** `FitResult` (params, hess errors, $\chi^2$, residuals).
5. **Resample loop**: refit each $y^{(k)}$, warm-started, same frozen $W$, giving $\{\hat\theta^{(k)}\}$.
6. **Quoted error**: jackknife/bootstrap spread $\sigma_a^{\mathrm{resample}}$.
7. **Goodness of fit**: $\chi^2/N_{\mathrm{dof}}$ and $p$ — **valid only if correlated**.
8. **Stability**: parameter distribution (always); $\chi^2$/dof spread (correlated only).

---

# The Core Design Equation

The value comes from the central fit; the error from the resample distribution:

$$\theta_a = \hat\theta_a^{\mathrm{cen}} \pm \sqrt{\frac{N_{\mathrm{res}}-1}{N_{\mathrm{res}}}\sum_k\big(\hat\theta_a^{(k)}-\overline{\hat\theta}_a\big)^2}$$

with $V^{\mathrm{hess}}=H^{-1}$ retained as a cross-check: when
$\sigma_a^{\mathrm{hess}}\approx\sigma_a^{\mathrm{resample}}$ the likelihood is
approximately Gaussian. The goodness-of-fit interpretation is only valid when
the full covariance is used (correlated fits).

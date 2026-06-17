Reference for the correlator-fitting pipeline implemented in `chigrad`
(`minimise.py`, `result.py`). Covers the $\chi^22$ functional, two-stage
minimisation, resampling, and how fit quality is evaluated differently for **correlated** vs **uncorrelated** fits.

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
| $f(t_i,\boldsymbol\theta)$ | model | — | `f(t,**p)` |
| $\boldsymbol\theta$ | parameter vector | — | `param_est` |
| $C_{ij}$ | data covariance | — | — |
| $W=C^{-1}$ | weight matrix | — | `cov_inv` |

---

## Inputs

**Central value** of the ensemble:

$$\bar y_i = \frac{1}{N_{\mathrm{res}}}\sum_{k=1}^{N_{\mathrm{res}}} y_i^{(k)}
\qquad\Leftrightarrow\qquad \texttt{y\_cen = np.mean(y, axis=0)}$$

**Frozen weight matrix**, built once from the full ensemble and reused for
every fit (central and all resamples):

$$W = C^{-1}\ (\texttt{cov\_inv},\ \text{correlated})
\qquad\text{or}\qquad
w_i = \frac{1}{\sigma_i^2}\ (\texttt{var\_inv},\ \text{uncorrelated})$$

---

## 1. Cost Function

Residual vector:

$$r_i(\boldsymbol\theta) = \bar y_i - f(t_i,\boldsymbol\theta)$$

**Correlated** χ² (`np.einsum("i,ij,j->", r, cov_inv, r)`):

$$\chi^2(\boldsymbol\theta)
= \sum_{i,j=1}^{N_{\mathrm{dat}}} r_i\,W_{ij}\,r_j + \chi^2_{\text{prior}}$$

**Uncorrelated** χ² (`np.einsum("i,i->", var_inv, r**2)`):

$$\tilde\chi^2(\boldsymbol\theta)
= \sum_{i=1}^{N_{\mathrm{dat}}} w_i\,r_i^2 + \chi^2_{\text{prior}}$$

**Prior term** (augmented χ², for parameters in the prior set $\mathcal P$):

$$\chi^2_{\text{prior}}
= \sum_{a\in\mathcal P}\left(\frac{\theta_a-\tilde\theta_a}{\tilde\sigma_a}\right)^2$$

Cost attributes: $\texttt{errordef}=1$ (so $\Delta\chi^2=1$ defines $1\sigma$),
$\texttt{ndata}=N_{\mathrm{dat}}+|\mathcal P|$ (priors count as data points).

---

## 2. Minimisation

The optimisation problem:

$$\hat{\boldsymbol\theta}
= \operatorname*{arg\,min}_{\boldsymbol\theta}\chi^2(\boldsymbol\theta)$$

Two-stage (when `enable_simplex`):

$$\boldsymbol\theta_0
\xrightarrow{\texttt{m.simplex()}} \boldsymbol\theta_{\text{rough}}
\xrightarrow{\texttt{m.migrad()}} \hat{\boldsymbol\theta}$$

The simplex stage (gradient-free Nelder–Mead) locates the correct basin;
migrad (gradient-based) refines it. At the minimum the gradient vanishes,

$$\frac{\partial\chi^2}{\partial\theta_a}\bigg|_{\hat{\boldsymbol\theta}}
= -2\sum_{i,j} J_{ia}\,W_{ij}\,r_j = 0,
\qquad J_{ia}=\frac{\partial f(t_i)}{\partial\theta_a}.$$

---

## 3. HESSE Covariance (cross-check)

The Hessian of χ² at the minimum (Gauss–Newton form, dropping the
small residual-weighted second-derivative term):

$$H_{ab}
= \tfrac{1}{2}\frac{\partial^2\chi^2}{\partial\theta_a\partial\theta_b}
\bigg|_{\hat{\boldsymbol\theta}}
\approx \sum_{i,j} J_{ia}\,W_{ij}\,J_{jb}$$

Parameter covariance and symmetric errors:

$$V^{\text{hess}} = H^{-1},
\qquad \sigma_a^{\text{hess}} = \sqrt{V^{\text{hess}}_{aa}}
\quad(\texttt{param\_hess})$$

This is retained **only as a cross-check** — the quoted error comes from
resampling (Section 6).

---

## 4. Resampling

Each resample is refit with the **same frozen** $W$, warm-started from the
central result:

$$\hat{\boldsymbol\theta}^{(k)}
= \operatorname*{arg\,min}_{\boldsymbol\theta}
\sum_{i,j}\big[y_i^{(k)}-f(t_i,\boldsymbol\theta)\big]\,W_{ij}\,
\big[y_j^{(k)}-f(t_j,\boldsymbol\theta)\big],
\qquad k=1,\dots,N_{\mathrm{res}}$$

The data $y^{(k)}$ varies; the weight matrix $W$ does not.

---

## 5. Quoted Parameters and Errors

**Quoted value** — from the central fit:

$$\hat\theta_a = \hat\theta_a^{\text{cen}} \quad(\texttt{param\_est})$$

**Quoted error** — from the resample spread.

*Jackknife:*

$$\sigma_a^{\text{jkn}}
= \sqrt{\frac{N_{\mathrm{res}}-1}{N_{\mathrm{res}}}
\sum_{k}\big(\hat\theta_a^{(k)}-\overline{\hat\theta}_a\big)^2},
\qquad \overline{\hat\theta}_a=\frac{1}{N_{\mathrm{res}}}\sum_k\hat\theta_a^{(k)}$$

*Bootstrap:*

$$\sigma_a^{\text{bst}}
= \mathrm{std}_{\text{ddof}=1}\big(\hat\theta_a^{(k)}\big)$$

**Headline result** (`param_final`):

$$\boxed{\ \theta_a = \hat\theta_a^{\text{cen}} \pm \sigma_a^{\text{resample}}\ }$$

---

## 6. Resample Parameter Covariance (error bands)

With deviations $d_a^{(k)}=\hat\theta_a^{(k)}-\overline{\hat\theta}_a$:

*Jackknife* (`(n-1)/n * d.T @ d`):

$$V^{\text{res}}_{ab}
= \frac{N_{\mathrm{res}}-1}{N_{\mathrm{res}}}\sum_k d_a^{(k)} d_b^{(k)}$$

*Bootstrap* (`np.cov(..., ddof=1)`):

$$V^{\text{res}}_{ab}
= \frac{1}{N_{\mathrm{res}}-1}\sum_k d_a^{(k)} d_b^{(k)}$$

The quoted error is the diagonal: $\sigma_a^{\text{resample}}=\sqrt{V^{\text{res}}_{aa}}$.
The model error band on a fine grid $t$ propagates this through the Jacobian:

$$\sigma_f^2(t) = \boldsymbol J(t)^\top V^{\text{res}}\,\boldsymbol J(t),
\qquad J_a(t)=\frac{\partial f(t)}{\partial\theta_a}\bigg|_{\hat{\boldsymbol\theta}^{\text{cen}}}$$

---

# Judging Fit Quality

This is where correlated and uncorrelated fits **differ fundamentally**.

## Degrees of Freedom (both)

$$N_{\mathrm{dof}} = N_{\mathrm{dat}} - N_{\mathrm{par}}$$

## Correlated Fits — χ² Is a True χ²

When the full covariance is used, the minimised statistic genuinely follows a
χ² distribution with $N_{\mathrm{dof}}$ degrees of freedom (for a correct,
approximately linear model). All standard goodness-of-fit machinery applies.

**Reduced χ²:**

$$\frac{\chi^2_{\min}}{N_{\mathrm{dof}}} \approx 1 \quad\text{indicates a good fit.}$$

**p-value** (regularised upper incomplete gamma $Q$, via `gammaincc`):

$$p = Q\!\left(\frac{N_{\mathrm{dof}}}{2},\,\frac{\chi^2_{\min}}{2}\right)
= \int_{\chi^2_{\min}}^{\infty} f_{\chi^2}(x;N_{\mathrm{dof}})\,\mathrm{d}x$$

Both $\chi^2/N_{\mathrm{dof}}$ and $p$ are **statistically valid** here.

## Uncorrelated Fits — χ² Is NOT a True χ²

The uncorrelated statistic

$$\tilde\chi^2 = \sum_i \frac{r_i^2}{\sigma_i^2}$$

ignores the off-diagonal covariance, but the data points **are** correlated
(measured on the same configurations). Consequently:

$$\mathbb{E}[\tilde\chi^2] \neq N_{\mathrm{dof}},
\qquad \tilde\chi^2 \not\sim \chi^2_{N_{\mathrm{dof}}}.$$

Therefore, for uncorrelated fits:

$$\boxed{\ \frac{\tilde\chi^2}{N_{\mathrm{dof}}}\approx 1\ \text{is NOT a valid GoF criterion}\ }$$

$$\boxed{\ p = Q\!\left(\tfrac{N_{\mathrm{dof}}}{2},\tfrac{\tilde\chi^2}{2}\right)\ \text{is NOT a valid p-value}\ }$$

The parameter estimates $\hat{\boldsymbol\theta}$ remain **unbiased**, and the
resample errors (Section 5) remain **valid**, because they do not rely on
interpreting $\tilde\chi^2$ as χ²-distributed. Only the *goodness-of-fit
interpretation* of $\tilde\chi^2$ breaks.

## Information Criteria (both, with caution)

$$\text{AIC} = \chi^2_{\min} + 2(N_{\mathrm{par}}-N_{\mathrm{dat}})$$

$$\text{AICc} = \text{AIC}
+ \frac{2N_{\mathrm{par}}(N_{\mathrm{par}}+1)}{N_{\mathrm{dat}}-N_{\mathrm{par}}-1}$$

AIC differences are meaningful for **model comparison at fixed correlation
treatment**; comparing AIC between a correlated and an uncorrelated fit is not
meaningful because the underlying $\chi^2$ statistics differ.

---

# Stability Diagnostics Across Resamples

Each resample fit yields a full set of statistics. Their **distributions**
diagnose stability — but *which* distribution to trust depends on the
correlation type.

## Parameter Distribution (valid for BOTH)

The most robust, correlation-agnostic diagnostic is the spread of the
parameters themselves. For each parameter, the resampled values

$$\{\hat\theta_a^{(k)}\}_{k=1}^{N_{\mathrm{res}}}$$

should, for a healthy fit, be approximately Gaussian about the central value:

$$\hat\theta_a^{(k)} \sim \mathcal{N}\!\big(\hat\theta_a^{\text{cen}},\,\sigma_a^2\big)$$

Diagnostics to compute per parameter:

| Quantity | Formula | Healthy value |
|---|---|---|
| mean | $\overline{\hat\theta}_a$ | $\approx \hat\theta_a^{\text{cen}}$ |
| std | $\sigma_a^{\text{resample}}$ | matches `param_err` |
| **bias** | $\overline{\hat\theta}_a - \hat\theta_a^{\text{cen}}$ | $\approx 0$ |
| outliers | $\#\{k:\,|\hat\theta_a^{(k)}-\overline{\hat\theta}_a|>3\sigma_a\}$ | few / none |

**Warning signs:**

- **Bimodal** → resamples landing in different minima (e.g. unbroken
  permutation symmetry $(A_0,E_0)\leftrightarrow(A_1,E_1)$). Fix with ordering
  limits $E_0 < E_1$.
- **Skewed** → parameter near a boundary or poorly constrained (often where
  $\sigma_a^{\text{resample}} \gg \sigma_a^{\text{hess}}$).
- **Large bias** → model nonlinearity; central value and resample mean disagree.

## χ²/dof Spread (correlated ONLY)

For correlated fits, the spread of the per-resample reduced χ² is a valid
secondary stability check:

$$\mathrm{std}_k\!\left(\frac{\chi^2_{(k)}}{N_{\mathrm{dof}}}\right)$$

- **Small spread** → consistent fit quality across resamples → stable.
- **Large spread** → fragile fit; small data fluctuations change the fit regime.

For **uncorrelated** fits this is **not** a valid diagnostic (the per-resample
$\tilde\chi^2$ is not a true χ²); rely on the parameter distribution instead.

## Summary Table — What to Trust

| Diagnostic | Correlated | Uncorrelated |
|---|---|---|
| $\chi^2/N_{\mathrm{dof}}\approx 1$ | valid | invalid |
| $p$-value | valid | invalid |
| χ²/dof spread across resamples | valid | invalid |
| parameter distribution $\{\hat\theta_a^{(k)}\}$ | valid | valid |
| resample errors $\sigma_a^{\text{resample}}$ | valid | valid |
| HESSE vs resample agreement | valid | valid |

---

# The Pipeline (ordered)

1. **Average** ensemble → central data $\bar y_i$.
2. **Build cost** from frozen $W$: $\chi^2=\sum_{ij}r_iW_{ij}r_j$ (+priors),
   with `errordef=1`, `ndata` set.
3. **Central fit**: simplex → migrad → hesse → $\hat{\boldsymbol\theta}^{\text{cen}}$,
   $V^{\text{hess}}$.
4. **Extract** `FitResult` (params, hess errors, χ², residuals).
5. **Resample loop**: refit each $y^{(k)}$, warm-started, same frozen $W$ →
   $\{\hat{\boldsymbol\theta}^{(k)}\}$.
6. **Quoted error**: jackknife/bootstrap spread $\sigma_a^{\text{resample}}$.
7. **Goodness of fit**: $\chi^2/N_{\mathrm{dof}}$, $p$ — **valid only if correlated**.
8. **Stability**: parameter distribution (always); χ²/dof spread (correlated only).

---

# The Core Design Equation

$$\theta_a
= \underbrace{\hat\theta_a^{\text{cen}}}_{\substack{\text{central fit}\\\text{(one fit on }\bar y\text{)}}}
\;\pm\;
\underbrace{\sqrt{\tfrac{N_{\mathrm{res}}-1}{N_{\mathrm{res}}}
\sum_k\big(\hat\theta_a^{(k)}-\overline{\hat\theta}_a\big)^2}}_{\substack{\text{resample spread}\\\text{(}N_{\mathrm{res}}\text{ refits)}}}$$

with $V^{\text{hess}}=H^{-1}$ retained as a cross-check: when
$\sigma_a^{\text{hess}}\approx\sigma_a^{\text{resample}}$ the likelihood is
approximately Gaussian. The value comes from the central fit; the error comes
from the resample distribution — and the **goodness-of-fit interpretation is
only valid when the full covariance is used** (correlated fits).

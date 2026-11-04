"""Power of the two test tracks for a number of seeds (advice only).

The functions answer one question: if a phantom configuration truly lowers the main metric by
delta, how likely is the analysis to detect the drop with N seeds? They compute and decide
nothing. The seeds of a run plan come only from the training section of the run config, and no
other part of the analysis reads the results of this module.

Track A is the paired t-test of each k level against the ground truth (two-sided). Its power has a
closed form with the noncentral t distribution. For some large noncentrality parameters the scipy
functions of this distribution give no value; the same probability then comes from a numerical
integration over the chi-square distribution of the sample variance.

Track B is the repeated-measures ANOVA with the Greenhouse-Geisser correction (the omnibus test)
and the Tukey test of the ground truth against one k level with the pooled error. Its power comes
from a Monte Carlo simulation: data of N seeds are drawn from a multivariate normal distribution
with the covariance of the configurations, the drop is put on one k level, and the share of the
simulated data sets in which each test rejects is the power. The same random draws serve every N
(common random numbers), so the power curve over N is smooth and the result can be repeated with
the same random seed. A Tukey p-value below alpha is the same as a studentized range statistic
above its critical value, so the critical value is computed once for each N.

The variance of the data is uncertain when it comes from a few seeds. sd_upper() and
variance_factor() give the upper confidence limit of a standard deviation, which makes the advice
cautious: the true spread is probably not larger than this value.

Only numpy and scipy are used.
"""

from functools import lru_cache

import numpy as np
from scipy import integrate, special, stats

DEFAULT_TARGET = 0.8
DEFAULT_CONFIDENCE = 0.8
MIN_SEEDS = 3


def _check_alpha(alpha):
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be between 0 and 1, got {alpha}")
    return alpha


def _check_fraction(value, name):
    value = float(value)
    if not 0.0 < value < 1.0:
        raise ValueError(f"{name} must be between 0 and 1, got {value}")
    return value


def variance_factor(n, confidence=DEFAULT_CONFIDENCE):
    """Factor that turns a sample variance of n values into its upper confidence limit.

    The limit is s^2 * (n - 1) / chi2(1 - confidence; n - 1), where chi2(q; df) is the q quantile
    of the chi-square distribution. With confidence 0.8, the true variance is below the limit with
    80 % confidence (normal data).
    """
    confidence = _check_fraction(confidence, "confidence")
    n = int(n)
    if n < 2:
        raise ValueError(f"an upper limit of the SD needs at least 2 values, got {n}")
    return (n - 1) / stats.chi2.ppf(1.0 - confidence, n - 1)


def sd_upper(s, n, confidence=DEFAULT_CONFIDENCE):
    """Upper confidence limit of a standard deviation s (a number or an array) estimated from n values."""
    limit = np.asarray(s, dtype=float) * np.sqrt(variance_factor(n, confidence))
    return float(limit) if limit.ndim == 0 else limit


def track_a_power(delta, sigma, n, alpha):
    """Power of the two-sided paired t-test with n pairs.

    delta: the true mean difference; sigma: the SD of the paired differences. Returns
    (power, note). The note is empty when the power has a value.
    """
    alpha = _check_alpha(alpha)
    n = int(n)
    if n < 2:
        raise ValueError(f"the paired t-test needs at least 2 pairs, got {n}")
    sigma = float(sigma)
    if not np.isfinite(sigma) or sigma <= 0.0:
        return np.nan, "the SD of the differences is 0 or missing: no power value"
    df = n - 1
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    ncp = abs(float(delta)) * np.sqrt(n) / sigma
    with np.errstate(all="ignore"):
        power = float(stats.nct.sf(t_crit, df, ncp) + stats.nct.cdf(-t_crit, df, ncp))
    if not np.isfinite(power):
        power = power_by_integration(t_crit, df, ncp)
    if not np.isfinite(power):
        return np.nan, "the power could not be computed (numerical fault)"
    return min(1.0, max(0.0, power)), ""


def power_by_integration(t_crit, df, ncp):
    """
    P(|T| > t_crit) for a noncentral t variable T with df degrees of freedom and
    noncentrality ncp, by numerical integration.

    T = (Z + ncp) / sqrt(V / df), with Z standard normal and V chi-square with df
    degrees of freedom. For a fixed V the probability is a normal probability, so
    the result is the integral of that probability over the density of V. The
    integral runs between the 1e-16 and 1 - 1e-16 quantiles of V.
    """
    half = df / 2.0
    log_norm = half * np.log(2.0) + special.gammaln(half)

    def integrand(v):
        scale = t_crit * np.sqrt(v / df)
        density = np.exp(special.xlogy(half - 1.0, v) - v / 2.0 - log_norm)
        return (special.ndtr(ncp - scale) + special.ndtr(-scale - ncp)) * density

    lower, upper = float(stats.chi2.ppf(1e-16, df)), float(stats.chi2.isf(1e-16, df))
    value, _ = integrate.quad(integrand, lower, upper, points=[float(df)], epsabs=1e-14, epsrel=1e-12,
                              limit=200)
    return float(value)


def track_a_n(delta, sigma, alpha, target=DEFAULT_TARGET, n_min=MIN_SEEDS, n_max=100):
    """Smallest number of pairs n_min <= n <= n_max with Track A power >= target.

    Returns (n, power at n, note). n is None when the target is not reached by n_max or when the
    power has no value; the note gives the reason.
    """
    target = _check_fraction(target, "target power")
    n_min, n_max = int(n_min), int(n_max)
    if n_min < 2 or n_max < n_min:
        raise ValueError(f"need 2 <= n_min <= n_max, got {n_min} and {n_max}")
    power = np.nan
    for n in range(n_min, n_max + 1):
        power, note = track_a_power(delta, sigma, n, alpha)
        if note:
            return None, np.nan, note
        if power >= target:
            return n, power, ""
    return None, power, f"the target power {target} is not reached with {n_max} seeds"


def contrast_basis(m):
    """Orthonormal contrasts (m x (m - 1)): normalised Helmert contrasts."""
    m = int(m)
    if m < 2:
        raise ValueError(f"need at least 2 configurations, got {m}")
    basis = np.zeros((m, m - 1))
    for j in range(1, m):
        basis[:j, j - 1] = -1.0
        basis[j, j - 1] = float(j)
        basis[:, j - 1] /= np.sqrt(j * (j + 1))
    return basis


def rm_batch(y, reference=0):
    """Repeated-measures statistics for many data sets at once.

    y: array (data sets, seeds, configurations), or (seeds, configurations) for one data set.
    reference: the column of the ground truth.

    Returns a dict of arrays with one value per data set: f_stat, df1, df2, gg_epsilon, p_gg
    (Greenhouse-Geisser), mse (the pooled error of the seed-by-configuration stratum) and q_ref
    (data sets x configurations: the studentized range statistic |mean_j - mean_ref| / sqrt(mse / n)
    of the reference against each column; 0 for the reference itself). A data set without residual
    variance gets NaN values.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 2:
        y = y[None, :, :]
    if y.ndim != 3:
        raise ValueError(f"y must have 2 or 3 dimensions, got {y.ndim}")
    _, n, m = y.shape
    if n < 2 or m < 2:
        raise ValueError(f"need at least 2 seeds and 2 configurations, got {n} and {m}")
    grand = y.mean(axis=(1, 2))
    col_means = y.mean(axis=1)
    row_means = y.mean(axis=2)
    ss_cond = n * np.sum((col_means - grand[:, None]) ** 2, axis=1)
    ss_subj = m * np.sum((row_means - grand[:, None]) ** 2, axis=1)
    ss_total = np.sum((y - grand[:, None, None]) ** 2, axis=(1, 2))
    ss_err = ss_total - ss_cond - ss_subj
    df1, df2 = m - 1, (n - 1) * (m - 1)
    scale = np.maximum(ss_total, 1e-300)
    no_residual = ss_err <= 1e-12 * scale
    with np.errstate(divide="ignore", invalid="ignore"):
        mse = np.where(no_residual, np.nan, ss_err / df2)
        f_stat = (ss_cond / df1) / mse
        z = y @ contrast_basis(m)
        z = z - z.mean(axis=1, keepdims=True)
        s = np.einsum("rnp,rnq->rpq", z, z) / (n - 1)
        trace_s = np.trace(s, axis1=1, axis2=2)
        trace_s2 = np.sum(s * s, axis=(1, 2))
        gg = np.where(no_residual, np.nan, trace_s ** 2 / (df1 * trace_s2))
        p_gg = stats.f.sf(f_stat, gg * df1, gg * df2)
        q_ref = np.abs(col_means - col_means[:, [reference]]) / np.sqrt(mse / n)[:, None]
    return {"f_stat": f_stat, "df1": df1, "df2": df2, "gg_epsilon": gg, "p_gg": p_gg, "mse": mse,
            "q_ref": q_ref}


def tukey_critical(alpha, m, df):
    """Critical value of the studentized range for m means and df error degrees of freedom."""
    return _tukey_critical(_check_alpha(alpha), int(m), int(df))


@lru_cache(maxsize=None)
def _tukey_critical(alpha, m, df):
    # the quantile needs a numerical integration; a run uses each value many times
    return float(stats.studentized_range.ppf(1.0 - alpha, m, df))


def covariance_root(cov):
    """Symmetric square root of a covariance matrix (eigenvalues below 0 from round-off set to 0).

    A covariance from n seeds has rank n - 1 at most, so a Cholesky factor may not exist; the
    symmetric root exists for every positive semi-definite matrix.
    """
    cov = np.asarray(cov, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError(f"the covariance must be a square matrix, got shape {cov.shape}")
    if not np.all(np.isfinite(cov)):
        raise ValueError("the covariance has a missing or infinite value")
    if not np.allclose(cov, cov.T, rtol=1e-10, atol=1e-14):
        raise ValueError("the covariance is not symmetric")
    values, vectors = np.linalg.eigh(cov)
    tolerance = 1e-10 * max(values.max(), 0.0)
    if values.min() < -max(tolerance, 1e-15):
        raise ValueError(f"the covariance is not positive semi-definite (eigenvalue {values.min():.3g})")
    values = np.clip(values, 0.0, None)
    return (vectors * np.sqrt(values)) @ vectors.T


def track_b_power_curve(cov, delta, level, n_grid, alphas, replicates=2000, random_seed=0, reference=0):
    """Monte Carlo power of Track B over a grid of seed numbers.

    cov: covariance of the configurations (m x m); delta: the true drop, put on column `level`
    (the reference column keeps mean 0); n_grid: seed numbers (each at least 3); alphas: one or
    more significance levels; replicates: simulated data sets for each seed number.

    Returns a list of dicts, one per (n, alpha): n, alpha, power_omnibus (share with p_gg < alpha),
    power_pair (share with the Tukey test of the reference against `level` significant),
    mc_se_omnibus, mc_se_pair (Monte Carlo standard errors), n_no_residual (data sets without
    residual variance; they count as not significant).
    """
    root = covariance_root(cov)
    m = root.shape[0]
    level, reference = int(level), int(reference)
    if not (0 <= level < m and 0 <= reference < m and level != reference):
        raise ValueError(f"level {level} and reference {reference} must be different columns of 0..{m - 1}")
    n_grid = sorted({int(n) for n in n_grid})
    if not n_grid or n_grid[0] < MIN_SEEDS:
        raise ValueError(f"every seed number of the grid must be at least {MIN_SEEDS}")
    alphas = [_check_alpha(a) for a in np.atleast_1d(alphas)]
    replicates = int(replicates)
    if replicates < 1:
        raise ValueError(f"replicates must be at least 1, got {replicates}")
    mean = np.zeros(m)
    mean[level] = -float(delta)
    draws = np.random.default_rng(random_seed).standard_normal((replicates, n_grid[-1], m))
    rows = []
    for n in n_grid:
        y = draws[:, :n, :] @ root.T + mean
        result = rm_batch(y, reference=reference)
        no_residual = int(np.sum(np.isnan(result["mse"])))
        q = result["q_ref"][:, level]
        for alpha in alphas:
            omnibus = float(np.mean(np.nan_to_num(result["p_gg"], nan=1.0) < alpha))
            pair = float(np.mean(np.nan_to_num(q, nan=0.0) > tukey_critical(alpha, m, result["df2"])))
            rows.append({"n": n, "alpha": alpha, "power_omnibus": omnibus, "power_pair": pair,
                         "mc_se_omnibus": float(np.sqrt(omnibus * (1 - omnibus) / replicates)),
                         "mc_se_pair": float(np.sqrt(pair * (1 - pair) / replicates)),
                         "n_no_residual": no_residual})
    return rows


def smallest_n(rows, column, alpha, target=DEFAULT_TARGET):
    """First n of a power curve (rows of track_b_power_curve) with `column` >= target at alpha.

    Returns (n, note); n is None and the note gives the reason when the target is not reached.
    """
    target = _check_fraction(target, "target power")
    alpha = _check_alpha(alpha)
    curve = sorted((row["n"], row[column]) for row in rows if np.isclose(row["alpha"], alpha))
    if not curve:
        raise ValueError(f"the curve has no rows for alpha {alpha}")
    for n, power in curve:
        if power >= target:
            return n, ""
    return None, f"the target power {target} is not reached with {curve[-1][0]} seeds"

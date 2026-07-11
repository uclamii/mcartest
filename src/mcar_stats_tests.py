"""Statistical hypothesis test for Missing Completely At Random (MCAR)

This module is derived from pyampute (https://github.com/RianneSchouten/pyampute),
BSD 3-Clause License.

Copyright (c) 2018, the respective contributors, as shown by the AUTHORS file.
All rights reserved.

Original authors: Rianne Schouten, Davina Zamanzadeh.
Modified by Leon Shpaner to add effect-size reporting.

See LICENSES/pyampute-LICENSE for the full license text.
"""

# Author: Rianne Schouten <https://rianneschouten.github.io/>
# Co-Author: Davina Zamanzadeh <https://davinaz.me/>
# Modified to report effect sizes alongside p-values.

from logging import error
from typing import Union
import numpy as np
import pandas as pd
from math import pow
from scipy.stats import chi2, ttest_ind

# Local
Matrix = Union[pd.DataFrame, np.ndarray]


def _cohens_d(a: pd.Series, b: pd.Series) -> float:
    """
    Pooled-variance Cohen's d for two samples.

    Returns np.nan when either group has fewer than 2 usable values or the
    pooled standard deviation is 0.
    """
    a = a.dropna()
    b = b.dropna()
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return np.nan
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    if pooled == 0:
        return np.nan
    return (a.mean() - b.mean()) / pooled


def _effect_label(
    d, thresholds=(0.2, 0.5, 0.8), labels=("negligible", "small", "medium", "large")
):
    """Map a Cohen's d to its magnitude band. NaN passes through."""
    if pd.isna(d):
        return np.nan
    a = abs(d)
    small_t, medium_t, large_t = thresholds
    if a >= large_t:
        return labels[3]
    elif a >= medium_t:
        return labels[2]
    elif a >= small_t:
        return labels[1]
    return labels[0]


def _elementwise(df, func):
    """Apply ``func`` elementwise to a DataFrame, across pandas versions.

    ``DataFrame.map`` was introduced in pandas 2.1.0; earlier versions expose
    the same behavior as ``DataFrame.applymap``. This tries ``.map`` first and
    falls back to ``.applymap`` when it is unavailable, so the library works on
    both older (Python 3.8 era) and current pandas.
    """
    try:
        return df.map(func)  # pandas >= 2.1
    except AttributeError:
        return df.applymap(func)  # pandas < 2.1


class MCARTest:
    """
    Statistical hypothesis test for Missing Completely At Random (MCAR)

    Performs Little's MCAR test (see `Little, R.J.A. (1988)`_). Null hypothesis: data is Missing Completely At Random (MCAR). Alternative hypothesis: data is not MCAR.

    .. _`Little, R.J.A. (1988)`: https://www.tandfonline.com/doi/abs/10.1080/01621459.1988.10478722

    Parameters
    ----------
    method : str, {"little", "ttest"}, default : "little"
        Whether to perform a chi-square test on the entire dataset ("little") or separate t-tests for every combination of variables ("ttest").

    See also
    --------
    :class:`~pyampute.exploration.md_patterns.mdPatterns` : Displays missing data patterns in incomplete datasets

    :class:`~pyampute.ampute.MultivariateAmputation` : Transformer for generating multivariate missingness in complete datasets

    Notes
    -----
    We advise to use Little's MCAR test carefully. Rejecting the null hypothesis may not always mean that data is not MCAR, nor is accepting the null hypothesis a guarantee that data is MCAR. See `Schouten et al. (2021)`_ for a thorough discussion of missingness mechanisms.

    .. _`Schouten et al. (2021)`: https://journals.sagepub.com/doi/full/10.1177/0049124118799376

    Examples
    --------
    >>> import pandas as pd
    >>> from pyampute.exploration.mcar_statistical_tests import MCARTest
    >>> data_mcar = pd.read_table("data/missingdata_mcar.csv")
    >>> mt = MCARTest(method="little")
    >>> print(mt.little_mcar_test(data_mcar))
    0.17365464213775494
    """

    def __init__(self, method: str = "little"):
        self.method = method

    def __call__(self, data: Matrix) -> float:
        if self.method == "little":
            return self.little_mcar_test(data)
        elif self.method == "ttest":
            return self.mcar_t_tests(data)
        else:
            error(
                f"Chose {self.method} as test method, which is not supported. Please choose from [little, ttest]."
            )

    @staticmethod
    def little_mcar_test(X: Matrix, return_stats: bool = False):
        """
        Implementation of Little's MCAR test

        Parameters
        ----------
        X : Matrix of shape `(n, m)`
            Dataset with missing values. `n` rows (samples) and `m` columns (features).

        return_stats : bool, default : False
            If False, returns only the p-value (original behavior). If True,
            returns a dict with the p-value, the chi-square statistic, the
            degrees of freedom, and an effect size.

        Returns
        -------
        pvalue : float
            The p-value of a chi-square hypothesis test when `return_stats` is
            False. Null hypothesis: data is Missing Completely At Random (MCAR).
            Alternative hypothesis: data is not MCAR.

        stats : dict
            Returned when `return_stats` is True, with keys:

            - ``"pvalue"`` : chi-square p-value
            - ``"statistic"`` : Little's chi-square statistic ``d2``
            - ``"df"`` : degrees of freedom
            - ``"effect_size"`` : ``sqrt(d2 / n)``, a Cohen's w style measure.
              Because ``d2`` is a sample-size-weighted sum of Mahalanobis
              distances of the pattern means from the grand mean, this equals
              the root of the average squared Mahalanobis distance and is
              interpretable as a standardized effect magnitude.
        """

        dataset = X.copy()
        vars = dataset.dtypes.index.values
        n_var = dataset.shape[1]
        n_obs = dataset.shape[0]

        # mean and covariance estimates
        # ideally, this is done with a maximum likelihood estimator
        gmean = dataset.mean()
        gcov = dataset.cov()

        # set up missing data patterns
        r = 1 * dataset.isnull()
        mdp = np.dot(r, list(map(lambda x: pow(2, x), range(n_var))))
        sorted_mdp = sorted(np.unique(mdp))
        n_pat = len(sorted_mdp)
        correct_mdp = list(map(lambda x: sorted_mdp.index(x), mdp))
        dataset["mdp"] = pd.Series(correct_mdp, index=dataset.index)

        # calculate statistic and df
        pj = 0
        d2 = 0
        for i in range(n_pat):
            dataset_temp = dataset.loc[dataset["mdp"] == i, vars]
            select_vars = ~dataset_temp.isnull().any()
            pj += np.sum(select_vars)
            select_vars = vars[select_vars]
            means = dataset_temp[select_vars].mean() - gmean[select_vars]
            select_cov = gcov.loc[select_vars, select_vars]
            mj = len(dataset_temp)
            parta = np.dot(
                means.T, np.linalg.solve(select_cov, np.identity(select_cov.shape[1]))
            )
            d2 += mj * (np.dot(parta, means))

        df = pj - n_var

        # perform test and save output
        pvalue = 1 - chi2.cdf(d2, df)

        if return_stats:
            effect_size = np.sqrt(d2 / n_obs) if n_obs > 0 else np.nan
            return {
                "pvalue": pvalue,
                "statistic": float(d2),
                "df": int(df),
                "effect_size": float(effect_size),
            }

        return pvalue

    @staticmethod
    def mcar_t_tests(
        X,
        effect_size=False,
        size_label=False,
        label_mcar=False,
        label_not_mcar=False,
        effect_if_not_mcar=False,
        alpha=0.05,
    ):
        """
        Performs t-tests for MCAR for each pair of features.

        Parameters
        ----------
        X : Matrix of shape `(n, m)`
            Dataset with missing values. `n` rows (samples) and `m` columns
            (features). Non-numeric columns are ignored.

        effect_size : bool, default : False
            If False, returns only the p-value matrix. If True, returns a tuple
            ``(pvalues, effect_matrix)``.

        size_label : bool, default : False
            If True, the effect matrix reports magnitude bands
            ("negligible" / "small" / "medium" / "large") instead of numeric
            absolute Cohen's d.

        label_mcar : bool, default : False
            If True, the p-value matrix is replaced with "MCAR" where the null is
            not rejected (p > alpha) and "" elsewhere.

        label_not_mcar : bool, default : False
            If True, the p-value matrix is replaced with "not MCAR" where the null
            is rejected (p <= alpha) and "" elsewhere. Complementary to
            `label_mcar`. If both are set, `label_mcar` takes precedence.

        effect_if_not_mcar : bool, default : False
            If True, the effect matrix reports magnitude labels only where MCAR is
            rejected (p <= alpha) and "" elsewhere. Under MCAR there is no
            association to quantify, so effect sizes are only meaningful where the
            test rejects. Implies label output for the effect matrix.

        alpha : float, default : 0.05
            Significance threshold used by `label_mcar`, `label_not_mcar`, and
            `effect_if_not_mcar`.

        Returns
        -------
        pvalues : pandas DataFrame of shape `(m, m)`
            The p-values of t-tests for each pair of features (or label strings
            when `label_mcar` / `label_not_mcar` is True). Null hypothesis for cell
            :math:`pvalues[h,j]`: data in feature :math:`h` is Missing Completely
            At Random (MCAR) with respect to feature :math:`j`. Diagonal values do
            not exist.

        effect_matrix : pandas DataFrame of shape `(m, m)`
            Returned when `effect_size` is True. Absolute Cohen's d for each pair
            (magnitude only), or magnitude labels when `size_label` or
            `effect_if_not_mcar` is set.
        """
        dataset = X.copy()
        dataset = dataset.select_dtypes(include="number")
        vars = dataset.dtypes.index.values
        mcar_matrix = pd.DataFrame(
            data=np.zeros(shape=(dataset.shape[1], dataset.shape[1])),
            columns=vars,
            index=vars,
        )
        effect_matrix = pd.DataFrame(
            data=np.full(shape=(dataset.shape[1], dataset.shape[1]), fill_value=np.nan),
            columns=vars,
            index=vars,
        )

        for var in vars:
            for tvar in vars:
                part_one = dataset.loc[dataset[var].isnull(), tvar].dropna()
                part_two = dataset.loc[~dataset[var].isnull(), tvar].dropna()
                mcar_matrix.loc[var, tvar] = ttest_ind(
                    part_one, part_two, equal_var=False
                ).pvalue
                effect_matrix.loc[var, tvar] = _cohens_d(part_one, part_two)

        pvalues = mcar_matrix[mcar_matrix.notnull()]

        # effect matrix as magnitude labels if requested (or implied by gating)
        if size_label or effect_if_not_mcar:
            effect_matrix = _elementwise(effect_matrix, _effect_label)

        # gate effect labels to NON-MCAR cells only (p <= alpha); blank elsewhere
        if effect_if_not_mcar:
            not_mcar = pvalues <= alpha
            effect_matrix = effect_matrix.where(not_mcar, "")

        # p-value matrix as MCAR / not-MCAR labels if requested
        if label_mcar:
            pvalues = _elementwise(
                pvalues, lambda p: "MCAR" if pd.notna(p) and p > alpha else ""
            )
        elif label_not_mcar:
            pvalues = _elementwise(
                pvalues, lambda p: "not MCAR" if pd.notna(p) and p <= alpha else ""
            )

        if effect_size:
            return pvalues, effect_matrix
        return pvalues

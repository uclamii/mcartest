<picture><img src="https://github.com/uclamii/mcartest/blob/main/assets/mcartest_logo.svg" width="350" alt="My Logo"/></picture>

MCAR statistical tests in Python: Little's chi-square test and pairwise t-tests, extended with Cohen's d effect sizes and MCAR/not-MCAR labeling

## Prerequisites

Before installing `mcartest`, ensure your system meets the following requirements:

## Python Version

`mcartest` requires **Python 3.8 or higher**. Specific dependency versions vary depending on your Python version.

## Dependencies

The following dependencies will be automatically installed with `mcartest`:

- `numpy`
- `pandas`
- `scipy`

## 💾 Installation

You can install `mcartest` directly from PyPI:

```bash
pip install mcartest
```

## 📄 Official Documentation

[Documentation](https://uclamii.github.io/mcartest/)

## 🌐 Author Website

https://www.mii.ucla.edu/

## ⚖️ License

`mcartest` is distributed under the Apache License. See [LICENSE](https://github.com/uclamii/mcartest?tab=Apache-2.0-1-ov-file) for more information.

This library includes code derived from [pyampute](https://github.com/RianneSchouten/pyampute), used under the BSD 3-Clause License. See [NOTICE](https://github.com/uclamii/mcartest/blob/main/NOTICE) and [THIRD_PARTY_LICENSES.md](https://github.com/uclamii/mcartest/blob/main/THIRD_PARTY_LICENSES.md) for attribution and the full third-party license text.

## 📚 Citing `mcartest`

If you use `mcartest` in your research or projects, please consider citing it.

<!-- ```bibtex
@software{shpaner_mcartest,
   author       = {Shpaner, Leonid},
   title        = {mcartest},
   year         = 2026,
   publisher    = {Zenodo},
   version      = {0.0.1},
   doi          = {[YOUR_DOI]},
   url          = {[YOUR_ZENODO_URL]}
}
``` -->

## Support

If you have any questions or issues with `mcartest`, please open an issue on this [GitHub repository](https://github.com/uclamii/mcartest/).

## Acknowledgements

This work builds on [pyampute](https://github.com/RianneSchouten/pyampute) by Rianne Schouten and Davina Zamanzadeh, whose implementation of Little's MCAR test and the pairwise t-test approach forms the statistical core of this library. The effect-size and labeling extensions were added on top of their work and are also being contributed back upstream.

---

# Sample Use Cases

`mcartest` provides two complementary tests for assessing whether missing data is Missing Completely At Random (MCAR): Little's chi-square test across the whole dataset, and pairwise t-tests between every combination of features. Both are extended here to report Cohen's d effect sizes and human-readable MCAR labels alongside the raw statistics.

## Little's MCAR Test

Little's test evaluates the whole dataset at once. The null hypothesis is that the data is MCAR; a small p-value is evidence against MCAR.

**Step 1. Import library**

```python
import pandas as pd
from mcartest import MCARTest
```

**Step 2. Run the test**

```python
mt = MCARTest(method="little")
pvalue = mt.little_mcar_test(df)
print(pvalue)
```

**Step 3. Return the full statistics with an effect size**

Passing `return_stats=True` returns the p-value along with the chi-square statistic, degrees of freedom, and a Cohen's w style effect size. Because Little's statistic is a sample-size-weighted sum of Mahalanobis distances of the pattern means from the grand mean, `sqrt(d2 / n)` is interpretable as a standardized effect magnitude.

```python
stats = mt.little_mcar_test(df, return_stats=True)
print(stats)
# {'pvalue': ..., 'statistic': ..., 'df': ..., 'effect_size': ...}
```

## Pairwise t-tests

The t-test approach compares the distribution of each feature between the rows where another feature is missing versus present, for every pair of features. The result is an `m x m` matrix.

**Step 1. Run the test**

```python
mt = MCARTest(method="ttest")
pvalues = mt.mcar_t_tests(df)
```

Each cell `[h, j]` is the p-value testing whether feature `h` is MCAR with respect to feature `j`. Non-numeric columns are ignored automatically.

**Step 2. Add effect sizes**

Set `effect_size=True` to also return a matrix of absolute Cohen's d for each feature pair. Magnitude tells you how strong the association is, independent of the p-value's sample-size sensitivity.

```python
pvalues, effects = mt.mcar_t_tests(df, effect_size=True)
```

**Step 3. Report effect sizes as magnitude bands**

`size_label=True` replaces the numeric Cohen's d with `"negligible"`, `"small"`, `"medium"`, or `"large"` using the conventional 0.2 / 0.5 / 0.8 cutoffs.

```python
pvalues, effect_labels = mt.mcar_t_tests(df, effect_size=True, size_label=True)
```

## MCAR and not-MCAR labeling

For a quick, readable summary you can replace the p-value matrix with labels.

`label_mcar=True` marks the cells consistent with MCAR (p > alpha):

```python
labels = mt.mcar_t_tests(df, label_mcar=True)
```

`label_not_mcar=True` marks the cells that reject MCAR (p <= alpha), which is the complement:

```python
labels = mt.mcar_t_tests(df, label_not_mcar=True)
```

## Effect sizes only where MCAR is rejected

Under MCAR there is no association to quantify, so effect size is only meaningful where the test rejects. `effect_if_not_mcar=True` reports magnitude labels only in the cells that reject MCAR (p <= alpha) and leaves the rest blank.

```python
pvalues, effects = mt.mcar_t_tests(df, effect_size=True, effect_if_not_mcar=True)
```

## Styling and export

The matrices are plain pandas DataFrames, so they drop straight into pandas styling and Excel export. A common pattern is to build a filter mask once off the numeric p-value matrix and apply it to every derived matrix so they all share the same rows and columns.

```python
pvalues, effects = mt.mcar_t_tests(df, effect_size=True, effect_if_not_mcar=True)

mask = pvalues.notna().any(axis=1)
pvalues = pvalues[mask].T
effects = effects[mask].T

def highlight_effect(val):
    colors = {
        "large": "background-color: #27ae60; color: white",
        "medium": "background-color: #2ecc71",
        "small": "background-color: #a9dfbf",
    }
    return colors.get(val, "")

effects.style.map(highlight_effect).to_excel("effects.xlsx")
```

## A note on interpreting results

We advise using MCAR tests carefully. Rejecting the null does not always mean the data is not MCAR, nor is failing to reject a guarantee that it is. See [Schouten et al. (2021)](https://journals.sagepub.com/doi/full/10.1177/0049124118799376) for a thorough discussion of missingness mechanisms. Effect sizes are provided to complement the p-values: a significant result driven by a large sample may still reflect a trivial difference, and a large effect in a sparse missing-group may not reach significance.

## References

1. Little, R. J. A. (1988). A Test of Missing Completely at Random for Multivariate Data with Missing Values. *Journal of the American Statistical Association*, 83(404), 1198-1202. https://doi.org/10.1080/01621459.1988.10478722

2. Schouten, R. M., Lugtig, P., & Vink, G. (2018). Generating missing values for simulation purposes: A multivariate amputation procedure. *Journal of Statistical Computation and Simulation*, 88(15), 2909-2930. https://doi.org/10.1080/00949655.2018.1491577

3. Schouten, R. M., Zamanzadeh, D., & Singh, P. (2022). *pyampute: a Python library for data amputation*. Zenodo. https://doi.org/10.25080/majora-212e5952-03e
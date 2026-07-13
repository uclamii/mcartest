<picture><img src="https://raw.githubusercontent.com/uclamii/mcartest/main/assets/mcartest_logo.svg" width="350" alt="My Logo"/></picture>

<br>

[![Downloads](https://pepy.tech/badge/mcartest)](https://pepy.tech/project/mcartest) [![PyPI](https://img.shields.io/pypi/v/mcartest.svg)](https://pypi.org/project/mcartest/) [![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) 

`mcartest` provides statistical tests for assessing whether missing data is Missing Completely At Random (MCAR), the assumption most imputation methods quietly depend on. It implements Little's chi-square test across the full dataset and pairwise *t*-tests between every combination of features, returning results as a readable matrix rather than a single opaque *p*-value.

Where most implementations stop at significance, `mcartest` also reports **Cohen's *d* effect sizes**, because a *p*-value alone is misleading: with a large sample, a trivial difference becomes "significant," while a real association in a sparse missing-group may never reach the threshold. Effect sizes tell you *how strongly* missingness is tied to a feature, not just whether the test rejected.

Results can be returned as raw *p*-values, as human-readable **MCAR / not-MCAR labels**, or as **effect-size magnitude bands** (negligible, small, medium, large) reported only where MCAR is rejected, since under MCAR there is no association to quantify. Styling helpers render the matrices with color-coded cells in notebooks and export them straight to Excel.

Built on [pyampute](https://github.com/RianneSchouten/pyampute) (BSD 3-Clause), extended with the effect-size and labeling functionality described above.

## Prerequisites

Before installing `mcartest`, ensure your system meets the following requirements:

## Python Version

`mcartest` requires **Python 3.8 or higher**. Specific dependency versions vary depending on your Python version.

## Dependencies

The following dependencies will be automatically installed with `mcartest`:

- `jinja2>=3.0.0`
- `numpy>=1.19.0`
- `openpyxl>=3.0.0`
- `pandas>=1.3.0`
- `scipy>=1.5.0`

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

## Support

If you have any questions or issues with `mcartest`, please open an issue on this [GitHub repository](https://github.com/uclamii/mcartest/).

## Acknowledgements

This work builds on [pyampute](https://github.com/RianneSchouten/pyampute) by Rianne Schouten and Davina Zamanzadeh, whose implementation of Little's MCAR test and the pairwise t-test approach forms the statistical core of this library. The effect-size and labeling extensions were added on top of their work and are also being contributed back upstream.

## References

1. Little, R. J. A. (1988). A Test of Missing Completely at Random for Multivariate Data with Missing Values. *Journal of the American Statistical Association*, 83(404), 1198-1202. https://doi.org/10.1080/01621459.1988.10478722

2. Schouten, R. M., Lugtig, P., & Vink, G. (2018). Generating missing values for simulation purposes: A multivariate amputation procedure. *Journal of Statistical Computation and Simulation*, 88(15), 2909-2930. https://doi.org/10.1080/00949655.2018.1491577

3. Schouten, R. M., Zamanzadeh, D., & Singh, P. (2022). *pyampute: a Python library for data amputation*. Zenodo. https://doi.org/10.25080/majora-212e5952-03e
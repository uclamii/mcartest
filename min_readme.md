<picture><img src="https://github.com/uclamii/mcartest/blob/main/assets/mcartest_logo.svg" width="350" alt="My Logo"/></picture>

MCAR statistical tests in Python: Little's chi-square test and pairwise t-tests, extended with Cohen's d effect sizes and MCAR/not-MCAR labeling

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

<!-- ```bibtex
@software{shpaner_mcartest,
   author       = {Shpaner, Leonid},
   title        = {mcartest},
   year         = 2026,
   publisher    = {Zenodo},
   version      = {0.0.0a},
   doi          = {[YOUR_DOI]},
   url          = {[YOUR_ZENODO_URL]}
}
``` -->

## Support

If you have any questions or issues with `mcartest`, please open an issue on this [GitHub repository](https://github.com/uclamii/mcartest/).

## Acknowledgements

This work builds on [pyampute](https://github.com/RianneSchouten/pyampute) by Rianne Schouten and Davina Zamanzadeh, whose implementation of Little's MCAR test and the pairwise t-test approach forms the statistical core of this library. The effect-size and labeling extensions were added on top of their work and are also being contributed back upstream.
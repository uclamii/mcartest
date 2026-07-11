from .mcar_stats_tests import MCARTest
from ._dataframe_utils import style_significant, style_label, style_effect
from .logo import *

import sys
import builtins

# Detailed Documentation
detailed_doc = """
The `mcartest` library provides statistical tests for assessing whether missing
data is Missing Completely At Random (MCAR). It implements Little's chi-square
test across the whole dataset and pairwise t-tests between every combination of
features, extended with Cohen's d effect sizes and human-readable MCAR /
not-MCAR labeling.

Effect sizes complement the p-values: a significant result driven by a large
sample may still reflect a trivial difference, while a large effect in a sparse
missing-group may not reach significance. Styling helpers are included for
rendering the resulting matrices in notebooks and exporting to Excel.

This library builds on pyampute (BSD 3-Clause) by Rianne Schouten and Davina
Zamanzadeh.

PyPI: https://pypi.org/project/mcartest
Documentation: https://uclamii.github.io/mcartest/
Version: 0.0.0a3
"""

# Assign only the detailed documentation to __doc__
__doc__ = detailed_doc

__version__ = "0.0.0a3"
__author__ = "Leonid Shpaner"
__email__ = "lshpaner@ucla.edu"

__all__ = [
    "MCARTest",
    "style_significant",
    "style_label",
    "style_effect",
]



# Define the custom help function
def custom_help(obj=None):
    """
    Custom help function to dynamically include ASCII art in help() output.
    """
    if (
        obj is None or obj is sys.modules[__name__]
    ):  # When `help()` is called for this module
        print(mcartest_logo)  # Print ASCII art first
        print(detailed_doc)  # Print the detailed documentation
    else:
        original_help(obj)  # Use the original help for other objects


# Backup the original help function
original_help = builtins.help

# Override the global help function in builtins
builtins.help = custom_help
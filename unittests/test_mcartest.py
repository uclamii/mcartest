"""Tests for mcartest."""

import pandas as pd
import numpy as np
import pytest

from mcartest.mcar_stats_tests import (
    MCARTest,
    _cohens_d,
    _effect_label,
    _elementwise,
)
from mcartest._dataframe_utils import (
    style_significant,
    style_label,
    style_effect,
    add_missing_counts,
    _css,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mcar_df():
    """Data where missingness is unrelated to any value (true MCAR)."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "a": rng.normal(size=300),
            "b": rng.normal(size=300),
            "c": rng.normal(size=300),
        }
    )
    # knock out cells at random, independent of the data itself
    mask = rng.random(size=df.shape) < 0.2
    return df.mask(mask)


@pytest.fixture
def mar_df():
    """Data where missingness in 'b' depends on 'a' (not MCAR)."""
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "a": rng.normal(size=400),
            "b": rng.normal(size=400),
            "c": rng.normal(size=400),
        }
    )
    # b goes missing only in the upper tail of a: a strong, detectable link
    df.loc[df["a"] > 0.5, "b"] = np.nan
    return df


@pytest.fixture
def mixed_dtype_df(mcar_df):
    """MCAR data with a non-numeric column bolted on."""
    df = mcar_df.copy()
    df["label"] = "category_" + (df.index % 3).astype(str)
    return df


# ---------------------------------------------------------------------------
# _cohens_d
# ---------------------------------------------------------------------------


class TestCohensD:
    def test_returns_magnitude_only(self):
        """Sign is discarded; only magnitude is reported."""
        low = pd.Series([1.0, 2.0, 3.0, 4.0])
        high = pd.Series([11.0, 12.0, 13.0, 14.0])
        assert _cohens_d(low, high) == _cohens_d(high, low)
        assert _cohens_d(low, high) > 0

    def test_identical_groups_give_zero(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0])
        assert _cohens_d(s, s) == pytest.approx(0.0)

    def test_known_value(self):
        a = pd.Series([1.0, 2.0, 3.0])
        b = pd.Series([2.0, 3.0, 4.0])
        assert abs(_cohens_d(a, b)) == pytest.approx(1.0)

    def test_too_few_samples_returns_nan(self):
        assert np.isnan(_cohens_d(pd.Series([1.0]), pd.Series([1.0, 2.0, 3.0])))
        assert np.isnan(_cohens_d(pd.Series([], dtype=float), pd.Series([1.0, 2.0])))

    def test_zero_variance_returns_nan(self):
        constant = pd.Series([5.0, 5.0, 5.0])
        assert np.isnan(_cohens_d(constant, constant))

    def test_nans_are_dropped(self):
        with_nan = pd.Series([1.0, 2.0, 3.0, np.nan])
        without = pd.Series([1.0, 2.0, 3.0])
        other = pd.Series([4.0, 5.0, 6.0])
        assert _cohens_d(with_nan, other) == pytest.approx(_cohens_d(without, other))


# ---------------------------------------------------------------------------
# _effect_label
# ---------------------------------------------------------------------------


class TestEffectLabel:
    @pytest.mark.parametrize(
        "d,expected",
        [
            (0.0, "negligible"),
            (0.19, "negligible"),
            (0.2, "small"),
            (0.49, "small"),
            (0.5, "medium"),
            (0.79, "medium"),
            (0.8, "large"),
            (5.0, "large"),
        ],
    )
    def test_bands(self, d, expected):
        assert _effect_label(d) == expected

    def test_uses_absolute_value(self):
        assert _effect_label(-0.9) == "large"
        assert _effect_label(-0.05) == "negligible"

    def test_nan_passes_through(self):
        assert pd.isna(_effect_label(np.nan))

    def test_custom_thresholds(self):
        assert _effect_label(0.35, thresholds=(0.4, 0.6, 0.9)) == "negligible"
        assert _effect_label(0.45, thresholds=(0.4, 0.6, 0.9)) == "small"


# ---------------------------------------------------------------------------
# _elementwise
# ---------------------------------------------------------------------------


class TestElementwise:
    def test_applies_to_every_cell(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        out = _elementwise(df, lambda x: x * 10)
        pd.testing.assert_frame_equal(out, pd.DataFrame({"a": [10, 20], "b": [30, 40]}))

    def test_preserves_shape_and_labels(self):
        df = pd.DataFrame({"x": [1.0]}, index=["r"])
        out = _elementwise(df, lambda v: v)
        assert out.shape == df.shape
        assert list(out.index) == ["r"]
        assert list(out.columns) == ["x"]


# ---------------------------------------------------------------------------
# little_mcar_test
# ---------------------------------------------------------------------------


class TestLittleMCARTest:
    def test_returns_float_pvalue(self, mcar_df):
        p = MCARTest.little_mcar_test(mcar_df)
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0

    def test_return_stats_keys(self, mcar_df):
        stats = MCARTest.little_mcar_test(mcar_df, return_stats=True)
        assert set(stats) == {"pvalue", "statistic", "df", "effect_size"}

    def test_return_stats_pvalue_matches_default(self, mcar_df):
        plain = MCARTest.little_mcar_test(mcar_df)
        stats = MCARTest.little_mcar_test(mcar_df, return_stats=True)
        assert stats["pvalue"] == pytest.approx(plain)

    def test_effect_size_is_non_negative(self, mcar_df):
        stats = MCARTest.little_mcar_test(mcar_df, return_stats=True)
        assert stats["effect_size"] >= 0

    def test_callable_dispatches_to_little(self, mcar_df):
        mt = MCARTest(method="little")
        assert mt(mcar_df) == pytest.approx(MCARTest.little_mcar_test(mcar_df))


# ---------------------------------------------------------------------------
# mcar_t_tests: shape and defaults
# ---------------------------------------------------------------------------


class TestMcarTTestsBasics:
    def test_returns_square_matrix(self, mcar_df):
        out = MCARTest.mcar_t_tests(mcar_df)
        assert out.shape == (3, 3)
        assert list(out.index) == list(out.columns) == ["a", "b", "c"]

    def test_default_returns_dataframe_not_tuple(self, mcar_df):
        out = MCARTest.mcar_t_tests(mcar_df)
        assert isinstance(out, pd.DataFrame)

    def test_effect_size_returns_tuple(self, mcar_df):
        out = MCARTest.mcar_t_tests(mcar_df, effect_size=True)
        assert isinstance(out, tuple)
        assert len(out) == 2
        pvals, effects = out
        assert pvals.shape == effects.shape

    def test_non_numeric_columns_ignored(self, mixed_dtype_df):
        out = MCARTest.mcar_t_tests(mixed_dtype_df)
        assert "label" not in out.columns
        assert "label" not in out.index
        assert out.shape == (3, 3)

    def test_pvalues_in_unit_interval(self, mcar_df):
        out = MCARTest.mcar_t_tests(mcar_df)
        vals = out.to_numpy().ravel()
        vals = vals[~pd.isna(vals)]
        assert ((vals >= 0) & (vals <= 1)).all()

    def test_effects_are_non_negative(self, mcar_df):
        _, effects = MCARTest.mcar_t_tests(mcar_df, effect_size=True)
        vals = effects.to_numpy().ravel()
        vals = vals[~pd.isna(vals)]
        assert (vals >= 0).all()

    def test_input_is_not_mutated(self, mcar_df):
        before = mcar_df.copy()
        MCARTest.mcar_t_tests(mcar_df)
        pd.testing.assert_frame_equal(mcar_df, before)


# ---------------------------------------------------------------------------
# mcar_t_tests: statistical behavior
# ---------------------------------------------------------------------------


class TestMcarTTestsStatistics:
    def test_mar_data_is_detected(self, mar_df):
        """Missingness in b driven by a should reject MCAR for that pair."""
        pvals = MCARTest.mcar_t_tests(mar_df)
        assert pvals.loc["b", "a"] < 0.05

    def test_mar_data_shows_large_effect(self, mar_df):
        _, effects = MCARTest.mcar_t_tests(mar_df, effect_size=True)
        assert effects.loc["b", "a"] > 0.8

    def test_mcar_data_mostly_fails_to_reject(self, mcar_df):
        """Truly random missingness should rarely produce significant cells."""
        pvals = MCARTest.mcar_t_tests(mcar_df)
        vals = pvals.to_numpy().ravel()
        vals = vals[~pd.isna(vals)]
        significant = (vals < 0.05).mean()
        assert significant < 0.5


# ---------------------------------------------------------------------------
# mcar_t_tests: labeling modes
# ---------------------------------------------------------------------------


class TestLabelingModes:
    def test_label_mcar_values(self, mcar_df):
        out = MCARTest.mcar_t_tests(mcar_df, label_mcar=True)
        assert set(out.to_numpy().ravel()) <= {"MCAR", ""}

    def test_label_not_mcar_values(self, mcar_df):
        out = MCARTest.mcar_t_tests(mcar_df, label_not_mcar=True)
        assert set(out.to_numpy().ravel()) <= {"not MCAR", ""}

    def test_label_both_values(self, mcar_df):
        out = MCARTest.mcar_t_tests(mcar_df, label_both=True)
        assert set(out.to_numpy().ravel()) <= {"MCAR", "not MCAR", ""}

    def test_label_both_blank_only_where_untestable(self, mcar_df):
        """Under label_both, a blank cell must correspond to a NaN p-value."""
        pvals = MCARTest.mcar_t_tests(mcar_df)
        both = MCARTest.mcar_t_tests(mcar_df, label_both=True)
        blank = both == ""
        pd.testing.assert_frame_equal(blank, pvals.isna())

    def test_labels_agree_with_pvalues(self, mar_df):
        pvals = MCARTest.mcar_t_tests(mar_df)
        both = MCARTest.mcar_t_tests(mar_df, label_both=True)
        for r in pvals.index:
            for c in pvals.columns:
                p = pvals.loc[r, c]
                if pd.isna(p):
                    assert both.loc[r, c] == ""
                elif p > 0.05:
                    assert both.loc[r, c] == "MCAR"
                else:
                    assert both.loc[r, c] == "not MCAR"

    def test_mcar_and_not_mcar_are_complementary(self, mar_df):
        """Every testable cell is labeled by exactly one of the two modes."""
        mcar = MCARTest.mcar_t_tests(mar_df, label_mcar=True)
        not_mcar = MCARTest.mcar_t_tests(mar_df, label_not_mcar=True)
        pvals = MCARTest.mcar_t_tests(mar_df)

        labeled_by_one = (mcar != "") ^ (not_mcar != "")
        testable = pvals.notna()
        pd.testing.assert_frame_equal(labeled_by_one, testable)

    def test_alpha_is_respected(self, mcar_df):
        strict = MCARTest.mcar_t_tests(mcar_df, label_mcar=True, alpha=0.001)
        loose = MCARTest.mcar_t_tests(mcar_df, label_mcar=True, alpha=0.5)
        # a higher bar for "MCAR" means at most as many MCAR cells
        assert (loose == "MCAR").sum().sum() <= (strict == "MCAR").sum().sum()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"label_mcar": True, "label_not_mcar": True},
            {"label_mcar": True, "label_both": True},
            {"label_not_mcar": True, "label_both": True},
            {"label_mcar": True, "label_not_mcar": True, "label_both": True},
        ],
    )
    def test_conflicting_label_flags_raise(self, mcar_df, kwargs):
        with pytest.raises(ValueError, match="at most one"):
            MCARTest.mcar_t_tests(mcar_df, **kwargs)


# ---------------------------------------------------------------------------
# mcar_t_tests: effect-size modes
# ---------------------------------------------------------------------------


class TestEffectModes:
    def test_size_label_values(self, mcar_df):
        _, effects = MCARTest.mcar_t_tests(mcar_df, effect_size=True, size_label=True)
        vals = {v for v in effects.to_numpy().ravel() if not pd.isna(v)}
        assert vals <= {"negligible", "small", "medium", "large"}

    def test_size_label_matches_numeric_bands(self, mcar_df):
        _, numeric = MCARTest.mcar_t_tests(mcar_df, effect_size=True)
        _, labeled = MCARTest.mcar_t_tests(mcar_df, effect_size=True, size_label=True)
        for r in numeric.index:
            for c in numeric.columns:
                d = numeric.loc[r, c]
                if pd.isna(d):
                    assert pd.isna(labeled.loc[r, c])
                else:
                    assert labeled.loc[r, c] == _effect_label(d)

    def test_effect_if_not_mcar_blanks_the_mcar_cells(self, mar_df):
        pvals = MCARTest.mcar_t_tests(mar_df)
        _, effects = MCARTest.mcar_t_tests(
            mar_df, effect_size=True, effect_if_not_mcar=True
        )
        for r in pvals.index:
            for c in pvals.columns:
                p = pvals.loc[r, c]
                if pd.isna(p) or p > 0.05:
                    assert effects.loc[r, c] == ""
                else:
                    assert effects.loc[r, c] in {
                        "negligible",
                        "small",
                        "medium",
                        "large",
                    }

    def test_effect_if_not_mcar_implies_labels(self, mar_df):
        """Gating produces label strings even without size_label set."""
        _, effects = MCARTest.mcar_t_tests(
            mar_df, effect_size=True, effect_if_not_mcar=True
        )
        vals = {v for v in effects.to_numpy().ravel() if v != ""}
        assert vals <= {"negligible", "small", "medium", "large"}

    def test_pvalues_unaffected_by_effect_flags(self, mcar_df):
        plain = MCARTest.mcar_t_tests(mcar_df)
        pvals, _ = MCARTest.mcar_t_tests(mcar_df, effect_size=True, size_label=True)
        pd.testing.assert_frame_equal(plain, pvals)


# ---------------------------------------------------------------------------
# Filtering pattern (the documented mask workflow)
# ---------------------------------------------------------------------------


class TestMaskWorkflow:
    def test_shared_mask_keeps_views_aligned(self, mcar_df):
        """The documented pattern: build the mask once, apply it to every view."""
        pvals = MCARTest.mcar_t_tests(mcar_df)
        both = MCARTest.mcar_t_tests(mcar_df, label_both=True)
        _, effects = MCARTest.mcar_t_tests(
            mcar_df, effect_size=True, effect_if_not_mcar=True
        )

        mask = pvals.notna().any(axis=1)

        views = [pvals[mask].T, both[mask].T, effects[mask].T]
        shapes = {v.shape for v in views}
        assert len(shapes) == 1

        first = views[0]
        for v in views[1:]:
            assert list(v.index) == list(first.index)
            assert list(v.columns) == list(first.columns)


# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------


class TestCss:
    def test_both_colors(self):
        assert _css("red", "white") == "background-color: red; color: white"

    def test_background_only(self):
        assert _css("red", None) == "background-color: red"

    def test_text_only(self):
        assert _css(None, "white") == "color: white"

    def test_neither(self):
        assert _css(None, None) == ""


class TestStylers:
    def test_style_significant_returns_styler(self, mcar_df):
        pvals = MCARTest.mcar_t_tests(mcar_df)
        styler = style_significant(pvals)
        assert isinstance(styler, pd.io.formats.style.Styler)
        styler.to_html()  # must render without raising

    def test_style_label_returns_styler(self, mcar_df):
        both = MCARTest.mcar_t_tests(mcar_df, label_both=True)
        styler = style_label(both)
        assert isinstance(styler, pd.io.formats.style.Styler)
        styler.to_html()

    def test_style_effect_returns_styler(self, mcar_df):
        _, effects = MCARTest.mcar_t_tests(mcar_df, effect_size=True, size_label=True)
        styler = style_effect(effects)
        assert isinstance(styler, pd.io.formats.style.Styler)
        styler.to_html()

    def test_style_significant_colors_only_above_alpha(self):
        df = pd.DataFrame({"x": [0.01, 0.9, np.nan]})
        html = style_significant(df, color="green").to_html()
        # exactly one cell (the 0.9) should carry the background
        assert html.count("background-color: green") == 1

    def test_style_label_distinguishes_the_two_labels(self):
        df = pd.DataFrame({"x": ["MCAR", "not MCAR", ""]})
        html = style_label(df, mcar_color="green", not_mcar_color="crimson").to_html()
        assert "background-color: green" in html
        assert "background-color: crimson" in html

    def test_custom_colors_are_used(self):
        df = pd.DataFrame({"x": ["large"]})
        html = style_effect(df, large="#123456").to_html()
        assert "#123456" in html

    def test_unsupported_method(self, mcar_df):
        mt = MCARTest(method="bogus")
        mt(mcar_df)  # currently logs an error and returns None


# ---------------------------------------------------------------------------
# add_missing_counts
# ---------------------------------------------------------------------------


class TestAddMissingCounts:
    def test_column_is_prepended(self, mcar_df):
        pvals = MCARTest.mcar_t_tests(mcar_df)
        out = add_missing_counts(pvals, mcar_df)
        assert out.columns[0] == "n_missing"
        assert out.shape[1] == pvals.shape[1] + 1

    def test_counts_are_correct(self):
        source = pd.DataFrame(
            {
                "a": [1.0, np.nan, 3.0, np.nan],
                "b": [1.0, 2.0, 3.0, 4.0],
                "c": [np.nan, np.nan, np.nan, 4.0],
            }
        )
        matrix = pd.DataFrame(index=["a", "b", "c"], columns=["x"], data=[0, 0, 0])
        out = add_missing_counts(matrix, source)
        assert out.loc["a", "n_missing"] == 2
        assert out.loc["b", "n_missing"] == 0
        assert out.loc["c", "n_missing"] == 3

    def test_aligns_by_variable_name_not_position(self):
        """Counts follow the variable, whatever order the matrix is in."""
        source = pd.DataFrame(
            {
                "a": [1.0, np.nan, np.nan],
                "b": [1.0, 2.0, 3.0],
            }
        )
        # matrix rows deliberately reversed relative to source columns
        matrix = pd.DataFrame(index=["b", "a"], columns=["x"], data=[0, 0])
        out = add_missing_counts(matrix, source)
        assert out.loc["a", "n_missing"] == 2
        assert out.loc["b", "n_missing"] == 0

    def test_works_after_transpose(self, mcar_df):
        """Orientation is decided by the caller; the helper must not care."""
        pvals = MCARTest.mcar_t_tests(mcar_df)
        mask = pvals.notna().any(axis=1)
        transposed = pvals[mask].T

        out = add_missing_counts(transposed, mcar_df)
        expected = mcar_df.isna().sum()
        for var in out.index:
            assert out.loc[var, "n_missing"] == expected[var]

    def test_missing_variable_gets_nan(self):
        """A row not present in the source yields NaN rather than raising."""
        source = pd.DataFrame({"a": [1.0, np.nan]})
        matrix = pd.DataFrame(index=["a", "ghost"], columns=["x"], data=[0, 0])
        out = add_missing_counts(matrix, source)
        assert out.loc["a", "n_missing"] == 1
        assert pd.isna(out.loc["ghost", "n_missing"])

    def test_does_not_mutate_input(self, mcar_df):
        pvals = MCARTest.mcar_t_tests(mcar_df)
        before = pvals.copy()
        add_missing_counts(pvals, mcar_df)
        pd.testing.assert_frame_equal(pvals, before)

    def test_custom_column_name(self, mcar_df):
        pvals = MCARTest.mcar_t_tests(mcar_df)
        out = add_missing_counts(pvals, mcar_df, col_name="n_na")
        assert out.columns[0] == "n_na"
        assert "n_missing" not in out.columns

    def test_styler_ignores_the_count_column(self):
        """Counts render as plain numbers, not as effect bands."""
        matrix = pd.DataFrame({"x": ["large"]}, index=["a"])
        source = pd.DataFrame({"a": [1.0, np.nan, np.nan]})
        out = add_missing_counts(matrix, source)

        html = style_effect(out, large="#123456").to_html()
        assert "#123456" in html  # the label still gets colored
        assert ">2<" in html  # the count renders untouched

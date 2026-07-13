import pandas as pd


def _style_map(styler, func):
    """Apply func to every cell of a Styler, across pandas versions.

    Styler.map was added in pandas 2.1.0; earlier versions use Styler.applymap.
    """
    try:
        return styler.map(func)  # pandas >= 2.1
    except AttributeError:
        return styler.applymap(func)  # pandas < 2.1


def _css(bg, text):
    """Build a CSS string from optional background and text colors."""
    parts = []
    if bg:
        parts.append(f"background-color: {bg}")
    if text:
        parts.append(f"color: {text}")
    return "; ".join(parts)


def style_significant(df, color="green", text_color=None, alpha=0.05):
    """Shade cells where p > alpha (consistent with MCAR).

    Parameters
    ----------
    df : pandas.DataFrame
        The p-value matrix to style.
    color : str
        Background color for cells above the threshold. Any CSS color.
    text_color : str, optional
        Text color for those cells. Default leaves it unchanged.
    alpha : float, default 0.05
        Significance threshold.
    """
    style = _css(color, text_color)

    def _fn(val):
        return style if pd.notna(val) and val > alpha else ""

    return _style_map(df.style, _fn)


def style_label(
    df,
    mcar_color="green",
    not_mcar_color="#c0392b",
    mcar_text="white",
    not_mcar_text="white",
):
    """Shade labeled cells, coloring 'MCAR' and 'not MCAR' distinctly.

    Works whether the matrix contains only one label or both. Blank and NaN
    cells are left unstyled.

    Parameters
    ----------
    df : pandas.DataFrame
        The label matrix to style (from label_mcar, label_not_mcar, or
        label_both).
    mcar_color, not_mcar_color : str
        Background colors for 'MCAR' and 'not MCAR' cells. Any CSS color.
    mcar_text, not_mcar_text : str, optional
        Text colors for each. Default white.
    """
    styles = {
        "MCAR": _css(mcar_color, mcar_text),
        "not MCAR": _css(not_mcar_color, not_mcar_text),
    }

    def _fn(val):
        return styles.get(val, "")

    return _style_map(df.style, _fn)


def style_effect(
    df,
    large="#27ae60",
    medium="#2ecc71",
    small="#a9dfbf",
    negligible="#eafaf1",
    text_color=None,
    large_text=None,
    medium_text=None,
    small_text=None,
    negligible_text=None,
):
    """Shade cells by effect-size band, with per-band colors.

    text_color sets the text color for all bands; the per-band *_text
    args override it for individual bands when given.
    """
    band_styles = {
        "large": _css(large, large_text if large_text is not None else text_color),
        "medium": _css(medium, medium_text if medium_text is not None else text_color),
        "small": _css(small, small_text if small_text is not None else text_color),
        "negligible": _css(
            negligible, negligible_text if negligible_text is not None else text_color
        ),
    }

    def _fn(val):
        return band_styles.get(val, "")

    return _style_map(df.style, _fn)


def add_missing_counts(df, source, col_name="n_missing"):
    """Prepend each variable's missing-value count to a result matrix.

    An effect size means something very different depending on how much data
    is actually missing for that variable: a large effect backed by three
    missing rows is noise, while the same effect backed by three hundred is
    worth acting on. Counts are aligned by variable name, so this works
    regardless of how the matrix has been transposed.

    Parameters
    ----------
    df : pandas.DataFrame
        The result matrix (p-values, labels, or effect sizes), in whatever
        orientation you want it.
    source : pandas.DataFrame
        The original data the tests were run on, used to count missing values.
    col_name : str, default "n_missing"
        Name for the inserted column.

    Returns
    -------
    pandas.DataFrame
        A copy of `df` with the count column inserted as the first column.
        Variables absent from `source` receive NaN.
    """
    counts = source.isna().sum()
    out = df.copy()
    out.insert(0, col_name, counts.reindex(out.index))
    return out

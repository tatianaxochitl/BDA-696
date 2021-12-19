import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly import figure_factory as ff
from scipy import stats


# make categorical vs categorical graph
def heat_mx(df, predictor, response):
    crosstab_df = pd.crosstab(
        index=df[predictor], columns=df[response], margins=False
    )  # noqa: E501

    fig = go.Figure(
        data=go.Heatmap(
            x=crosstab_df.index,
            y=crosstab_df.columns,
            z=crosstab_df.values,
            zmin=0,
            zmax=crosstab_df.max().max(),
        )
    )  # noqa: E501

    fig.update_layout(
        title=f"{predictor} {response} Heatmap",
        xaxis_title=f"{predictor}",
        yaxis_title=f"{response}",
    )

    filename = f"plots/{predictor}_{response}_heatmap.html"

    fig.write_html(
        file=filename,
        include_plotlyjs="cdn",
    )

    return filename


# Used correlation function from Lecture 7
def fill_na(data):
    if isinstance(data, pd.Series):
        return data.fillna(0)
    else:
        return np.array([value if value is not None else 0 for value in data])


def cat_correlation(x, y, bias_correction=True, tschuprow=False):
    """
    Calculates correlation statistic for categorical-categorical association.
    The two measures supported are:
    1. Cramer'V ( default )
    2. Tschuprow'T

    SOURCES:
    1.) CODE: https://github.com/MavericksDS/pycorr
    2.) Used logic from:
        https://stackoverflow.com/questions/20892799/using-pandas-calculate-cram%C3%A9rs-coefficient-matrix
        to ignore yates correction factor on 2x2
    3.) Haven't validated Tschuprow

    Bias correction and formula's taken from :
    https://www.researchgate.net/publication/270277061_A_bias-correction_for_Cramer's_V_and_Tschuprow's_T

    Wikipedia for Cramer's V: https://en.wikipedia.org/wiki/Cram%C3%A9r%27s_V
    Wikipedia for Tschuprow' T: https://en.wikipedia.org/wiki/Tschuprow%27s_T
    Parameters:
    -----------
    x : list / ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    bias_correction : Boolean, default = True
    tschuprow : Boolean, default = False
               For choosing Tschuprow as measure
    Returns:
    --------
    float in the range of [0,1]
    """
    corr_coeff = np.nan
    try:
        # using pandas since only passing df through this program
        if pd.isna(x).any():
            x = fill_na(x)
        if pd.isna(y).any():
            y = fill_na(y)
        crosstab_matrix = pd.crosstab(x, y)
        n_obs = crosstab_matrix.sum().sum()

        yates_correct = True
        if bias_correction:
            if crosstab_matrix.shape == (2, 2):
                yates_correct = False

        chi2, _, _, _ = stats.chi2_contingency(
            crosstab_matrix, correction=yates_correct
        )
        phi2 = chi2 / n_obs

        # r and c are number of categories of x and y
        r, c = crosstab_matrix.shape
        if bias_correction:
            phi2_correct = max(0, phi2 - ((r - 1) * (c - 1)) / (n_obs - 1))
            r_correct = r - ((r - 1) ** 2) / (n_obs - 1)
            c_correct = c - ((c - 1) ** 2) / (n_obs - 1)
            if tschuprow:
                corr_coeff = np.sqrt(
                    phi2_correct / np.sqrt((r_correct - 1) * (c_correct - 1))
                )
                return corr_coeff
            corr_coeff = np.sqrt(
                phi2_correct / min((r_correct - 1), (c_correct - 1))
            )  # noqa: E501
            return corr_coeff
        if tschuprow:
            corr_coeff = np.sqrt(phi2 / np.sqrt((r - 1) * (c - 1)))
            return corr_coeff
        corr_coeff = np.sqrt(phi2 / min((r - 1), (c - 1)))
        return corr_coeff
    except Exception as ex:
        print(ex)
        if tschuprow:
            warnings.warn("Error calculating Tschuprow's T", RuntimeWarning)
        else:
            warnings.warn("Error calculating Cramer's V", RuntimeWarning)
        return corr_coeff


def cat_cat_dwm(df: pd.DataFrame, pred1: str, pred2: str, response: str):
    # getting unique values for each category
    categories1 = df[pred1].unique().astype(str)
    categories2 = df[pred2].unique().astype(str)
    categories1 = sorted(categories1)
    categories2 = sorted(categories2)

    # get mean and bin count
    bin_mean = pd.crosstab(
        index=df[pred1],
        columns=df[pred2],
        values=df[response],
        aggfunc="mean",
        margins=False,
    )
    pop_mean = df[response].mean()
    bin_count = pd.crosstab(index=df[pred1], columns=df[pred2], margins=False)

    # calculating diff w/ mean (not weighted & weighted)
    diff = bin_mean - pop_mean
    sq_diff = np.square(diff)
    sum_sq_diff = np.nansum(sq_diff)
    total_bin = len(categories2) * len(categories1)
    msd = sum_sq_diff / total_bin
    pop_prop = bin_count / len(df[response])
    w_sq_diff = pop_prop * sq_diff
    wmsd = np.nansum(w_sq_diff)

    pop_prop = pop_prop.round(3)
    fig = ff.create_annotated_heatmap(
        x=bin_mean.columns.tolist(),
        y=bin_mean.index.tolist(),
        z=bin_mean.values,
        annotation_text=bin_mean.values.round(3),
        colorscale="curl",
        showscale=True,
        customdata=pop_prop.values,
        hovertemplate="<b>x</b>: %{x}<br>"
        + "<b>y</b>: %{y}<br>"
        + "<b>z</b>: %{z}"
        + "<extra>Population<br>Proportion: %{customdata}</extra>",
        hoverongaps=False,
        zmid=pop_mean,
        zmin=df[response].min(),
        zmax=df[response].max(),
    )

    fig.update_layout(title_text=f"{pred1} & {pred2} Bin Averages of Response")
    filename1 = f"plots/{pred1}_{pred2}_diff_of_mean_resp_bin.html"

    fig.write_html(
        file=filename1,
        include_plotlyjs="cdn",
    )

    # residual plot
    # diff_w_pop = str(diff) + "(pop: " + str(bin_count) + ")"
    fig_2 = ff.create_annotated_heatmap(
        x=categories2,
        y=categories1,
        z=diff.values,
        colorscale="curl",
        customdata=pop_prop,
        hovertemplate="<b>x</b>: %{x}<br>"
        + "<b>y</b>: %{y}<br>"
        + "<b>z</b>: %{z}"
        + "<extra>Population<br>Proportion: %{customdata}</extra>",
        annotation_text=diff.values.round(3),
        hoverongaps=False,
        showscale=True,
        zmid=pop_mean,
        zmin=df[response].min(),
        zmax=df[response].max(),
    )

    fig_2.update_layout(title_text=f"{pred1} & {pred2} Bin Average")
    filename2 = f"plots/{pred1}_{pred2}_dwm_of_resp_residual.html"

    fig_2.write_html(
        file=filename2,
        include_plotlyjs="cdn",
    )

    return filename1, filename2, round(msd, 3), round(wmsd, 3)

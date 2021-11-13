import itertools
import os
import random
import re
import string
import warnings
import webbrowser

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_string_dtype,
)
from plotly import figure_factory as ff
from plotly import graph_objects as go
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder


def process_dataframe(
    pandas_df: pd.DataFrame, predictor_columns: list, response_column: str
):
    # set up things create dir for plots
    path = os.path.join(os.getcwd(), "plots")
    if not os.path.exists(path):
        os.mkdir(path)

    # lists to put separate predictors and response var
    cont_list = []
    cat_list = []

    # seperate predictors first
    for pred in predictor_columns:
        pred_data = pandas_df[pred]
        if (
            is_string_dtype(pred_data)
            or is_bool_dtype(pred_data)
            or is_datetime64_any_dtype(pred_data)
            or is_categorical_dtype(pred_data)
        ):
            cat_list.append(pred)
        elif is_numeric_dtype(pred_data):
            uniq = pred_data.unique()
            sorted_diff = sum(np.diff(sorted(uniq)))
            if len(uniq) <= 5 and sorted_diff == (len(uniq) - 1):
                cat_list.append(pred)
            else:
                cont_list.append(pred)
        else:
            raise ValueError("Predictor is neither categorical nor continous.")

    # # seperate response
    # if is_bool_dtype(pandas_df[response_column]):
    #     resp = "boolean"
    # elif is_numeric_dtype(pandas_df[response_column]):
    #     resp = "continuous"
    # else:
    #     raise ValueError("Response is neither boolean nor continous.")

    # Continuous/Continuous
    cont_cont_df = pd.DataFrame(
        columns=[
            "Predictors",
            "Pearson's r",
            "Absolute Value of Correlation",
            "Linear Regression Plot",
        ]
    )
    pred_cont_list_comb = itertools.combinations(cont_list, 2)
    cont_cont_corr_matrix = pd.DataFrame(columns=cont_list, index=cont_list)
    cont_cont_diff = pd.DataFrame(
        columns=[
            "Predictor 1",
            "Predictor 2",
            "Difference of Mean Response",
            "Weighted Difference of Mean Response",
            "Bin Plot",
            "Residual Plot",
        ]
    )
    for pred1, pred2 in pred_cont_list_comb:
        # Plotting
        file = linear_regression(pandas_df, pred1, pred2)
        # Correlation Stats
        cont_cont_corr, p = stats.pearsonr(pandas_df[pred1], pandas_df[pred2])
        # Put value in correlation matrix
        cont_cont_corr_matrix.at[pred1, pred2] = cont_cont_corr
        cont_cont_corr_matrix.at[pred2, pred1] = cont_cont_corr
        # Put correlation value and plot into table
        new_row = {
            "Predictors": f"{pred1} and {pred2}",
            "Pearson's r": cont_cont_corr,
            "Absolute Value of Correlation": abs(cont_cont_corr),
            "Linear Regression Plot": file,
        }
        cont_cont_df = cont_cont_df.append(new_row, ignore_index=True)
        # Brute Force
        file1, file2, diff, w_diff = cont_cont_dwm(
            pandas_df, pred1, pred2, response_column
        )
        new_row = {
            "Predictor 1": pred1,
            "Predictor 2": pred2,
            "Difference of Mean Response": diff,
            "Weighted Difference of Mean Response": w_diff,
            "Bin Plot": file1,
            "Residual Plot": file2,
        }
        cont_cont_diff = cont_cont_diff.append(new_row, ignore_index=True)

    # Continuous/Categorical
    cont_cat_df = pd.DataFrame(
        columns=[
            "Predictors",
            "Correlation Ratio",
            "Absolute Value of Correlation",
            "Violin Plot",
            "Distribution Plot",
        ]
    )
    cont_cat_corr_matrix = pd.DataFrame(columns=cat_list, index=cont_list)
    cont_cat_diff = pd.DataFrame(
        columns=[
            "Predictor 1",
            "Predictor 2",
            "Difference of Mean Response",
            "Weighted Difference of Mean Response",
            "Bin Plot",
            "Residual Plot",
        ]
    )
    all_pred_df = pd.DataFrame(
        columns=[
            "Predictor",
            "p-value",
            "t-value",
            "Random Forest Importance",
            "Difference of Mean Response",
            "Weighted Difference of Mean Response",
            "Bin Plot",
            "Response Plot",
            "Response Plot 2",
        ]
    )
    rf = rf_importance(pandas_df, cont_list, response_column)
    i = 0
    for pred1 in cont_list:
        # ranking and plots for cont
        p_value, t_value = logistic_regression(
            pandas_df, pred1, response_column
        )  # noqa: E501
        file1, msd, wmsd = cont_dwm(pandas_df, pred1, response_column)
        file2, file3 = cont_cat_graph(pandas_df, pred1, response_column)
        new_row = {
            "Predictor": pred1,
            "p-value": p_value,
            "t-value": t_value,
            "Random Forest Importance": rf[i],
            "Difference of Mean Response": msd,
            "Weighted Difference of Mean Response": wmsd,
            "Bin Plot": file1,
            "Response Plot": file2,
            "Response Plot 2": file3,
        }
        all_pred_df = all_pred_df.append(new_row, ignore_index=True)
        i += 1
        for pred2 in cat_list:
            # ranking and plots for cat
            file1, msd, wmsd = cat_dwm(pandas_df, pred2, response_column)
            file2 = heat_mx(pandas_df, pred2, response_column)
            new_row = {
                "Predictor": pred2,
                "Difference of Mean Response": msd,
                "Weighted Difference of Mean Response": wmsd,
                "Bin Plot": file1,
                "Response Plot": file2,
            }
            all_pred_df = all_pred_df.append(new_row, ignore_index=True)
            # Plotting
            file1, file2 = cont_cat_graph(pandas_df, pred1, pred2)
            # Correlation Stats
            pred1_array = pandas_df[pred1].to_numpy()
            pred2_array = pandas_df[pred2].to_numpy()
            cont_cat_corr = cat_cont_corr_ratio(pred2_array, pred1_array)
            # Put value in correlation matrix
            cont_cat_corr_matrix.at[pred1, pred2] = cont_cat_corr
            # Put correlation value and plots into table
            new_row = {
                "Predictors": f"{pred1} and {pred2}",
                "Correlation Ratio": cont_cat_corr,
                "Absolute Value of Correlation": abs(cont_cat_corr),
                "Violin Plot": file1,
                "Distribution Plot": file2,
            }
            cont_cat_df = cont_cat_df.append(new_row, ignore_index=True)
            # Brute Force
            file1, file2, diff, w_diff = cont_cat_dwm(
                pandas_df, pred1, pred2, response_column
            )
            new_row = {
                "Predictor 1": pred1,
                "Predictor 2": pred2,
                "Difference of Mean Response": diff,
                "Weighted Difference of Mean Response": w_diff,
                "Bin Plot": file1,
                "Residual Plot": file2,
            }
            cont_cat_diff = cont_cat_diff.append(new_row, ignore_index=True)

    # Categorical/Categorical
    cat_cat_df = pd.DataFrame(
        columns=[
            "Predictors",
            "Cramer's V",
            "Absolute Value of Correlation",
            "Heatmap",
        ]  # noqa: E80
    )
    cat_cat_corr_matrix = pd.DataFrame(columns=cat_list, index=cat_list)
    cat_cat_diff = pd.DataFrame(
        columns=[
            "Predictor 1",
            "Predictor 2",
            "Difference of Mean Response",
            "Weighted Difference of Mean Response",
            "Bin Plot",
            "Residual Plot",
        ]
    )
    pred_cat_list_comb = itertools.combinations(cat_list, 2)
    for pred1, pred2 in pred_cat_list_comb:
        # Plotting
        file = heat_mx(pandas_df, pred1, pred2)
        # Correlations Stats
        cat_cat_corr = cat_correlation(pandas_df[pred1], pandas_df[pred2])
        # Put value in correlation matrix
        cat_cat_corr_matrix.at[pred1, pred2] = cat_cat_corr
        cat_cat_corr_matrix.at[pred2, pred1] = cat_cat_corr
        # Put correlation value and plot into table
        new_row = {
            "Predictors": f"{pred1} and {pred2}",
            "Cramer's V": cat_cat_corr,
            "Absolute Value of Correlation": abs(cat_cat_corr),
            "Heatmap": file,
        }
        cat_cat_df = cat_cat_df.append(new_row, ignore_index=True)
        # Brute Force
        file1, file2, diff, w_diff = cat_cat_dwm(
            pandas_df, pred1, pred2, response_column
        )
        new_row = {
            "Predictor 1": pred1,
            "Predictor 2": pred2,
            "Difference of Mean Response": diff,
            "Weighted Difference of Mean Response": w_diff,
            "Bin Plot": file1,
            "Residual Plot": file2,
        }
        cat_cat_diff = cat_cat_diff.append(new_row, ignore_index=True)

    # sort dataframes by abs val of corr
    cont_cont_df = cont_cont_df.sort_values(
        by="Absolute Value of Correlation", ascending=False
    )
    cont_cat_df = cont_cat_df.sort_values(
        by="Absolute Value of Correlation", ascending=False
    )
    cat_cat_df = cat_cat_df.sort_values(
        by="Absolute Value of Correlation", ascending=False
    )

    # fill empty with 1
    cont_cont_corr_matrix = cont_cont_corr_matrix.fillna(value=1)
    cat_cat_corr_matrix = cat_cat_corr_matrix.fillna(value=1)

    # sort dataframes by weighted dwmor
    cont_cont_diff = cont_cont_diff.sort_values(
        by="Weighted Difference of Mean Response", ascending=False
    )
    cont_cat_diff = cont_cat_diff.sort_values(
        by="Weighted Difference of Mean Response", ascending=False
    )
    cat_cat_diff = cat_cat_diff.sort_values(
        by="Weighted Difference of Mean Response", ascending=False
    )

    # make clickable links for all plots
    all_pred_df["Bin Plot"] = make_html_link(all_pred_df["Bin Plot"])

    all_pred_df["Response Plot"] = make_html_link(all_pred_df["Response Plot"])

    all_pred_df["Response Plot 2"] = make_html_link(
        all_pred_df["Response Plot 2"]
    )  # noqa: E501

    cont_cont_df["Linear Regression Plot"] = make_html_link(
        cont_cont_df["Linear Regression Plot"]
    )
    cont_cont_diff["Bin Plot"] = make_html_link(cont_cont_diff["Bin Plot"])
    cont_cont_diff["Residual Plot"] = make_html_link(
        cont_cont_diff["Residual Plot"]
    )  # noqa: E80
    cont_cat_df["Violin Plot"] = make_html_link(cont_cat_df["Violin Plot"])
    cont_cat_df["Distribution Plot"] = make_html_link(
        cont_cat_df["Distribution Plot"]
    )  # noqa: E80
    cont_cat_diff["Bin Plot"] = make_html_link(cont_cat_diff["Bin Plot"])
    cont_cat_diff["Residual Plot"] = make_html_link(
        cont_cat_diff["Residual Plot"]
    )  # noqa: E80
    cat_cat_df["Heatmap"] = make_html_link(cat_cat_df["Heatmap"])
    cat_cat_diff["Bin Plot"] = make_html_link(cat_cat_diff["Bin Plot"])
    cat_cat_diff["Residual Plot"] = make_html_link(
        cat_cat_diff["Residual Plot"]
    )  # noqa: E80

    # create html file
    tag = "".join(random.choices(string.ascii_uppercase + string.digits, k=5))
    page = open(f"midterm_analysis_{tag}.html", "w")
    header = f"""<!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" type="text/css" href="style.css">
        <title>Midterm</title>
    </head>
    <body>
    <h1>Predictor Analysis</h1>
    <h2>Continous/Continous Predictor Pairs</h2>
    """
    page.write(header)
    page.write("<h3>Predictor Ranking</h3>")
    page.write(
        all_pred_df.to_html(escape=False, index=False, justify="center")
    )  # noqa: E501
    page.write("<h3>Correlation Table</h3>")
    page.write(
        cont_cont_df.to_html(escape=False, index=False, justify="center")
    )  # noqa: E501
    page.write("<h3>Correlation Matrix</h3>")
    page.write(make_heatmap_html(cont_cont_corr_matrix))
    page.write('<h3>"Brute Force" Table</h3>')
    page.write(
        cont_cont_diff.to_html(escape=False, index=False, justify="center")
    )  # noqa: E501
    page.write("<h2>Continous/Categorical Predictor Pairs</h2>")
    page.write("<h3>Correlation Table</h3>")
    page.write(
        cont_cat_df.to_html(escape=False, index=False, justify="center")
    )  # noqa: E501
    page.write("<h3>Correlation Matrix</h3>")
    page.write(make_heatmap_html(cont_cat_corr_matrix))
    page.write('<h3>"Brute Force" Table</h3>')
    page.write(
        cont_cat_diff.to_html(escape=False, index=False, justify="center")
    )  # noqa: E501
    page.write("<h2>Categorical/Categorical Predictor Pairs</h2>")
    page.write("<h3>Correlation Table</h3>")
    page.write(
        cat_cat_df.to_html(escape=False, index=False, justify="center")
    )  # noqa: E501
    page.write("<h3>Correlation Matrix</h3>")
    page.write(make_heatmap_html(cat_cat_corr_matrix))
    page.write('<h3>"Brute Force" Table</h3>')
    page.write(
        cat_cat_diff.to_html(escape=False, index=False, justify="center")
    )  # noqa: E501

    footer = """<p>BDA696 Midterm, Fall 2021, by Tatiana Chavez</p>
    </body>
    </html>
    """
    page.write(footer)

    webbrowser.open(f"assignment5_analysis_{tag}.html", new=2)
    return


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


# linear regression model
def linear_regression(df, predictor, response):
    predictor_for_model = statsmodels.api.add_constant(df[predictor])
    linear_regression_model = statsmodels.api.OLS(
        df[response], predictor_for_model
    )  # noqa: E501
    linear_regression_model_fitted = linear_regression_model.fit()

    # Get the stats
    t_value = round(linear_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

    filename = f"plots/{predictor}_{response}_linear_regression.html"

    # Plot the figure
    fig = px.scatter(
        data_frame=df,
        x=predictor,
        y=response,
        trendline="ols",
        title=f"(t-value={t_value}) (p-value={p_value})",
    )

    fig.write_html(
        file=filename,
        include_plotlyjs="cdn",
    )

    return filename


def cont_cat_graph(df, predictor, response):
    filename1 = f"plots/{predictor}_{response}_dist_plot.html"
    filename2 = f"plots/{predictor}_{response}_violin_plot.html"
    group_labels = df[response].unique()
    hist_data = []

    for label in group_labels:
        sub_df = df[df[response] == label]
        group = sub_df[predictor]
        hist_data.append(group)

    # Create distribution plot
    fig_1 = px.histogram(df, x=predictor, color=response, marginal="rug")
    fig_1.update_layout(
        title=f"{predictor} by {response}",
        xaxis_title=predictor,
        yaxis_title=response,
    )

    fig_1.write_html(
        file=filename1,
        include_plotlyjs="cdn",
    )

    fig_2 = px.violin(df, y=predictor, color=response, box=True)

    fig_2.update_layout(
        title=f"{response} by {predictor} ",
        xaxis_title=predictor,
        yaxis_title=response,
    )

    fig_2.write_html(
        file=filename2,
        include_plotlyjs="cdn",
    )

    return filename1, filename2


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


def cat_cont_corr_ratio(categories, values):
    """
    Correlation Ratio: https://en.wikipedia.org/wiki/Correlation_ratio
    SOURCE:
    1.) https://towardsdatascience.com/
    the-search-for-categorical-correlation-a1cf7f1888c9
    :param categories: Numpy array of categories
    :param values: Numpy array of values
    :return: correlation
    """
    f_cat, _ = pd.factorize(categories)
    cat_num = np.max(f_cat) + 1
    y_avg_array = np.zeros(cat_num)
    n_array = np.zeros(cat_num)
    for i in range(0, cat_num):
        cat_measures = values[np.argwhere(f_cat == i).flatten()]
        n_array[i] = len(cat_measures)
        y_avg_array[i] = np.average(cat_measures)
    y_total_avg = np.sum(np.multiply(y_avg_array, n_array)) / np.sum(n_array)
    numerator = np.sum(
        np.multiply(
            n_array, np.power(np.subtract(y_avg_array, y_total_avg), 2)
        )  # noqa: E501
    )
    denominator = np.sum(np.power(np.subtract(values, y_total_avg), 2))
    if numerator == 0:
        eta = 0.0
    else:
        eta = np.sqrt(numerator / denominator)
    return eta


def make_heatmap_html(matrix: pd.DataFrame):
    fig = go.Figure(
        data=go.Heatmap(
            x=matrix.columns,
            y=matrix.index,
            z=matrix.values,
            zmin=0,
            zmax=1,
            colorscale="curl",
        )
    )
    matrix_html = fig.to_html()
    return matrix_html


def cont_cont_dwm(
    df: pd.DataFrame, pred1: str, pred2: str, response: str
):  # noqa: E501
    pred1_edges = np.histogram_bin_edges(df[pred1], bins=10)
    pred2_edges = np.histogram_bin_edges(df[pred2], bins=10)

    bin_mean, pred1_edges, pred2_edges, b = stats.binned_statistic_2d(
        df[pred1],
        df[pred2],
        df[response],
        statistic="mean",
        bins=(pred1_edges, pred2_edges),
    )

    bin_count, pred1_edges, pred2_edges, b = stats.binned_statistic_2d(
        df[pred1],
        df[pred2],
        df[response],
        statistic="count",
        bins=(pred1_edges, pred2_edges),
    )

    pop_mean = df[response].mean()

    # using later for graphs and also for calculations
    pred1_centers = (pred1_edges[:-1] + pred1_edges[1:]) / 2
    pred2_centers = (pred2_edges[:-1] + pred2_edges[1:]) / 2
    pred1_centers = pred1_centers.tolist()
    pred2_centers = pred2_centers.tolist()

    # calculating diff w/ mean (not weighted & weighted)
    diff = bin_mean - pop_mean
    sq_diff = np.square(diff)
    sum_sq_diff = np.nansum(sq_diff)
    total_bin = bin_mean.size
    msd = sum_sq_diff / total_bin
    pop_prop = bin_count / len(df[response])
    w_sq_diff = pop_prop * sq_diff
    wmsd = np.nansum(w_sq_diff)

    pop_prop = pop_prop.round(3)
    # note: plotly defines x opposite of binned stats so
    # I have swapped pred 1 & 2 so now pred2 is x ....
    fig = ff.create_annotated_heatmap(
        x=pred2_centers,
        y=pred1_centers,
        z=bin_mean,
        annotation_text=bin_mean.round(3),
        colorscale="curl",
        showscale=True,
        customdata=pop_prop,
        hovertemplate="<b>x</b>: %{x}<br>"
        + "<b>y</b>: %{y}<br>"
        + "<b>z</b>: %{z}"
        + "<extra>Population<br>Proportion: %{customdata}</extra>",
        hoverongaps=False,
        zmid=pop_mean,
        zmin=df[response].min(),
        zmax=df[response].max(),
    )

    fig.update_layout(
        title_text=f"{pred1} & {pred2} Bin Averages of Response",
        xaxis=dict(tickmode="array", tickvals=np.around(pred2_edges, 3)),
        yaxis=dict(tickmode="array", tickvals=np.around(pred1_edges, 3)),
    )
    filename1 = f"plots/{pred1}_{pred2}_diff_of_mean_resp_bin.html"

    fig.write_html(
        file=filename1,
        include_plotlyjs="cdn",
    )

    # residual plot
    fig_2 = ff.create_annotated_heatmap(
        x=pred2_centers,
        y=pred1_centers,
        z=diff,
        colorscale="curl",
        customdata=pop_prop,
        hovertemplate="<b>x</b>: %{x}<br>"
        + "<b>y</b>: %{y}<br>"
        + "<b>z</b>: %{z}"
        + "<extra>Population<br>Proportion: %{customdata}</extra>",
        annotation_text=diff.round(3),
        hoverongaps=False,
        showscale=True,
        zmid=pop_mean,
        zmin=df[response].min(),
        zmax=df[response].max(),
    )

    fig_2.update_layout(
        title_text=f"{pred1} & {pred2} Bin Average Residual",
        xaxis=dict(tickmode="array", tickvals=np.around(pred2_edges, 3)),
        yaxis=dict(tickmode="array", tickvals=np.around(pred1_edges, 3)),
    )
    filename2 = f"plots/{pred1}_{pred2}_dwm_of_resp_residual.html"

    fig_2.write_html(
        file=filename2,
        include_plotlyjs="cdn",
    )

    return filename1, filename2, round(msd, 3), round(wmsd, 3)


# predictor 1 is continous predictor 2 is categorical
def cont_cat_dwm(df: pd.DataFrame, pred1: str, pred2: str, response: str):
    pred1_edges = np.histogram_bin_edges(df[pred1], bins="sturges")
    # getting values for each category
    categories = df[pred2].unique()
    label_encoder = LabelEncoder()
    int_encoded = label_encoder.fit_transform(categories)

    # using later for graphs and also for calculations
    pred1_centers = (pred1_edges[:-1] + pred1_edges[1:]) / 2
    pred1_centers = pred1_centers.tolist()
    all_bin = pd.DataFrame(index=int_encoded, columns=pred1_centers)
    bin_count = pd.DataFrame(index=int_encoded, columns=pred1_centers)
    bin_mean = pd.DataFrame(index=int_encoded, columns=pred1_centers)

    for x in range(len(categories)):
        for i in range(len(pred1_edges) - 1):
            temp_bin = []
            temp_bin = df[
                (df[pred2] == categories[x])
                & (df[pred1] >= pred1_edges[i])
                & (df[pred1] < pred1_edges[i + 1])
            ][response]
            all_bin.at[int_encoded[x], pred1_centers[i]] = temp_bin
            bin_count.loc[int_encoded[x], pred1_centers[i]] = len(temp_bin)
            bin_mean.loc[int_encoded[x], pred1_centers[i]] = np.mean(temp_bin)

    pop_mean = df[response].mean()

    # calculating diff w/ mean (not weighted & weighted)
    diff = bin_mean - pop_mean
    sq_diff = np.square(diff)
    sum_sq_diff = np.nansum(sq_diff)
    total_bin = len(pred1_centers) * len(categories)
    msd = sum_sq_diff / total_bin
    pop_prop = bin_count / len(df[response])
    w_sq_diff = pop_prop * sq_diff
    wmsd = np.nansum(w_sq_diff)

    pop_prop = pop_prop.astype("float").round(3)
    round_bin_mean = bin_mean.astype("float").round(3)
    fig = ff.create_annotated_heatmap(
        x=pred1_centers,
        y=categories.tolist(),
        z=bin_mean.values.tolist(),
        annotation_text=round_bin_mean.values.tolist(),
        colorscale="curl",
        showscale=True,
        customdata=pop_prop,
        hovertemplate="<b>x</b>: %{x}<br>"
        + "<b>y</b>: %{y}<br>"
        + "<b>z</b>: %{z}"
        + "<extra>Population<br>Proportion: %{customdata}</extra>",
        hoverongaps=False,
        zmid=pop_mean,
        zmin=df[response].min(),
        zmax=df[response].max(),
    )

    fig.update_layout(
        title_text=f"{pred1} & {pred2} Bin Averages of Response",
        xaxis=dict(tickmode="array", tickvals=np.around(pred1_edges, 3)),
    )
    filename1 = f"plots/{pred1}_{pred2}_diff_of_mean_resp_bin.html"

    fig.write_html(
        file=filename1,
        include_plotlyjs="cdn",
    )

    # residual plot
    round_diff = diff.astype("float").round(3)
    fig_2 = ff.create_annotated_heatmap(
        x=pred1_centers,
        y=categories.tolist(),
        z=diff.values.tolist(),
        colorscale="curl",
        customdata=pop_prop,
        hovertemplate="<b>x</b>: %{x}<br>"
        + "<b>y</b>: %{y}<br>"
        + "<b>z</b>: %{z}"
        + "<extra>Population<br>Proportion: %{customdata}</extra>",
        annotation_text=round_diff.values.tolist(),
        hoverongaps=False,
        showscale=True,
        zmid=pop_mean,
        zmin=df[response].min(),
        zmax=df[response].max(),
    )

    fig_2.update_layout(
        title_text=f"{pred1} & {pred2} Bin Average",
        xaxis=dict(tickmode="array", tickvals=np.around(pred1_edges, 3)),
    )
    filename2 = f"plots/{pred1}_{pred2}_dwm_of_resp_residual.html"

    fig_2.write_html(
        file=filename2,
        include_plotlyjs="cdn",
    )

    return filename1, filename2, round(msd, 3), round(wmsd, 3)


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


def make_html_link(plot_col: pd.Series):
    # regex for making link text
    regex = ".+/([^/]+).html$"
    for x in range(len(plot_col)):
        text = re.findall(regex, plot_col[x])
        link_html = f'<a target="_blank" href="{plot_col[x]}">{text[0]}</a>'
        plot_col[x] = link_html
    return plot_col


def logistic_regression(pandas_df, predictor, response):
    predictor_for_model = statsmodels.api.add_constant(pandas_df[predictor])
    logistic_regression_model = statsmodels.api.Logit(
        pandas_df[response], predictor_for_model
    )
    logistic_regression_model_fitted = logistic_regression_model.fit()

    # Get the stats
    t_value = round(logistic_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(logistic_regression_model_fitted.pvalues[1])
    return p_value, t_value


def cont_dwm(pandas_df, predictor, response):
    mean, edges, bin_number = stats.binned_statistic(
        pandas_df[predictor], pandas_df[response], statistic="mean", bins=10
    )
    count, edges, bin_number = stats.binned_statistic(
        pandas_df[predictor], pandas_df[response], statistic="count", bins=10
    )
    pop_mean = np.mean(pandas_df[response])
    edge_centers = (edges[:-1] + edges[1:]) / 2
    mean_diff = mean - pop_mean
    mdsq = mean_diff ** 2
    pop_prop = count / len(pandas_df[response])
    wmdsq = pop_prop * mdsq
    msd = np.nansum(mdsq) / 10
    wmsd = np.sum(wmdsq)

    fig = px.scatter(
        x=edge_centers,
        y=mean_diff,
        color=px.Constant("$\u03BC_{i}$ - $\u03BC_{pop}$"),
        labels=dict(x="Predictor Bin", y="response"),
    )
    fig.add_bar(x=edge_centers, y=count, name="Population")
    fig.add_shape(
        type="line",
        x0=edges[0],
        y0=pop_mean,
        x1=edges[-1],
        y1=pop_mean,
        name="$\u03BC_{pop}$",
    )

    filename = f"plots/{predictor}_{response}_dwm.html"

    fig.write_html(
        file=filename,
        include_plotlyjs="cdn",
    )

    return filename, msd, wmsd


def cat_dwm(pandas_df, predictor, response):
    categories = pandas_df[predictor].unique().astype(str)
    categories = sorted(categories)
    mini_df = pandas_df[[predictor, response]]
    mean = mini_df.groupby([predictor]).mean()
    count = mini_df.groupby([predictor]).count()
    pop_mean = np.mean(pandas_df[response])
    mean_diff = mean.values - pop_mean
    mdsq = mean_diff ** 2
    pop_prop = count.values / len(pandas_df[response])
    wmdsq = pop_prop * mdsq
    msd = np.nansum(mdsq) / 10
    wmsd = np.nansum(wmdsq)

    fig = px.scatter(
        x=categories,
        y=mean_diff.flatten(),
        color=px.Constant("$\u03BC_{i}$ - $\u03BC_{pop}$"),
        labels=dict(x="Predictor Bin", y="response"),
    )
    fig.add_bar(x=categories, y=count.values, name="Population")
    fig.add_shape(
        type="line",
        x0=categories[0],
        y0=pop_mean,
        x1=categories[1],
        y1=pop_mean,
        name="$\u03BC_{pop}$",
    )

    filename = f"plots/{predictor}_{response}_dwm.html"

    fig.write_html(
        file=filename,
        include_plotlyjs="cdn",
    )

    return filename, msd, wmsd


def rf_importance(df, predictors, response):
    df_X = df[predictors]
    df_y = df[response]
    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(df_X, df_y)
    importances = clf.feature_importances_
    return importances.tolist()

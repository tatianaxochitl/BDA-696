import itertools
import os
import random
import re
import string
import webbrowser

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import (
    is_bool_dtype,
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_string_dtype,
)
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import RandomForestClassifier

from cat_cat import cat_cat_dwm, cat_correlation, heat_mx
from cont_cat import (
    cat_cont_corr_ratio,
    cont_cat_dwm,
    cont_cat_graph,
    logistic_regression,
)
from cont_cont import cont_cont_dwm, linear_regression


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

    if len(cont_list) != 0:
        cont_cont_df["Linear Regression Plot"] = make_html_link(
            cont_cont_df["Linear Regression Plot"]
        )
        cont_cont_diff["Bin Plot"] = make_html_link(cont_cont_diff["Bin Plot"])
        cont_cont_diff["Residual Plot"] = make_html_link(
            cont_cont_diff["Residual Plot"]
        )  # noqa: E80

    if len(cont_list) != 0 and len(cat_list) != 0:
        cont_cat_df["Violin Plot"] = make_html_link(cont_cat_df["Violin Plot"])
        cont_cat_df["Distribution Plot"] = make_html_link(
            cont_cat_df["Distribution Plot"]
        )  # noqa: E80
        cont_cat_diff["Bin Plot"] = make_html_link(cont_cat_diff["Bin Plot"])
        cont_cat_diff["Residual Plot"] = make_html_link(
            cont_cat_diff["Residual Plot"]
        )  # noqa: E80

    if len(cat_list) != 0:
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
    """
    page.write(header)
    page.write("<h2>Predictor Ranking</h2>")
    page.write(
        all_pred_df.to_html(escape=False, index=False, justify="center")
    )  # noqa: E501
    if len(cont_list) != 0:
        page.write("<h2>Continous/Continous Predictor Pairs</h2>")
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

    if len(cont_list) != 0 and len(cat_list) != 0:
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

    if len(cat_list) != 0:
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
    pop_mean_list = [pop_mean] * 10

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=edge_centers,
            y=mean_diff,
            name="$\\mu_{i}$ - $\\mu_{pop}$",
            mode="lines+markers",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(x=edge_centers, y=count, name="Population"), secondary_y=True
    )  # noqa: E501
    fig.add_trace(
        go.Scatter(
            y=pop_mean_list,
            x=edge_centers,
            mode="lines",
            name="$\\mu_{pop}$",
        )
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
    msd = np.nansum(mdsq) / len(categories)
    wmsd = np.nansum(wmdsq)

    pop_mean_list = [pop_mean] * len(categories)

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=categories,
            y=mean_diff.flatten(),
            name="$\\mu_{i}$ - $\\mu_{pop}$",
            mode="lines+markers",
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Bar(x=categories, y=count.values, name="Population"),
        secondary_y=True,  # noqa: E501
    )
    fig.add_trace(
        go.Scatter(
            y=pop_mean_list,
            x=categories,
            mode="lines",
            name="$\\mu_{pop}$",
        )
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


def make_html_link(plot_col: pd.Series):
    # regex for making link text
    regex = ".+/([^/]+).html$"
    for x in range(len(plot_col)):
        text = re.findall(regex, plot_col[x])
        link_html = (
            f'<a target="_blank" href="{plot_col[x]}">{text[0]}</a>'  # noqa: E501
        )
        plot_col[x] = link_html
    return plot_col

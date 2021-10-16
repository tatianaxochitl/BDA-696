import os
import webbrowser

import numpy as np  # noqa: F401
import pandas as pd  # noqa: F401
import plotly.figure_factory as ff
import plotly.graph_objects as go
import statsmodels.api
from plotly import express as px
from plotly.subplots import make_subplots
from sklearn import datasets  # noqa: F401
from sklearn.metrics import confusion_matrix


def process_dataframe(pandas_df, predictor_column, response_column):
    # look at response columns
    response_types = []

    for response in response_column:
        if pandas_df[response].dytpe == "bool":
            response_types.append("boolean")

        elif pandas_df[response].is_numeric_dtype:
            response_types.append("continous")

        else:
            raise ValueError(
                "Value in response column is neither boolean nor continous."
            )

    # look at predictor colums
    predictor_types = []

    for predictor in predictor_column:
        if pandas_df[predictor].is_numeric_dtype:
            predictor_types.append("continuous")
        elif pandas_df[predictor].is_string_dtype:
            predictor_types.append("categorical")
        else:
            raise ValueError(
                "Value in response column is neither boolean nor continous."
            )

    # create html file
    page = open(f"{pandas_df}_analysis.html", "w")
    header = f"""<!DOCTYPE html>
    <html>
    <head>
        <link rel="stylesheet" type="text/css" href="tyle.css">
        <title>{pandas_df} Analysis</title>
    </head>
    <body>
    <h1>{pandas_df} Analysis</h1>
    <h2>Predictor and Response Columns</h2>
    <div class="div-table">
        <div class="div-table-row">
            <div class="div-table-column">Predictor Columns</div>
            <div class="div-table-column">{predictor_column}</div>
        </div>
        <div class="div-table-row">
            <div class="div-table-column">Response Columns</div>
            <div class="div-table-column">{response_column}</div>
        </div>
    </div>
    """
    page.write(header)

    # for saving plotly files
    if not os.path.exists("plots"):
        os.mkdir("plots")

    table = create_table(
        pandas_df,
        predictor_column,
        predictor_types,
        response_column,
        response_types,  # noqa: E501
    )

    page.write(table)

    footer = """<p>BDA696 Assignment 4, Fall 2021, by Tatiana Chavez</p>
    </body>
    </html>
    """
    page.write(footer)

    webbrowser.open(f"{pandas_df}_analysis.html", new=2)

    return


def create_table(df, p_column, p_types, r_column, r_types):
    html_div_table = """<h2>Plots</h2>
    <div class="div-table">
    """
    for pt, pc in p_types, p_column:
        html_div_table_row = f"""  <div class="div-table-row">
                <div class="div-table-column">{pc}</div>
        """
        graph_filename = ""
        if pt == "categorical":
            graph_filename = cat_graph(df, pc)
            html_div_table_column = f'      <div class="div-table-column"><a href="plots/{graph_filename}">{pc} Plot</a></div>'  # noqa: E501
            html_div_table_row = "\n".join(
                [html_div_table_row, html_div_table_column]
            )  # noqa: E501
            for rt, rc in r_types, r_column:
                if rt == "boolean":
                    graph_filename = cat_bool_graph(df, pc, rc)

                elif rt == "continous":
                    graph_filename = cat_cont_graph(df, pc, rc)

                else:
                    raise ValueError("Invalid response column type")
            html_div_table_column = f'      <div class="div-table-column"><a href="plots/{graph_filename}">{pc} by {rc} Plot</a></div>'  # noqa: E501
            html_div_table_row = "\n".join(
                [html_div_table_row, html_div_table_column]
            )  # noqa: E501
        elif pt == "continuous":
            for rt, rc in r_types, r_column:
                if rt == "boolean":
                    graph_filename = cont_bool_graph(df, pc, rc)

                elif rt == "continous":
                    graph_filename = cont_cont_graph(df, pc, rc)

                else:
                    raise ValueError("Invalid response column type")
            html_div_table_column = f'      <div class="div-table-column"><a href="plots/{graph_filename}">{pc} vs {rc} Plot</a></div>'  # noqa: E501
            html_div_table_row = "\n".join(
                [html_div_table_row, html_div_table_column]
            )  # noqa: E501

        else:
            raise ValueError("Invalid predictor column type")

        "\n".join([html_div_table, html_div_table_row, "    </div>"])

    "\n".join([html_div_table, "</div>"])
    return html_div_table


# make graphs just looking at the categorical predictor
def cat_graph(pandas_df, predictor):
    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[
            [
                {"type": "xy"},
                {"type": "domain"},
            ]
        ],
    )
    fig.add_trace(
        go.Bar(
            pandas_df[predictor],
            x=pandas_df[predictor].value_counts().index,
            y=pandas_df[predictor].value_counts(),
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Pie(
            pandas_df[predictor],
            values=pandas_df[predictor].value_counts(),
            names=pandas_df[predictor].value_counts().index,
        ),
        row=1,
        col=2,
    )

    fig.write_image(f"plots/{pandas_df}_{predictor}_cat_graph.png")

    return f"{pandas_df}_{predictor}_cat_graph.png"


def cat_bool_graph(pandas_df, predictor, response):
    x = pandas_df[predictor]
    y = pandas_df[response]

    x_2 = [1 if abs(x_) > 1.5 else 0 for x_ in x]
    y_2 = [1 if abs(y_) > 1.5 else 0 for y_ in y]

    conf_matrix = confusion_matrix(x_2, y_2)

    fig = go.Figure(
        data=go.Heatmap(z=conf_matrix, zmin=0, zmax=conf_matrix.max())
    )  # noqa: E501
    fig.update_layout(
        title=f"Categorical Predictor by Categorical Response",
        xaxis_title=f"{predictor}",
        yaxis_title=f"{response}",
    )

    fig.write_image(
        f"plots/{pandas_df}_{predictor}_{response}_confusion_matrix.png"
    )  # noqa: E501

    return f"{pandas_df}_{predictor}_{response}_confusion_matrix.png"


def cat_cont_graph(pandas_df, predictor, response):
    group_labels = [pandas_df[predictor].unique()]

    hist_data = []

    for predictor_name in group_labels:
        hist_data.append(
            pandas_df[pandas_df[predictor] == predictor_name][response]
        )  # noqa: E501

        # Create distribution plot with custom bin_size
        fig = ff.create_distplot(hist_data, group_labels)
        fig.update_layout(
            title=f"{predictor} by {response}",
            xaxis_title=response,
            yaxis_title="Distribution",
        )
        fig.write_image(
            file=f"plots/{predictor}_{response}_dist_plot.html",
            include_plotlyjs="cdn",
        )
    return f"plots/{predictor}_{response}_dist_plot.html"


def cont_bool_graph(pandas_df, predictor, response):
    return


def cont_cont_graph(predictor_name, predictor_column, response_name, response):
    return


def linear_regression(pandas_df, predictor, response):
    predictor_for_model = statsmodels.api.add_constant(pandas_df[predictor])
    linear_regression_model = statsmodels.api.OLS(
        pandas_df[response], predictor_for_model
    )
    linear_regression_model_fitted = linear_regression_model.fit()
    print(f"Variable: {predictor}")
    print(linear_regression_model_fitted.summary())
    # Get the stats
    t_value = round(linear_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

    # Plot the figure
    fig = px.scatter(
        x=pandas_df[predictor], y=pandas_df[response], trendline="ols"
    )  # noqa: E501
    fig.update_layout(
        title=f"Variable: {predictor}: (t-value={t_value}) (p-value={p_value})",  # noqa: E501
        xaxis_title=f"Variable: {predictor}",
        yaxis_title="y",
    )

    fig.write_image(
        f"plots/{pandas_df}_{predictor}_{response}_linear_regression.png"
    )  # noqa: E501

    return f"{pandas_df}_{predictor}_{response}_linear_regression.png"

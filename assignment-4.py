import os
import webbrowser

import numpy as np  # noqa: F401
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objects as go
import statsmodels.api
from plotly import express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix

# using these to make code shorter also so i don't have to write as much
ROW_OPEN = '<div class="div-table-row">'
COL_OPEN = '<div class="div-table-column">'

# shouldnt make this global but I will need to rework the logic later
CONT_PRED = pd.DataFrame(
    columns=["Predictor", "t-score", "p-score", "RFC Importance"]
)  # noqa: E501


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
        <link rel="stylesheet" type="text/css" href="style.css">
        <title>{pandas_df} Analysis</title>
    </head>
    <body>
    <h1>{pandas_df} Analysis</h1>
    <h2>Predictor and Response Columns</h2>
    <div class="div-table">
        {ROW_OPEN}
            {COL_OPEN}Predictor Columns</div>
            {COL_OPEN}{predictor_column}</div>
        </div>
        {ROW_OPEN}
            {COL_OPEN}Response Columns</div>
            {COL_OPEN}{response_column}</div>
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
    tbl = """<h2>Plots</h2>
    <div class="div-table">
    """

    for pt, pc in p_types, p_column:
        tbl_row = f"""  {ROW_OPEN}
                {COL_OPEN}{pc}</div>
        """
        if pt == "categorical":
            for rt, rc in r_types, r_column:
                if rt == "boolean":
                    tbl_column = conf_mx(df, pc, rc)
                elif rt == "continous":
                    tbl_column = cat_pred_cont_resp_graph(df, pc, rc)

                else:
                    raise ValueError("Invalid response column type")
            tbl_row = "\n".join([tbl_row, tbl_column])  # noqa: E501
        elif pt == "continuous":
            CONT_PRED.loc[-1] = [pc, "", "", "", ""]
            for rt, rc in r_types, r_column:
                if rt == "boolean":
                    tbl_column = logistic_regression(df, pc, rc)

                elif rt == "continous":
                    tbl_column = linear_regression(df, pc, rc)

                else:
                    raise ValueError("Invalid response column type")

            tbl_row = "\n".join([tbl_row, tbl_column])

        else:
            raise ValueError("Invalid predictor column type")

        "\n".join([tbl, tbl_row, "    </div>"])

    "\n".join([tbl, "</div>"])

    # Random Forest Classifier Ranking
    df_X = df[CONT_PRED["Predictor"]]
    df_y = df[r_column]

    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(df_X, df_y)
    importances = clf.feature_importances_
    CONT_PRED["RFC Importance"] = importances.tolist()

    return tbl


def conf_mx(pandas_df, predictor, response):
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

    fig.write_html(
        f"plots/{pandas_df}_{predictor}_{response}_confusion_matrix.html",
        include_plotlyjs="cdn",
    )
    filename = f"{pandas_df}_{predictor}_{response}_confusion_matrix.html"
    col = f'    {COL_OPEN}<a href="plots/{filename}">{predictor} vs {response} Confusion Matrix</a></div>'  # noqa: E501
    return col


def cat_pred_cont_resp_graph(pandas_df, predictor, response):
    group_labels = [pandas_df[predictor].unique()]

    hist_data = []

    for predictor_name in group_labels:
        hist_data.append(
            pandas_df[pandas_df[predictor] == predictor_name][response]
        )  # noqa: E501

    # Create distribution plot
    fig = ff.create_distplot(hist_data, group_labels)
    fig.update_layout(
        title=f"{predictor} by {response}",
        xaxis_title=response,
        yaxis_title="Distribution",
    )

    fig.write_html(
        file=f"plots/{predictor}_{response}_dist_plot.html",
        include_plotlyjs="cdn",
    )

    fig_2 = go.Figure()
    for curr_hist, curr_group in zip(hist_data, group_labels):
        fig_2.add_trace(
            go.Violin(
                x=curr_group,
                y=curr_hist,
                name=curr_group,
                box_visible=True,
                meanline_visible=True,
            )
        )

    fig_2.update_layout(
        title=f"{response}by {predictor} ",
        xaxis_title="Groupings",
        yaxis_title=response,
    )
    fig_2.show()
    fig_2.write_html(
        file=f"plots/{predictor}_{response}_violin_plot.html",
        include_plotlyjs="cdn",
    )

    file2 = f"plots/{predictor}_{response}_violin_plot.html"
    file1 = f"plots/{predictor}_{response}_dist_plot.html"

    both_plots = f"""        {COL_OPEN}<a href="{file1}">{predictor} by {response} Distribution Plot</a></div>
            {COL_OPEN}<a href={file2}">{predictor} by {response} Violin Plot</a></div>
    """  # noqa: E501
    return both_plots


def logistic_regression(pandas_df, predictor, response):
    predictor_for_model = statsmodels.api.add_constant(pandas_df[predictor])
    logistic_regression_model = statsmodels.api.Logit(
        pandas_df[response], predictor_for_model
    )
    logistic_regression_model_fitted = logistic_regression_model.fit()

    # Get the stats
    t_value = round(logistic_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(logistic_regression_model_fitted.pvalues[1])

    CONT_PRED[CONT_PRED["Predictor"] == predictor]["t-value"] = t_value
    CONT_PRED[CONT_PRED["Predictor"] == predictor]["p-value"] = p_value

    # Plot the figure
    fig = px.scatter(
        x=pandas_df[predictor],
        y=pandas_df[response],
        trendline="ols",
        trendline_options=dict(log_x=True),
    )  # noqa: E501
    fig.update_layout(
        title=f"Variable: {predictor}: (t-value={t_value}) (p-value={p_value})",  # noqa: E501
        xaxis_title=f"Variable: {predictor}",
        yaxis_title="y",
    )

    fig.write_html(
        file=f"plots/{pandas_df}_{predictor}_{response}_logistic_regression.html",  # noqa: E501
        include_plotlyjs="cdn",
    )  # noqa: E501
    filename = f"plots/{pandas_df}_{predictor}_{response}_logistic_regression.html"  # noqa: E501
    col = f'    {COL_OPEN}<a href="plots/{filename}">{predictor} vs {response} Logistic Regression Plot</a></div>'  # noqa: E501
    return col


def linear_regression(pandas_df, predictor, response):
    predictor_for_model = statsmodels.api.add_constant(pandas_df[predictor])
    linear_regression_model = statsmodels.api.OLS(
        pandas_df[response], predictor_for_model
    )
    linear_regression_model_fitted = linear_regression_model.fit()

    # Get the stats
    t_value = round(linear_regression_model_fitted.tvalues[1], 6)
    p_value = "{:.6e}".format(linear_regression_model_fitted.pvalues[1])

    CONT_PRED[CONT_PRED["Predictor"] == predictor]["t-value"] = t_value
    CONT_PRED[CONT_PRED["Predictor"] == predictor]["p-value"] = p_value

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
        file=f"plots/{pandas_df}_{predictor}_{response}_linear_regression.html",  # noqa: E501
        include_plotlyjs="cdn",
    )  # noqa: E501
    filename = (
        f"plots/{pandas_df}_{predictor}_{response}_linear_regression.html"  # noqa: E501
    )
    col = f'    {COL_OPEN}<a href="plots/{filename}">{predictor} vs {response} Linear Regression Plot</a></div>'  # noqa: E501
    return col

import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api
from plotly import figure_factory as ff
from scipy import stats


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

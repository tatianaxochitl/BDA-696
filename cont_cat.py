import numpy as np
import pandas as pd
import plotly.express as px
import statsmodels.api
from plotly import figure_factory as ff
from sklearn.preprocessing import LabelEncoder


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

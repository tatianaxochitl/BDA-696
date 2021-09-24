import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

header_list = [
    "sepal length in cm",
    "sepal width in cm",
    "petal length in cm",
    "petal width in cm",
    "class",
]


def get_stats(iris_data):
    # create statistics from the dataframe
    iris_means = iris_data.mean(axis=0)
    iris_min = iris_data.min(axis=0)
    iris_max = iris_data.max(axis=0)
    iris_quart = []
    for column_name in iris_data.columns[:-1]:
        temp_narray = iris_data[column_name].to_numpy()
        iris_quart.append(np.histogram_bin_edges(temp_narray, bins=4))
    print("-" * 100)
    print("Mean")
    print("-" * 100)
    print_stats(iris_means)
    print("-" * 100)
    print("Minimum")
    print("-" * 100)
    print_stats(iris_min)
    print("-" * 100)
    print("Maximum")
    print("-" * 100)
    print_stats(iris_max)
    print("-" * 100)
    print("Quartile")
    print("-" * 100)
    print_stats(iris_quart)


def print_stats(iris_stats):
    for x in range(4):
        print("{:<30s}{}".format(header_list[x] + ":", str(iris_stats[x])))
    return ()


def main():
    # importing data into dataframe with correct labels
    iris_df = pd.read_csv("iris.data", header=None, names=header_list)

    # create dfs for the different classes
    iris_setosa = iris_df[iris_df["class"] == "Iris-setosa"]
    iris_versicolor = iris_df[iris_df["class"] == "Iris-versicolor"]
    iris_virginica = iris_df[iris_df["class"] == "Iris-virginica"]

    # find and print results
    print("Stats Summary for Iris Data")
    print("-" * 100)
    print("Overall")
    get_stats(iris_df[:-1])
    print("-" * 100)
    print("Iris sesota")
    get_stats(iris_setosa[:-1])
    print("-" * 100)
    print("Iris Versicolor")
    get_stats(iris_versicolor[:-1])
    print("-" * 100)
    print("Iris Virginica")
    get_stats(iris_virginica[:-1])
    print("-" * 100)

    # plotly sections
    # create a scatter plot
    # tutorial: https://plotly.com/python/line-and-scatter/
    fig = px.scatter(
        iris_df,
        x="sepal width in cm",
        y="sepal length in cm",
        color="class",
        size="petal length in cm",
        hover_data=["petal width in cm"],
        title="Iris Scatter Plot",
    )
    fig.update_layout(font_family="Helvetica", font_size=18)
    fig.show()

    # scatterplot matrix
    # tutorial: https://plotly.com/python/splom/
    fig = px.scatter_matrix(
        iris_df,
        dimensions=[
            "sepal length in cm",
            "sepal width in cm",
            "petal length in cm",
            "petal width in cm",
        ],
        color="class",
        symbol="class",
        title="Scatter matrix of iris data set",
    )
    fig.update_traces(diagonal_visible=False)
    fig.update_layout(font_family="Helvetica", font_size=18)
    fig.show()

    # create a violin plot using
    # tutorial: https://plotly.com/python/violin/
    fig = go.Figure()
    fig.update_layout(title="Iris Boxplot")

    fig.add_trace(
        go.Violin(
            x=iris_df["class"],
            y=iris_df["sepal length in cm"],
            legendgroup="sepal length in cm",
            scalegroup="sepal length in cm",
            name="sepal length in cm",
            line_color="#404040",
        )
    )

    fig.add_trace(
        go.Violin(
            x=iris_df["class"],
            y=iris_df["sepal width in cm"],
            legendgroup="sepal width in cm",
            scalegroup="sepal width in cm",
            name="sepal width in cm",
            line_color="#2a6a6c",
        )
    )

    fig.add_trace(
        go.Violin(
            x=iris_df["class"],
            y=iris_df["petal length in cm"],
            legendgroup="petal length in cm",
            scalegroup="petal length in cm",
            name="petal length in cm",
            line_color="#f29724",
        )
    )

    fig.add_trace(
        go.Violin(
            x=iris_df["class"],
            y=iris_df["petal width in cm"],
            legendgroup="petal width in cm",
            scalegroup="petal width in cm",
            name="petal width in cm",
            line_color="#b90d49",
        )
    )

    fig.update_traces(box_visible=True, meanline_visible=True)
    fig.update_layout(violinmode="group", title="Iris Violin Plot")
    fig.show()

    # boxplot
    # https://plotly.com/python/box-plots/#grouped-horizontal-box-plot
    fig = px.box(
        iris_df,
        y=[
            "sepal length in cm",
            "sepal width in cm",
            "petal length in cm",
            "petal width in cm",
        ],
        color="class",
    )
    fig.update_traces(quartilemethod="exclusive")
    fig.update_layout(font_family="Helvetica", font_size=18)
    fig.show()

    # Marginal Distribution Plots
    # ref: https://plotly.com/python/marginal-plots/
    fig = px.scatter(
        iris_df,
        x="sepal length in cm",
        y="sepal width in cm",
        color="class",
        marginal_x="box",
        marginal_y="violin",
        title="Sepal Length vs Sepal Width",
    )
    fig.update_layout(font_family="Helvetica", font_size=18)
    fig.show()

    fig = px.scatter(
        iris_df,
        x="petal length in cm",
        y="petal width in cm",
        color="class",
        marginal_x="box",
        marginal_y="violin",
        title="Petal Length vs Petal Width",
    )
    fig.update_layout(font_family="Helvetica", font_size=18)
    fig.show()

    # Create Model
    # Preprocess data
    np.random.seed(0)

    # splitting class from rest of data
    X = iris_df[header_list[:-1]]
    y = iris_df[header_list[-1]]

    # making transformation pipeline for RFC
    # iris_rfc = Pipeline([
    #   ('scaler', StandardScaler()),
    #   ('rfc', RandomForestClassifier())
    # ])
    # iris_rfc.fit(X,y)
    # rfc_score = iris_rfc.score(X,y)
    # Note this is just for posterity

    # testing a lot of classifiers and comparing them
    # list from https://scikit-learn.org/stable/auto_examples/classification
    # /plot_classifier_comparison.html?highlight=classifier%20comparison
    # link was too long for one line ^

    classifier_list = [
        RandomForestClassifier(),
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]

    print("Classifier Scores")
    print("-" * 100)

    for z in classifier_list:
        temp_pipe = Pipeline([("scaler", StandardScaler()), ("classifier", z)])
        temp_pipe.fit(X, y)
        print("{:<65s}{}".format(str(z) + ": ", str(temp_pipe.score(X, y))))


if __name__ == "__main__":
    sys.exit(main())

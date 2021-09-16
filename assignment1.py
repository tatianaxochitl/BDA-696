import numpy as np
import pandas as pd
import plotly.express as px 
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# importing data into dataframe with correct labels
header_list = [
    "sepal length in cm",
    "sepal width in cm",
    "petal length in cm",
    "petal width in cm",
    "class",
]
iris_df = pd.read_csv("iris.data", header=None, names=header_list)

# create dfs for the different classes
iris_setosa = iris_df[iris_df["class"] == "Iris-setosa"]
iris_versicolor = iris_df[iris_df["class"] == "Iris-versicolor"]
iris_virginica = iris_df[iris_df["class"] == "Iris-virginica"]


def get_stats(iris_data):
    # create statistics from the dataframe
    iris_means = iris_data.mean(axis=0)
    iris_min = iris_data.min(axis=0)
    iris_max = iris_data.max(axis=0)
    iris_quart = []
    for column_name in header_list[:-1]:
        temp_narray = iris_data[column_name].to_numpy()
        iris_quart.append(np.histogram_bin_edges(temp_narray, bins=4))
    print("Mean")
    print_stats(iris_means)
    print("Minimum")
    print_stats(iris_min)
    print("Maximum")
    print_stats(iris_max)
    print("Quartile")
    print_stats(iris_quart)


def print_stats(iris_stats):
    for x in range(3):
        print(header_list[x] + ": " + str(iris_stats[x]))
    return ()

#find and print results
print("Stats Summary for Iris Data")
print("-------------------------------")
print("Overall")
get_stats(iris_df)
print("-------------------------------")
print("Iris sesota")
get_stats(iris_setosa)
print("-------------------------------")
print("Iris Versicolor")
get_stats(iris_versicolor)
print("-------------------------------")
print("Iris Virginica")
get_stats(iris_virginica)


#plotly sections
#creating a scatter plot using tutorial from: https://plotly.com/python/line-and-scatter/
fig = px.scatter(iris_df, x="sepal width in cm", y="sepal length in cm", color="class",
                 size='petal length in cm', hover_data=['petal width in cm'], title="Iris Scatter Plot")
fig.update_layout(
    font_family="Helvetica",
    font_size=18
)
fig.show()

#scatterplot matrix using tutorial from: https://plotly.com/python/splom/
fig = px.scatter_matrix(iris_df,
    dimensions=["sepal length in cm", "sepal width in cm", "petal length in cm", "petal width in cm"],
    color="class", symbol="class",
    title="Scatter matrix of iris data set")
fig.update_traces(diagonal_visible=False)
fig.update_layout(
    font_family="Helvetica",
    font_size=18
)
fig.show()

#create a violin plot using tutorial from: 
fig = go.Figure()

fig.add_trace(go.Violin(x=iris_df['class'],
                        y=iris_df['sepal length in cm'],
                        legendgroup='sepal length in cm', scalegroup='sepal length in cm', name='sepal length in cm',
                        line_color='#404040')
             )

fig.add_trace(go.Violin(x=iris_df['class'],
                        y=iris_df['sepal width in cm'],
                        legendgroup='sepal width in cm', scalegroup='sepal width in cm', name='sepal width in cm',
                        line_color='#2a6a6c')
             )

fig.add_trace(go.Violin(x=iris_df['class'],
                        y=iris_df['petal length in cm'],
                        legendgroup='petal length in cm', scalegroup='petal length in cm', name='petal length in cm',
                        line_color='#f29724')
             )

fig.add_trace(go.Violin(x=iris_df['class'],
                        y=iris_df['petal width in cm'],
                        legendgroup='petal width in cm', scalegroup='petal width in cm', name='petal width in cm',
                        line_color='#b90d49')
             )

fig.update_traces(box_visible=True, meanline_visible=True)
fig.update_layout(violinmode='group',title='Isis Violin Plot')
fig.show()

#boxplot from:https://plotly.com/python/box-plots/#grouped-horizontal-box-plot
fig = px.box(iris_df, y=['sepal length in cm', 'sepal width in cm', 'petal length in cm', 'petal width in cm'], color="class")
fig.update_traces(quartilemethod="exclusive") # or "inclusive", or "linear" by default
fig.update_layout(title='Isis Boxplot',font_family="Helvetica",font_size=18)
fig.show()

#Marginal Distribution Plots ref: https://plotly.com/python/marginal-plots/
fig = px.scatter(iris_df, x="sepal length in cm", y="sepal width in cm", color="class", marginal_x='box', marginal_y="violin", title ="Sepal Length vs Sepal Width")
fig.update_layout(
    font_family="Helvetica",
    font_size=18
)
fig.show()

fig = px.scatter(iris_df, x="petal length in cm", y="petal width in cm", color="class", 
                 marginal_x="box", marginal_y="violin", title="Petal Length vs Petal Width")
fig.update_layout(
    font_family="Helvetica",
    font_size=18
)
fig.show()



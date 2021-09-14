import numpy as np
import pandas as pd

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

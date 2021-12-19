import re
import sys

import pandas as pd
import sqlalchemy
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from midterm import process_dataframe


def main():
    db_user = "root"
    db_pass = "root"  # pragma: allowlist secret
    db_host = "localhost"
    db_database = "baseball"
    connect_string = f"mariadb+mariadbconnector://{db_user}:{db_pass}@{db_host}/{db_database}"  # pragma: allowlist secret # noqa: E501

    sql_engine = sqlalchemy.create_engine(connect_string)

    query = """
        SELECT *
            FROM predictive_table
    """

    baseball_df = pd.read_sql_query(query, sql_engine)

    baseball_df[
        ["home_team_id", "away_team_id", "home_pitcher", "away_pitcher"]
    ] = baseball_df[
        ["home_team_id", "away_team_id", "home_pitcher", "away_pitcher"]
    ].astype(
        str
    )
    baseball_df[["home_BA", "away_BA"]] = baseball_df[
        ["home_BA", "away_BA"]
    ].astype(  # noqa: E501
        float
    )
    baseball_df["result"] = baseball_df["result"].map({"TRUE": 1, "FALSE": 0})
    predictor_list = [
        "away_BA",
        "away_BABIP",
        "away_OBP",
        "away_K9",
        "away_BB9",
        "away_FIP",
        "away_xFIP",
        "away_ERA",
        "home_BA",
        "home_BABIP",
        "home_OBP",
        "home_K9",
        "home_BB9",
        "home_FIP",
        "home_xFIP",
        "home_ERA",
        "diff_BA",
        "diff_BABIP",
        "diff_OBP",
        "diff_K9",
        "diff_BB9",
        "diff_FIP",
        "diff_xFIP",
        "diff_ERA",
    ]

    diff_predictor_list = [
        "diff_BA",
        "diff_BABIP",
        "diff_OBP",
        "diff_K9",
        "diff_BB9",
        "diff_FIP",
        "diff_xFIP",
        "diff_ERA",
    ]

    for pred in diff_predictor_list:
        m = re.match(r"^diff(\w+)$", pred)
        baseball_df[pred] = baseball_df[f"home{m.group(1)}"].subtract(
            baseball_df[f"away{m.group(1)}"]
        )
        baseball_df[pred]

    process_dataframe(baseball_df, predictor_list, "result")  # noqa: E501

    # make cat into cont
    baseball_df[["home_pitcher", "away_pitcher"]] = baseball_df[
        ["home_pitcher", "away_pitcher"]
    ].astype(  # noqa: E501
        float
    )

    reduced_predictor_list = [
        "away_OBP",
        "home_OBP",
        "diff_ERA",
        "diff_FIP",
    ]

    # Modeling
    train, test = train_test_split(
        baseball_df, test_size=0.33, random_state=42
    )  # noqa: E501

    ttrain_x, ttrain_y = train[predictor_list], train["result"]
    ttest_x, ttest_y = test[predictor_list], test["result"]

    train_x, train_y = train[reduced_predictor_list], train["result"]
    test_x, test_y = test[reduced_predictor_list], test["result"]

    classifier_list = [
        RandomForestClassifier(),
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        DecisionTreeClassifier(max_depth=5),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        LogisticRegression(),
    ]

    print("Classifier Scores for all features")
    print("-" * 100)

    for z in classifier_list:
        temp_pipe = Pipeline([("scaler", StandardScaler()), ("classifier", z)])
        temp_pipe.fit(ttrain_x, ttrain_y)
        print(
            "{:<65s}{}".format(
                str(z) + ": ", str(temp_pipe.score(ttest_x, ttest_y))
            )  # noqa: E501
        )

    print("Classifier Scores top features")
    print("-" * 100)

    for z in classifier_list:
        temp_pipe = Pipeline([("scaler", StandardScaler()), ("classifier", z)])
        temp_pipe.fit(train_x, train_y)
        print(
            "{:<65s}{}".format(
                str(z) + ": ", str(temp_pipe.score(test_x, test_y))
            )  # noqa: E501
        )


if __name__ == "__main__":
    sys.exit(main())

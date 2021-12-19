import re
import sys

import pandas as pd
import plotly.graph_objects as go
import sqlalchemy
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
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

    baseball_df[["home_team_id", "away_team_id"]] = baseball_df[
        ["home_team_id", "away_team_id"]
    ].astype(str)
    baseball_df[["home_BA", "away_BA"]] = baseball_df[
        ["home_BA", "away_BA"]
    ].astype(  # noqa: E501
        float
    )
    baseball_df["result"] = baseball_df["result"].map({"TRUE": 1, "FALSE": 0})

    baseball_df = baseball_df.dropna()

    # predictor_list = [
    #     "away_team_id",
    #     "away_league",
    #     "away_division",
    #     "away_BA",
    #     "away_BABIP",
    #     "away_OBP",
    #     "away_K9",
    #     "away_BB9",
    #     "away_FIP",
    #     "away_xFIP",
    #     "away_ERA",
    #     "away_WHIP",
    #     "home_team_id",
    #     "home_league",
    #     "home_division",
    #     "home_BA",
    #     "home_BABIP",
    #     "home_OBP",
    #     "home_K9",
    #     "home_BB9",
    #     "home_FIP",
    #     "home_xFIP",
    #     "home_ERA",
    #     "home_WHIP",
    #     "diff_BA",
    #     "diff_BABIP",
    #     "diff_OBP",
    #     "diff_K9",
    #     "diff_BB9",
    #     "diff_FIP",
    #     "diff_xFIP",
    #     "diff_ERA",
    #     "diff_WHIP",
    #     "combo_league",
    #     "combo_division",
    #     "combo_team_id",
    # ]

    diff_predictor_list = [
        "diff_BA",
        "diff_BABIP",
        "diff_OBP",
        "diff_K9",
        "diff_BB9",
        "diff_FIP",
        "diff_xFIP",
        "diff_ERA",
        "diff_WHIP",
    ]

    for pred in diff_predictor_list:
        m = re.match(r"^diff(\w+)$", pred)
        baseball_df[pred] = baseball_df[f"home{m.group(1)}"].subtract(
            baseball_df[f"away{m.group(1)}"]
        )
        baseball_df[pred]

    combo_predictor_list = [
        "combo_league",
        "combo_division",
        "combo_team_id",
    ]
    for pred in combo_predictor_list:
        m = re.match(r"^combo(\w+)$", pred)
        baseball_df[pred] = (
            baseball_df[f"home{m.group(1)}"]
            + " vs "
            + baseball_df[f"away{m.group(1)}"]  # noqa: E501
        )
        baseball_df[pred]

    # I used this to initially look at the predictors
    # but commenting out cuz it takes a very long
    # time to run
    # process_dataframe(baseball_df, predictor_list, "result")  # noqa: E501

    reduced_predictor_list_1 = [
        "diff_xFIP",
        "diff_WHIP",
        "diff_K9",
        "diff_FIP",
        "diff_ERA",
        "diff_BB9",
        "away_OBP",
        "home_BA",
        "away_BABIP",
        "combo_league",
        "combo_division",
        "combo_team_id",
    ]

    process_dataframe(baseball_df, reduced_predictor_list_1, "result")  # noqa: E501

    # Modeling
    # One Hot Encoding for the categorical variables
    for pred in combo_predictor_list:
        # integer encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(baseball_df[pred])
        # binary encode
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        baseball_df[pred] = onehot_encoder.fit_transform(integer_encoded)

    sys.stdout = open("regression_testing.txt", "w")

    print("Original Score\n")
    print("-" * 100)
    test_models(baseball_df, reduced_predictor_list_1, "result")

    # First Round of Reduction
    print("diff_WHIP vs diff_ERA")
    print("-" * 100)
    print("Without WHIP")
    print("-" * 100)
    reduced_predictor_list_2 = reduced_predictor_list_1.copy()
    reduced_predictor_list_2.remove("diff_WHIP")
    test_models(baseball_df, reduced_predictor_list_2, "result")
    print("-" * 100)
    print("Without ERA")
    print("-" * 100)
    reduced_predictor_list_3 = reduced_predictor_list_1.copy()
    reduced_predictor_list_3.remove("diff_ERA")
    test_models(baseball_df, reduced_predictor_list_3, "result")
    print("-" * 100)
    print("away_OBP vs away_BABIP")
    print("-" * 100)
    print("Without OBP")
    print("-" * 100)
    reduced_predictor_list_4 = reduced_predictor_list_1.copy()
    reduced_predictor_list_4.remove("away_OBP")
    test_models(baseball_df, reduced_predictor_list_4, "result")
    print("-" * 100)
    print("Without BABIP")
    print("-" * 100)
    reduced_predictor_list_5 = reduced_predictor_list_1.copy()
    reduced_predictor_list_5.remove("away_BABIP")
    test_models(baseball_df, reduced_predictor_list_5, "result")
    print("-" * 100)
    print("diff_xFIP vs diff_K9")
    print("-" * 100)
    print("Without xFIP")
    print("-" * 100)
    reduced_predictor_list_6 = reduced_predictor_list_1.copy()
    reduced_predictor_list_6.remove("diff_xFIP")
    test_models(baseball_df, reduced_predictor_list_6, "result")
    print("-" * 100)
    print("Without K9")
    print("-" * 100)
    reduced_predictor_list_7 = reduced_predictor_list_1.copy()
    reduced_predictor_list_7.remove("diff_K9")
    test_models(baseball_df, reduced_predictor_list_7, "result")
    print("-" * 100)
    print("diff_xFIP vs diff_FIP")
    print("-" * 100)
    print("Without xFIP")
    print("-" * 100)
    test_models(baseball_df, reduced_predictor_list_6, "result")
    print("-" * 100)
    print("Without FIP")
    print("-" * 100)
    reduced_predictor_list_8 = reduced_predictor_list_1.copy()
    reduced_predictor_list_8.remove("diff_FIP")
    test_models(baseball_df, reduced_predictor_list_8, "result")
    print("-" * 100)

    # NOW we do some more elimination
    print("Remove FIP & WHIP")
    print("-" * 100)
    reduced_predictor_list_9 = reduced_predictor_list_8.copy()
    reduced_predictor_list_9.remove("diff_WHIP")
    test_models(baseball_df, reduced_predictor_list_9, "result")
    print("Remove FIP & BABIP")
    print("-" * 100)
    reduced_predictor_list_10 = reduced_predictor_list_8.copy()
    reduced_predictor_list_10.remove("away_BABIP")
    test_models(baseball_df, reduced_predictor_list_10, "result")
    print("-" * 100)

    # Even more elimination
    for predictor in reduced_predictor_list_9:
        print(f"Remove {predictor}")
        print("-" * 100)
        temp_predictor_list = reduced_predictor_list_9.copy()
        temp_predictor_list.remove(predictor)
        test_models(baseball_df, temp_predictor_list, "result")
        print("-" * 100)

    # One more time :-)
    reduced_predictor_list_9.remove("diff_BB9")
    for predictor in reduced_predictor_list_9:
        print(f"Remove {predictor}")
        print("-" * 100)
        temp_predictor_list = reduced_predictor_list_9.copy()
        temp_predictor_list.remove(predictor)
        test_models(baseball_df, temp_predictor_list, "result")
        print("-" * 100)

    # again
    reduced_predictor_list_9.remove("combo_league")
    for predictor in reduced_predictor_list_9:
        print(f"Remove {predictor}")
        print("-" * 100)
        temp_predictor_list = reduced_predictor_list_9.copy()
        temp_predictor_list.remove(predictor)
        test_models(baseball_df, temp_predictor_list, "result")
        print("-" * 100)

    # please be the last time
    reduced_predictor_list_9.remove("home_BA")
    for predictor in reduced_predictor_list_9:
        print(f"Remove {predictor}")
        print("-" * 100)
        temp_predictor_list = reduced_predictor_list_9.copy()
        temp_predictor_list.remove(predictor)
        test_models(baseball_df, temp_predictor_list, "result")
        print("-" * 100)

    reduced_predictor_list_9.remove("combo_team_id")
    for predictor in reduced_predictor_list_9:
        print(f"Remove {predictor}")
        print("-" * 100)
        temp_predictor_list = reduced_predictor_list_9.copy()
        temp_predictor_list.remove(predictor)
        test_models(baseball_df, temp_predictor_list, "result")
        print("-" * 100)

    print("67/33 Split:")
    print("-" * 100)
    test_models(baseball_df, reduced_predictor_list_9, "result")
    print("-" * 100)
    # Here I am with another Loop :-)
    test_sizes = [0.35, 0.30, 0.25, 20]
    splits = ["65/35", "70/30", "75/25", "80/20"]
    for size, split in zip(test_sizes, splits):
        print(f"{split} Split:")
        print("-" * 100)
        test_models(
            baseball_df, reduced_predictor_list_9, "result", test_split=size
        )  # noqa: E501
        print("-" * 100)

    # OK Last loop I think for the love of all that is holy
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

    train, test = train_test_split(
        baseball_df, test_size=0.33, random_state=42
    )  # noqa: E501

    train_x, train_y = train[reduced_predictor_list_9], train["result"]
    test_x, test_y = test[reduced_predictor_list_9], test["result"]

    fig = go.Figure()
    fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)

    print("Classifier Scores for all features")
    print("-" * 100)

    for z in classifier_list:
        print(f"{z} Scores :")
        print("-" * 100)
        temp_pipe = Pipeline([("scaler", StandardScaler()), ("classifier", z)])
        temp_pipe.fit(train_x, train_y)
        pred_y = temp_pipe.predict(test_x)
        print(
            "{:<65s}{}".format(
                "Precision Score: ", str(precision_score(test_y, pred_y))
            )
        )
        print(
            "{:<65s}{}".format(
                "Accuracy Score: ", str(accuracy_score(test_y, pred_y))
            )  # noqa: E501
        )
        print("{:<65s}{}".format("F1 Score: ", str(f1_score(test_y, pred_y))))
        print("-" * 100)
        fpr, tpr, _ = roc_curve(test_y, pred_y)
        auc_score = roc_auc_score(test_y, pred_y)

        name = f"{z} (AUC={auc_score:.2f})"
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode="lines"))

    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain="domain"),
        width=700,
        height=500,
    )

    # figure was made using
    # tutorial from here:
    # https://plotly.com/python/roc-and-pr-curves/

    fig.write_html(
        f"plots/AAAAA_CLASSIFIER_ROC_CURVES.html",
        include_plotlyjs="cdn",
    )

    # used A at beginning so i can find it easily lol

    # IM SO SORRY I SHOULD BE MAKING FUNCTIONS FOR THESE BUT ITS ALMOST 11PM :(

    fig = go.Figure()
    fig.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)

    print("Classifier Scores for all features")
    print("-" * 100)

    for z in test_sizes:
        print(f"{z} Scores :")
        print("-" * 100)
        train, test = train_test_split(
            baseball_df, test_size=z, random_state=42
        )  # noqa: E501

        train_x, train_y = train[reduced_predictor_list_9], train["result"]
        test_x, test_y = test[reduced_predictor_list_9], test["result"]

        temp_pipe = Pipeline(
            [("scaler", StandardScaler()), ("classifier", GaussianNB())]
        )
        temp_pipe.fit(train_x, train_y)
        pred_y = temp_pipe.predict(test_x)
        print(
            "{:<65s}{}".format(
                "Precision Score: ", str(precision_score(test_y, pred_y))
            )
        )
        print(
            "{:<65s}{}".format(
                "Accuracy Score: ", str(accuracy_score(test_y, pred_y))
            )  # noqa: E501
        )
        print("{:<65s}{}".format("F1 Score: ", str(f1_score(test_y, pred_y))))
        print("-" * 100)
        fpr, tpr, _ = roc_curve(test_y, pred_y)
        auc_score = roc_auc_score(test_y, pred_y)

        name = f"{z} (AUC={auc_score:.2f})"
        fig.add_trace(go.Scatter(x=fpr, y=tpr, name=name, mode="lines"))

    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        xaxis=dict(constrain="domain"),
        width=700,
        height=500,
    )

    fig.write_html(
        f"plots/AAAAA_SPLIT_ROC_CURVES.html",
        include_plotlyjs="cdn",
    )

    sys.stdout.close()


def test_models(df, predictor_list, response, test_split=0.033):
    train, test = train_test_split(df, test_size=test_split, random_state=42)

    train_x, train_y = train[predictor_list], train[response]
    test_x, test_y = test[predictor_list], test[response]

    temp_pipe = Pipeline(
        [("scaler", StandardScaler()), ("classifier", LogisticRegression())]
    )
    temp_pipe.fit(train_x, train_y)
    pred_y = temp_pipe.predict(test_x)
    print(
        "{:<65s}{}".format(
            "Precision Score: ", str(precision_score(test_y, pred_y))
        )  # noqa: E501
    )  # noqa: E501
    print(
        "{:<65s}{}".format(
            "Accuracy Score: ", str(accuracy_score(test_y, pred_y))
        )  # noqa: E501
    )  # noqa: E501
    print("{:<65s}{}".format("F1 Score: ", str(f1_score(test_y, pred_y))))


if __name__ == "__main__":
    sys.exit(main())

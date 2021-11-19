import sys

from pyspark.sql import SparkSession
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from midterm import process_dataframe


def main():
    # start spark session
    spark = SparkSession.builder.master("local[*]").getOrCreate()

    # config db information, edit if your sign in information is different
    database = "baseball"
    port = "3306"
    username = "root"
    pw = "root"  # pragma: allowlist secret

    baseball_table = (
        spark.read.format("jdbc")
        .options(
            url=f"jdbc:mysql://localhost:{port}/{database}?zeroDateTimeBehavior=CONVERT_TO_NULL",  # noqa E501
            driver="com.mysql.cj.jdbc.Driver",
            dbtable="predictive_table",
            user=username,
            password=pw,
        )
        .load()
    )

    baseball_df = baseball_table.toPandas()
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
    predictor_list = [  # noqa: F841
        "home_team_id",
        "away_team_id",
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
        "home_pitcher",
        "away_pitcher",
    ]
    reduced_predictor_list = [
        "away_BABIP",
        "away_OBP",
        "home_pitcher",
        "away_pitcher",
    ]

    process_dataframe(baseball_df, reduced_predictor_list, "result")  # noqa: E501

    # make cat into cont
    baseball_df[["home_pitcher", "away_pitcher"]] = baseball_df[
        ["home_pitcher", "away_pitcher"]
    ].astype(  # noqa: E501
        float
    )

    # Modeling
    train, test = train_test_split(
        baseball_df, test_size=0.33, random_state=42
    )  # noqa: F501

    train_x, train_y = train[reduced_predictor_list], train["result"]
    test_x, test_y = test[reduced_predictor_list], test["result"]

    # Logistic Regression
    logreg_clf = LogisticRegression()
    logreg_clf.fit(train_x, train_y)
    log_prediction = logreg_clf.predict(test_x)

    # SVC
    SVC_model = SVC()
    SVC_model.fit(train_x, train_y)
    SVC_prediction = SVC_model.predict(test_x)

    print(
        f"Logistic Regression Accuracy Score: {accuracy_score(log_prediction, test_y)}"  # noqa: E501
    )
    print(f"SVC Accuracy Score: {accuracy_score(SVC_prediction, test_y)}")
    # logistic regression performs slightly better


if __name__ == "__main__":
    sys.exit(main())

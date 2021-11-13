import sys

from pyspark.sql import SparkSession

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
        "home_team_id",
        "away_team_id",
        "away_BA",
        "away_BABIP",
        "away_OBP",
        "away_xFIP",
        "away_ERA",
        "home_BA",
        "home_BABIP",
        "home_OBP",
        "home_xFIP",
        "home_ERA",
        "home_pitcher",
        "away_pitcher",
    ]

    print(baseball_df.info(verbose=True))
    hmtl = process_dataframe(  # noqa: F841
        baseball_df, reduced_predictor_list, "result"
    )


if __name__ == "__main__":
    sys.exit(main())

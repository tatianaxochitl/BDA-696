import sys

from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession

from calculate_rolling_average_transform import CalculateRollingAverage


def main():
    # start spark session
    spark = SparkSession.builder.master("local[*]").getOrCreate()

    # config db information, edit if your sign in information is different
    database = "baseball"
    port = "3306"
    username = "root"
    pw = "root"  # pragma: allowlist secret

    game = (
        spark.read.format("jdbc")
        .options(
            url=f"jdbc:mysql://localhost:{port}/{database}",
            driver="com.mysql.cj.jdbc.Driver",
            dbtable="game",
            user=username,
            password=pw,
        )
        .load()
    )

    game.createOrReplaceTempView("game")
    game.persist(StorageLevel.DISK_ONLY)

    batter_counts = (
        spark.read.format("jdbc")
        .options(
            url=f"jdbc:mysql://localhost:{port}/{database}",
            driver="com.mysql.cj.jdbc.Driver",
            dbtable="batter_counts",
            user=username,
            password=pw,
        )
        .load()
    )

    batter_counts.createOrReplaceTempView("batter_counts")
    batter_counts.persist(StorageLevel.DISK_ONLY)

    select_statement = spark.sql(
        """SELECT
            bc.batter
            , g.game_id
            , CAST(g.local_date AS DATE)
            , bc.Hit
            , bc.atBat
            FROM batter_counts as bc
            JOIN game as g ON g.game_id = bc.game_id
            ORDER BY bc.batter, g.game_id"""
    )

    calculate_rolling_average = CalculateRollingAverage(
        batter="batter",
        gameID="game_id",
        localDate="local_date",
        hit="Hit",
        atBat="atBat",
        outputCol="rolling_100_day_avg",
    )

    pipeline = Pipeline(stages=[calculate_rolling_average])
    model = pipeline.fit(select_statement)
    select_statement = model.transform(select_statement)
    select_statement.show()


if __name__ == "__main__":
    sys.exit(main())

from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasOutputCol, Param, Params
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import col, round, sum, window
from pyspark.sql.window import Window


class CalculateRollingAverage(
    Transformer,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    date = Param(
        Params._dummy(),
        "date",
        "date to fill",
    )

    batter = Param(
        Params._dummy(),
        "batter",
        "batter to fill",
    )

    hit = Param(
        Params._dummy(),
        "hit",
        "hit to fill",
    )

    atBat = Param(
        Params._dummy(),
        "atBat",
        "atBat to fill",
    )

    @keyword_only
    def __init__(self, date=None, batter=None, hit=0, atBat=0, outputCol=None):
        super(CalculateRollingAverage, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)
        return

    @keyword_only
    def setParams(
        self,
        date=None,
        batter=None,
        hit=0,
        atBat=0,
        outputCol=None,
    ):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setLocalDate(self, date):
        return self._set(date=date)

    def getLocalDate(self):
        return self.getOrDefault(self.date)

    def setBatter(self, batter):
        return self._set(batter=batter)

    def getBatter(self):
        return self.getOrDefault(self.batter)

    def setGameID(self, gameID):
        return self._set(gameID=gameID)

    def getGameID(self):
        return self.getOrDefault(self.gameID)

    def setHit(self, hit):
        return self._set(hit=hit)

    def getHit(self):
        return self.getOrDefault(self.hit)

    def setAtBat(self, atBat):
        return self._set(atBat=atBat)

    def getAtBat(self):
        return self.getOrDefault(self.atBat)

    def _transform(self, dataset):
        date = self.getLocalDate()
        batter = self.getBatter()
        hit = self.getHit()
        at_bat = self.getAtBat()
        output_col = self.getOutputCol()

        # do the calculations to get the rolling average
        dates = window(
            col(date).cast("timestamp"), "1 day", startTime="-100 days"
        )  # noqa E501

        d = (
            Window.partitionBy(col(batter))
            .orderBy(col(dates))
            .rangeBetween(-100, 0)  # noqa E501
        )

        output_col = round(
            (sum(col(hit)).over(d) / sum(col(at_bat)).over(d)), 3
        )  # noqa E501

        dataset = dataset.withColumn("rolling_100_day_avg", output_col)

        return dataset

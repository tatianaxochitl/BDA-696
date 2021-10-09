from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasOutputCol, Param, Params
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import datediff, max, min, round, sum
from pyspark.sql.window import Window


class CalculateRollingAverage(
    Transformer,
    HasOutputCol,
    DefaultParamsReadable,
    DefaultParamsWritable,
):
    localDate = Param(
        Params._dummy(),
        "localDate",
        "localDate to fill",
    )

    batter = Param(
        Params._dummy(),
        "batter",
        "batter to fill",
    )

    gameID = Param(
        Params._dummy(),
        "gameID",
        "gameID to fill",
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
        hit=None,
        atBat=None,
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
        dates = datediff(max(date), min(date))

        d = Window.partitionBy(batter).orderBy(dates).rangeBetween(-100, 0)

        output_col = round((sum(hit).over(d) / sum(at_bat).over(d)), 3)

        dataset = dataset.withColumn("rolling_100_day_avg", output_col)

        return dataset

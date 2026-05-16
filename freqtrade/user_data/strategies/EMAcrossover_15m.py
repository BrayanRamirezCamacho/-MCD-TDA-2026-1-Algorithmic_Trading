from freqtrade.strategy import IStrategy
from pandas import DataFrame
import talib.abstract as ta


class EMAcrossover_15m(IStrategy):

    timeframe = "15m"
    startup_candle_count = 210

    minimal_roi = {
        "0":   0.03,   # sale inmediato si +3%
        "45":  0.02,   # sale a los 45 min si +2%
        "90":  0.01,   # sale a 90 min si +1%
        "180": 0.005,  # sale a 3h si +0.5%
    }

    stoploss = -0.06

    trailing_stop = True
    trailing_stop_positive = 0.015
    trailing_stop_positive_offset = 0.02
    trailing_only_offset_is_reached = True

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema9"]   = ta.EMA(dataframe, timeperiod=9)
        dataframe["ema21"]  = ta.EMA(dataframe, timeperiod=21)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe.loc[
            (
                (dataframe["ema9"] > dataframe["ema21"]) &
                (dataframe["ema9"].shift(1) <= dataframe["ema21"].shift(1)) &
                (dataframe["close"] > dataframe["ema200"]) &
                (dataframe["volume"] > 0)
            ),
            ["enter_long", "enter_tag"]
        ] = [1, "ema9_cross_21_above200"]

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Sin señal de salida explícita.
        # ROI y trailing_stop manejan todas las salidas.
        return dataframe

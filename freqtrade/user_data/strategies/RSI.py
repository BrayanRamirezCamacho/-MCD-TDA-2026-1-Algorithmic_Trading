from freqtrade.strategy import IStrategy
from freqtrade.strategy import IntParameter
import talib.abstract as ta
from pandas import DataFrame


class RSI(IStrategy):

    INTERFACE_VERSION = 3

    timeframe = "15m"

    startup_candle_count = 50

    can_short = False  # Only Long Positions

    # Strategy params
    buy_rsi = IntParameter(10, 40, default=30, space="buy")

    sell_rsi = IntParameter(60, 90, default=70, space="sell")

    rsi_period = IntParameter(5, 30, default=14, space="buy")

    ## Strategy atributes
    stoploss = -0.05

    minimal_roi = {
        "0": 0.04,
        "30": 0.02,
        "60": 0.01,
        "120": 0
    }

    trailing_stop = True

    trailing_stop_positive = 0.01

    trailing_stop_positive_offset = 0.03

    trailing_only_offset_is_reached = True

    # Indicators
    def populate_indicators(
        self,
        dataframe: DataFrame,
        metadata: dict
    ) -> DataFrame:

        dataframe["rsi"] = ta.RSI(
            dataframe,
            timeperiod=self.rsi_period.value
        )

        return dataframe

    # Entry
    def populate_entry_trend(
        self,
        dataframe: DataFrame,
        metadata: dict
    ) -> DataFrame:

        dataframe.loc[
            (
                dataframe["rsi"] < self.buy_rsi.value
            ),
            "enter_long"
        ] = 1

        return dataframe

    # Exit
    def populate_exit_trend(
        self,
        dataframe: DataFrame,
        metadata: dict
    ) -> DataFrame:

        dataframe.loc[
            (
                dataframe["rsi"] > self.sell_rsi.value
            ),
            "exit_long"
        ] = 1

        return dataframe
    
import numpy as np
import pandas as pd
import talib.abstract as ta


# Returns
def add_simple_return(
    df: pd.DataFrame,
    column: str = "close",
    window: int = 1,
) -> pd.DataFrame:

    df[f"ret_{window}"] = df[column].pct_change(window)

    return df


def add_log_return(
    df: pd.DataFrame,
    column: str = "close",
    window: int = 1,
) -> pd.DataFrame:

    df[f"log_ret_{window}"] = np.log(
        df[column] / df[column].shift(window)
    )

    return df


# Momentum
def add_momentum(
    df: pd.DataFrame,
    column: str = "close",
    window: int = 12,
) -> pd.DataFrame:

    df[f"mom_{window}"] = ta.MOM(
        df[column],
        timeperiod=window,
    )

    return df


# Volatility
def add_volatility(
    df: pd.DataFrame,
    return_col: str = "log_ret_1",
    window: int = 12,
) -> pd.DataFrame:

    df[f"vol_{window}"] = (
        df[return_col]
        .rolling(window)
        .std()
    )

    return df


def add_atr(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    window: int = 14,
) -> pd.DataFrame:

    df[f"atr_{window}"] = ta.ATR(
        df[high_col],
        df[low_col],
        df[close_col],
        timeperiod=window,
    )

    return df


# Trend
def add_sma(
    df: pd.DataFrame,
    column: str = "close",
    window: int = 20,
) -> pd.DataFrame:

    df[f"sma_{window}"] = ta.SMA(
        df[column],
        timeperiod=window,
    )

    return df


def add_ema(
    df: pd.DataFrame,
    column: str = "close",
    window: int= 12,
) -> pd.DataFrame:

    df[f"ema_{window}"] = ta.EMA(
        df[column],
        timeperiod=window,
    )

    return df


def add_distance_to_sma(
    df: pd.DataFrame,
    column: str = "close",
    window: int = 20,
) -> pd.DataFrame:

    sma = ta.SMA(
        df[column],
        timeperiod=window,
    )

    df[f"dist_sma_{window}"] = (
        df[column] / sma
    ) - 1

    return df


def add_distance_to_ema(
    df: pd.DataFrame,
    column: str = "close",
    window: int = 20,
) -> pd.DataFrame:

    ema = ta.EMA(
        df[column],
        timeperiod=window,
    )

    df[f"dist_ema_{window}"] = (
        df[column] / ema
    ) - 1

    return df


def add_macd(
    df: pd.DataFrame,
    column: str = "close",
    fastperiod: int = 12,
    slowperiod: int = 26,
    signalperiod: int = 9,
) -> pd.DataFrame:

    macd, signal, hist = ta.MACD(
        df[column],
        fastperiod=fastperiod,
        slowperiod=slowperiod,
        signalperiod=signalperiod,
    )

    df["macd"] = macd
    df["macd_signal"] = signal
    df["macd_hist"] = hist

    return df



# Oscillators
def add_rsi(
    df: pd.DataFrame,
    column: str = "close",
    window: int = 14,
) -> pd.DataFrame:

    df[f"rsi_{window}"] = ta.RSI(
        df[column],
        timeperiod=window,
    )

    return df


def add_stochastic(
    df: pd.DataFrame,
    high_col: str = "high",
    low_col: str = "low",
    close_col: str = "close",
    window: int = 14,
) -> pd.DataFrame:

    slowk, slowd = ta.STOCH(
        df[high_col],
        df[low_col],
        df[close_col],
        fastk_period=window,
    )

    df[f"stoch_k_{window}"] = slowk
    df[f"stoch_d_{window}"] = slowd

    return df


# Bollinger bands
def add_bollinger_bands(
    df: pd.DataFrame,
    column: str = "close",
    window: int = 20,
    nbdevup: float = 2.0,
    nbdevdn: float = 2.0,

) -> pd.DataFrame:

    upperband, middleband, lowerband = ta.BBANDS(
        df[column].values,
        timeperiod=window,
        nbdevup=nbdevup,
        nbdevdn=nbdevdn,
    )

    df[f"bb_upper_{window}"] = upperband
    df[f"bb_middle_{window}"] = middleband
    df[f"bb_lower_{window}"] = lowerband

    return df

# Volume
def add_obv(
    df: pd.DataFrame,
    close_col: str = "close",
    volume_col: str = "volume",
) -> pd.DataFrame:

    df["obv"] = ta.OBV(
        df[close_col],
        df[volume_col],
    )

    return df


def add_relative_volume(
    df: pd.DataFrame,
    volume_col: str = "volume",
    window: int = 20,
) -> pd.DataFrame:

    vol_sma = (
        df[volume_col]
        .rolling(window)
        .mean()
    )

    df[f"rel_volume_{window}"] = (
        df[volume_col] / vol_sma
    )

    return df


# Normalization
def add_zscore(
    df: pd.DataFrame,
    columns: list[str],
    window: int = 252,
) -> pd.DataFrame:

    for col in columns:

        mean = (
            df[col]
            .rolling(window)
            .mean()
        )

        std = (
            df[col]
            .rolling(window)
            .std()
        )

        df[f"{col}_z_{window}"] = (
            (df[col] - mean) / std
        )

    return df

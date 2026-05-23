import pandas as pd

from trading_tda.config import RAW_DATA_DIR


def load_freqtrade_ohlc(
    pairs: list = ['BTC_USDT', 'ETH_USDT'], 
    timeframe="15m"
) -> pd.DataFrame:

    data_dir =  RAW_DATA_DIR / "binance"

    data = {}

    for pair in pairs:
        freqtrade_pair = pair.replace("/", "_")

        file_path = data_dir / f"{freqtrade_pair}-{timeframe}.feather"

        if not file_path.exists():
            raise FileNotFoundError(f"Archivo: {file_path} no existe.")

        df = pd.read_feather(file_path)

        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df.set_index("date")

        data[pair] = df

    # Concat
    out = pd.concat(data, axis=1)

    out = out.sort_index(axis=1)

    out.columns =  out.columns.set_names(["pair", "price"])

    return out

def filter_dataframe_by_timerange(
    df: pd.DataFrame,
    timerange: str,
) -> pd.DataFrame:
    
    df = df.copy()

    # Split dates
    start_date_str, end_date_str = timerange.split("-")

    filtered_df = df.loc[start_date_str:end_date_str]

    return filtered_df

import argparse
from pathlib import Path
from time import perf_counter

import pandas as pd
from loguru import logger

from trading_tda.data import load_freqtrade_ohlc
from trading_tda.config import DATA_DIR, PROCESSED_DATA_DIR
from trading_tda.data import filter_dataframe_by_timerange
from trading_tda.storage import (
    init_hdf5, 
    save_raw_dataset_to_hdf5
)
from trading_tda.pipelines.windows_pipelines import build_windows_pipeline
from trading_tda.pipelines.embeddings_pipelines import build_takens_pipeline
from trading_tda.pipelines.persistence_pipelines import build_persistence_pipeline
from trading_tda.pipelines.persistence_vectors_pipelines import build_persistence_vectors_pipeline
from trading_tda.pipelines.persistence_entropy_pipelines import build_persistence_entropy_pipeline
from trading_tda.pipelines.betti_curves_pipelines import build_betti_curves_pipeline
from trading_tda.pipelines.persistence_statistics_pipelines import build_persistence_statistics_pipeline
from trading_tda.h5_utils import print_hdf5_structure


logger.remove()

logger.add(
    sink=lambda msg: print(msg, end=""),
    level="INFO",
    colorize=True
)


def _build_h5_data_path(**kwargs):
    return Path(
        PROCESSED_DATA_DIR /
        f"{kwargs.get('pair')}_"
        f"{kwargs.get('timeframe')}_"
        f"{kwargs.get('timerange')}_"
        f"tda.h5"
    )


def _save_raw_data_pipeline(
        h5_path: Path,
        df: pd.DataFrame, 
        signal_names: list, 
): 
    
    # signals
    for signal_name in signal_names: 
        _ = save_raw_dataset_to_hdf5(
            h5_path=h5_path, 
            dataset_name=signal_name, 
            data=df[signal_name].values,
            astype="f4"
        )
    
    # timestamps
    _ = save_raw_dataset_to_hdf5(
        h5_path=h5_path, 
        dataset_name="timestamps", 
        data=df.index.tz_localize(None),
        astype="i8"
    )


def _compute_homology_dims(homology_dim=1): 
    return list(range(homology_dim + 1))
    

def run_data_pipeline(
        df: pd.DataFrame, 
        pair: str, 
        timeframe: str, 
        timerange: str, 
        window_size: int,
        stride: int, 
        dimension: int, 
        delay: int,
        exchange: str = "binance", 
        signal_names: list = ["close"], 
        homology_dim: int = 1
): 
    
    logger.info("Building HDF5 path...")
    
    h5_data_path = _build_h5_data_path(
        pair=pair, 
        timeframe=timeframe, 
        timerange=timerange,
    )

    logger.info(f"Initializing HDF5 store: {h5_data_path}")

    _ = init_hdf5(
        h5_path=h5_data_path, 
        pair=pair, 
        timeframe=timeframe, 
        timerange=timerange, 
        exchange=exchange
    )

    logger.info("Saving raw datasets...")
    _ = _save_raw_data_pipeline(
        h5_path=h5_data_path,
        df=df, 
        signal_names=signal_names
    )

    logger.info("Building windows...")
    _ = build_windows_pipeline(
        h5_path=h5_data_path,
        window_size=window_size, 
        stride=stride, 
        signal_names=signal_names 
    )

    logger.info("Building Takens embeddings...")

    _ = build_takens_pipeline(
        h5_path=h5_data_path, 
        window_size=window_size, 
        stride=stride, 
        dimension=dimension,
        delay=delay,
        signal_names=signal_names
    )
    
    logger.info("Computing persistence diagrams...")

    _ = build_persistence_pipeline(
        h5_path=h5_data_path, 
        window_size=window_size, 
        dimension=dimension, 
        delay=delay, 
        maxdim=homology_dim, 
        signal_names=signal_names
    )

    logger.info("Vectorizing persistence diagrams...")

    _ = build_persistence_vectors_pipeline(
        h5_path=h5_data_path, 
        window_size=window_size, 
        signal_names=signal_names, 
        dimension=dimension, 
        delay=delay, 
        homology_dims=_compute_homology_dims(homology_dim)   
    )

    logger.info("Computing Betti curves...")

    _ = build_betti_curves_pipeline(
        h5_path=h5_data_path,
        window_size=window_size, 
        signal_names=signal_names, 
        dimension=dimension, 
        delay=delay, 
        homology_dims=_compute_homology_dims(homology_dim)   
    )

    logger.info("Computing persistence entropy...")

    _ = build_persistence_entropy_pipeline(
        h5_path=h5_data_path,
        window_size=window_size, 
        signal_names=signal_names, 
        dimension=dimension, 
        delay=delay, 
        homology_dims=_compute_homology_dims(homology_dim)   
    )

    logger.info("Computing persistence statistics...")
    
    _ = build_persistence_statistics_pipeline(
        h5_path=h5_data_path,
        window_size=window_size, 
        signal_names=signal_names, 
        dimension=dimension, 
        delay=delay, 
        homology_dims=_compute_homology_dims(homology_dim)
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Trading TDA pipeline CLI"
    )

    parser.add_argument("--pair", type=str, default="BTC_USDT")
    parser.add_argument("--timeframe", type=str, default="4h")
    parser.add_argument("--timerange", type=str, default="20250501-20260501")

    parser.add_argument("--window-size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=1)

    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--delay", type=int, default=1)

    parser.add_argument("--exchange", type=str, default="binance")

    parser.add_argument(
        "--signals",
        nargs="+",
        default=["close", "volume"],
        help="Signals to process"
    )

    parser.add_argument(
        "--homology-dim",
        type=int,
        default=1
    )

    return parser.parse_args()


def main():

    args = parse_args()

    start = perf_counter()

    logger.info(
        f"Running pipeline | "
        f"pair={args.pair} "
        f"timeframe={args.timeframe} "
        f"window={args.window_size}"
    )

    logger.info("Loading OHLC data...")

    df = (
        load_freqtrade_ohlc(
            pairs=[args.pair],
            timeframe=args.timeframe
        )
        .xs(args.pair, axis=1, level="pair")
    )

    df = filter_dataframe_by_timerange(
        df,
        timerange=args.timerange
    )

    _ = run_data_pipeline(
        df=df,
        pair=args.pair,
        timeframe=args.timeframe,
        timerange=args.timerange,
        window_size=args.window_size,
        stride=args.stride,
        dimension=args.dimension,
        delay=args.delay,
        exchange=args.exchange,
        signal_names=args.signals,
        homology_dim=args.homology_dim
    )

    logger.info(
        "TDA-data pipeline completed "
        f"in {perf_counter() - start:.2f}"
    )

    logger.success("Pipeline completed successfully.")


if __name__ == '__main__': 
    _ = main()
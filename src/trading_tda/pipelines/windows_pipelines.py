import numpy as np
from pathlib import Path
import h5py

from trading_tda.windowing import (
    create_data_windows,
    create_timestamps_windows
)
from trading_tda.loaders import load_raw_signals
from trading_tda.storage import save_windows_to_hdf5


def build_windows(
    series: np.ndarray,
    timestamps: np.ndarray,
    window_size: int,
    stride: int,
):
    data_windows = create_data_windows(
        series=series,
        size=window_size,
        stride=stride,
    )

    timestamp_windows = create_timestamps_windows(
        timestamps=timestamps,
        size=window_size,
        stride=stride,
    )

    return data_windows, timestamp_windows


def build_windows_pipeline(
    h5_path: Path,
    window_size: int,
    signal_names: list,
    stride: int = 1,
):
    timestamps, signals = load_raw_signals(
        h5_path=h5_path,
        signal_names=signal_names,
    )

    for signal_name, series in signals.items():

        windows, timestamp_windows = build_windows(
            series=series,
            timestamps=timestamps,
            window_size=window_size,
            stride=stride,
        )

        _ = save_windows_to_hdf5(
            h5_path=h5_path,
            signal_name=signal_name,
            windows=windows,
            timestamps=timestamp_windows,
            window_size=window_size,
            stride=stride,
            astype="f4",
        )

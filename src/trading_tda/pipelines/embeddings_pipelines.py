from pathlib import Path

import numpy as np

from trading_tda.tda import batch_takens_embedding
from trading_tda.storage import save_takens_to_hdf5
from trading_tda.loaders import load_window_dataset


def compute_takens_embeddings(
    windows: np.ndarray,
    dimension: int,
    delay: int,
    stride: int,
):
    embeddings = batch_takens_embedding(
        windows=windows,
        dim=dimension,
        delay=delay,
        stride=stride,
    )

    return embeddings


def build_takens_dataset(
    h5_path: Path,
    signal_name: str,
    window_size: int,
    stride: int,
    dimension: int,
    delay: int,
):
    """
    Construye embeddings de Takens a partir de windows.
    """

    # load
    windows = load_window_dataset(
        h5_path=h5_path,
        signal_name=signal_name,
        window_size=window_size,
    )

    # transform
    embeddings = compute_takens_embeddings(
        windows=windows,
        dimension=dimension,
        delay=delay,
        stride=stride,
    )

    # persist
    _ = save_takens_to_hdf5(
        h5_path=h5_path,
        embeddings=embeddings,
        signal_name=signal_name,
        window_size=window_size,
        stride=stride,
        dimension=dimension,
        delay=delay,
    )


def build_takens_pipeline(
    h5_path: Path,
    window_size: int,
    signal_names: list,
    stride: int,
    dimension: int, 
    delay: int
):

    for signal_name in signal_names:
        _ = build_takens_dataset(
            h5_path=h5_path, 
            signal_name=signal_name, 
            window_size=window_size, 
            stride=stride, 
            dimension=dimension,
            delay=delay
        )

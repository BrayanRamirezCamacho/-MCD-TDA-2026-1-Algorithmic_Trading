from pathlib import Path

import h5py


def load_raw_signals(
    h5_path: Path,
    signal_names: list,
):
    with h5py.File(h5_path, "r") as h5:
        raw_group = h5["raw"]

        timestamps = raw_group["timestamps"][:]

        signals = {
            signal_name: raw_group[signal_name][:]
            for signal_name in signal_names
        }

    return timestamps, signals


def load_window_dataset(
    h5_path: Path,
    signal_name: str,
    window_size: int,
):
    source_group = f"windows/w{window_size}"
    window_dataset = f"{source_group}/{signal_name}"

    with h5py.File(h5_path, "r") as h5:
        windows = h5[window_dataset][:]

    return windows


def load_takens_embeddings(
    h5_path: Path,
    signal_name: str,
    window_size: int,
    dimension: int,
    delay: int,
):
    """Carga Takens embeddings desde HDF5."""

    source_path = (
        "takens/"
        f"w{window_size}/"
        f"d{dimension}_t{delay}/"
        f"{signal_name}"
    )
    with h5py.File(h5_path, "r") as h5:

        embeddings = h5[source_path][:]

    return embeddings


def load_persistence(
    h5_path: Path,
    signal_name: str,
    window_size: int,
    dimension: int,
    delay: int,
    homology_dim: int,
):
    """Carga persistence features desde HDF5."""

    source_path = (
        "persistence/"
        f"w{window_size}/"
        f"d{dimension}_t{delay}/"
        f"h{homology_dim}/"
        f"{signal_name}"
    )

    with h5py.File(h5_path, "r") as h5:
        persistence = h5[source_path][:]

    return persistence


def load_vectorization(
        h5_path: Path, 
        vectorization_type: str, 
        signal_name: str, 
        window_size: int, 
        dimension: int, 
        delay: int, 
        homology_dim: int, 
): 
    """Carga vectorizaciones desde HDF5."""

    source_path = (
        "vectorization/"
        f"{vectorization_type}/"
        f"w{window_size}/"
        f"d{dimension}_t{delay}/"
        f"h{homology_dim}/"
        f"{signal_name}"
    )

    with h5py.File(h5_path, "r") as h5: 
        data = h5[source_path][:]

    return data


def load_feature(
    h5_path: Path,
    feature_name: str,
    signal_name: str,
    window_size: int,
    dimension: int,
    delay: int,
    homology_dim: int
):
    """Carga features topológicas desde HDF5."""

    source_path = (
        "features/"
        f"w{window_size}/"
        f"d{dimension}_t{delay}/"
        f"{feature_name}/"
        f"h{homology_dim}/"
        f"{signal_name}"
    )

    with h5py.File(h5_path, "r") as h5:
        feature = h5[source_path][:]

    return feature
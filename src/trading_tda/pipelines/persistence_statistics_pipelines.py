from pathlib import Path

import numpy as np

from trading_tda.loaders import load_persistence

from trading_tda.loaders import (
    load_vectorization, 
    load_feature
)

from trading_tda.storage import save_feature_to_hdf5


def _contains_infinite(vectors: np.ndarray):
    return np.isinf(vectors).any()


def _contains_nan(vectors: np.ndarray):
    return np.isnan(vectors).any()


def _clean_inf_persistence_vectors(vectors: np.ndarray):
    """Elimina valores inf conservando estructura matricial."""

    m, n = vectors.shape

    finite_mask = np.isfinite(vectors)

    cleaned = vectors[finite_mask]

    # reshape conservando filas
    cleaned = cleaned.reshape(m, n - 1)  # 

    return cleaned


def persistence_statistics(
        persistence_vectors: np.ndarray
):
    """Calcula features estadísticas de vectores de persistencia."""

    if _contains_infinite(persistence_vectors): # H0
        persistence_vectors = _clean_inf_persistence_vectors(persistence_vectors)
    
    if _contains_nan(persistence_vectors): # H1
        persistence_vectors = np.nan_to_num(persistence_vectors, nan=0.0)
    
    # stats
    stats = np.column_stack([
            np.sum(persistence_vectors, axis=1),
            np.mean(persistence_vectors, axis=1), 
            np.std(persistence_vectors, axis=1),
            np.median(persistence_vectors, axis=1),
            np.max(persistence_vectors, axis=1),
            np.min(persistence_vectors, axis=1),
    ])

    return stats


def build_persistence_statistics_dataset(
    h5_path: Path,
    signal_name: str,
    window_size: int,
    dimension: int,
    delay: int,
    homology_dim: int,
):
    """
    Construye dataset de features tolológicas a partir de persistence vectors.
    """

    # load
    persistence_vectors = load_vectorization(
        h5_path=h5_path,
        signal_name=signal_name,
        window_size=window_size,
        dimension=dimension,
        delay=delay,
        homology_dim=homology_dim,
        vectorization_type="persistence_vectors"
    )

    # transform
    stats = persistence_statistics(persistence_vectors)

    feature_names = [
        "total_persistence",
        "mean_persistence", 
        "std_persistence", 
        "median_persistence",
        "max_persistence",
        "min_persistence",
    ]

    # save
    _ = save_feature_to_hdf5(
        h5_path=h5_path,
        feature_name="persistence_statistics",
        feature=stats,
        signal_name=signal_name,
        window_size=window_size,
        dimension=dimension,
        delay=delay,
        homology_dim=homology_dim,
        column_names=feature_names
    )


def build_persistence_statistics_pipeline(
    h5_path: Path,
    window_size: int,
    signal_names: list,
    dimension: int,
    delay: int,
    homology_dims: list,
):
    """
    Pipeline para construir datasets de persistent entropy
    para múltiples señales y dimensiones de homología.
    """

    for signal_name in signal_names:

        for homology_dim in homology_dims:

            _ = build_persistence_statistics_dataset(
                h5_path=h5_path,
                signal_name=signal_name,
                window_size=window_size,
                dimension=dimension,
                delay=delay,
                homology_dim=homology_dim,
            )

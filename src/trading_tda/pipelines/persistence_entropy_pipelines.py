from pathlib import Path

import numpy as np

from trading_tda.loaders import load_persistence
from trading_tda.storage import save_feature_to_hdf5
from trading_tda.tda import compute_persistence_entropy


def batch_persistence_entropy(
    persistence_diagrams: np.ndarray,
    homology_dim: int,
):
    """
    Calcula la entropía persistente para un batch de diagramas.
    """
    entropies = [
        compute_persistence_entropy(
            diagram=diagram, 
            homology_dim=homology_dim
        )[0]
        for diagram in persistence_diagrams
    ]

    return np.asarray(entropies)


def build_persistence_entropy_dataset(
    h5_path: Path,
    signal_name: str,
    window_size: int,
    dimension: int,
    delay: int,
    homology_dim: int,
):
    """
    Construye dataset de persistent entropy a partir de persistence diagrams.
    """

    # load
    persistence_diagrams = load_persistence(
        h5_path=h5_path,
        signal_name=signal_name,
        window_size=window_size,
        dimension=dimension,
        delay=delay,
        homology_dim=homology_dim,
    )

    # transform
    entropy = batch_persistence_entropy(
        persistence_diagrams=persistence_diagrams,
        homology_dim=homology_dim
    )

    # persist
    _ = save_feature_to_hdf5(
        h5_path=h5_path,
        feature_name="persistence_entropy",
        feature=entropy,
        signal_name=signal_name,
        window_size=window_size,
        dimension=dimension,
        delay=delay,
        homology_dim=homology_dim,
    )


def build_persistence_entropy_pipeline(
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

            _ = build_persistence_entropy_dataset(
                h5_path=h5_path,
                signal_name=signal_name,
                window_size=window_size,
                dimension=dimension,
                delay=delay,
                homology_dim=homology_dim,
            )

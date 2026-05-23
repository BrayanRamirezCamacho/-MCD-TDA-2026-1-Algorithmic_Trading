from pathlib import Path

import numpy as np

from trading_tda.loaders import load_persistence
from trading_tda.storage import save_vectors_to_hdf5
from trading_tda.tda import compute_persistence_vector


def batch_persistence_vectors(
    persistence_diagrams: np.ndarray,
    sort: bool = True,
    descending: bool = True,
):
    """
    Calcula vectores de persistencia para un batch de diagramas.
    """

    vectors = [
        compute_persistence_vector(
            diagram=diagram,
            sort=sort,
            descending=descending,
        )
        for diagram in persistence_diagrams
    ]

    return np.asarray(vectors)


def build_persistence_vectors_dataset(
    h5_path: Path,
    signal_name: str,
    window_size: int,
    dimension: int,
    delay: int,
    homology_dim: int,
    sort: bool = True,
    descending: bool = True,
):
    """
    Construye persistence vectors a partir de diagramas de persistencia.
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
    persistence_vectors = batch_persistence_vectors(
        persistence_diagrams=persistence_diagrams,
        sort=sort,
        descending=descending,
    )

    # persist
    _ = save_vectors_to_hdf5(
        h5_path=h5_path,
        vectors=persistence_vectors,
        signal_name=signal_name,
        window_size=window_size,
        dimension=dimension,
        delay=delay,
        homology_dim=homology_dim,
        vectorization_type="persistence_vectors"
    )


def build_persistence_vectors_pipeline(
    h5_path: Path,
    window_size: int,
    signal_names: list,
    dimension: int,
    delay: int,
    homology_dims: list,
    sort: bool = True,
    descending: bool = True,
):
    """
    Pipeline para construir vectores de persistencia
    para múltiples señales y dimensiones de homología.
    """

    for signal_name in signal_names:

        for homology_dim in homology_dims:

            _ = build_persistence_vectors_dataset(
                h5_path=h5_path,
                signal_name=signal_name,
                window_size=window_size,
                dimension=dimension,
                delay=delay,
                homology_dim=homology_dim,
                sort=sort,
                descending=descending,
            )

from pathlib import Path

import numpy as np

from trading_tda.loaders import load_persistence
from trading_tda.storage import save_betti_curves_to_hdf5
from trading_tda.tda import betti_curve


def batch_betti_curves(
    persistence_diagrams: np.ndarray,
    homology_dim: int,
    n_bins: int = 100,
):
    """
    Calcula Betti curves para un batch de diagramas.
    """

    curves = []

    t_reference = None

    for diagram in persistence_diagrams:

        t, curve = betti_curve(
            diagram=diagram,
            homology_dim=homology_dim,
            n_bins=n_bins,
        )

        if t_reference is None:
            t_reference = t

        curves.append(curve)

    curves = np.asarray(curves)

    return t_reference, curves


def build_betti_curves_dataset(
    h5_path: Path,
    signal_name: str,
    window_size: int,
    dimension: int,
    delay: int,
    homology_dim: int,
    n_bins: int = 100,
):
    """
    Construye Betti curves a partir de persistence diagrams.
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
    t, curves = batch_betti_curves(
        persistence_diagrams=persistence_diagrams,
        homology_dim=homology_dim,
        n_bins=n_bins,
    )

    # persist
    _ = save_betti_curves_to_hdf5(
        h5_path=h5_path,
        curves=curves,
        t=t,
        signal_name=signal_name,
        homology_dim=homology_dim,
        window_size=window_size,
        dimension=dimension,
        delay=delay,
    )


def build_betti_curves_pipeline(
    h5_path: Path,
    window_size: int,
    signal_names: list,
    dimension: int,
    delay: int,
    homology_dims: list,
    n_bins: int = 100,
):
    """
    Pipeline para construir Betti curves.
    
    vectorization/
        betti_curve/
            w64/
                d3_t1/
                    h0/
                        t
                        close
                        volume

                    h1/
                        t
                        close
                        volume
    """

    for signal_name in signal_names:

        for homology_dim in homology_dims:

            _ = build_betti_curves_dataset(
                h5_path=h5_path,
                signal_name=signal_name,
                window_size=window_size,
                dimension=dimension,
                delay=delay,
                homology_dim=homology_dim,
                n_bins=n_bins,
            )

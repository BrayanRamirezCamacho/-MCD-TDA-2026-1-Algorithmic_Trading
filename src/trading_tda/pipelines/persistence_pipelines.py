from pathlib import Path

import numpy as np
import h5py

from trading_tda.tda import compute_persistence_diagram
from trading_tda.storage import save_persistence_to_hdf5
from trading_tda.loaders import load_takens_embeddings


def batch_persistence_diagrams(
        embeddings: np.ndarray, 
        maxdim: int = 1
): 
    """Computa diagramas de persistencia para múltiples embeddings."""
    
    n_windows = embeddings.shape[0]

    diagrams_H0 = []
    diagrams_H1 = []
    
    for i in range(n_windows):

        H0, H1 = compute_persistence_diagram(
            point_cloud=embeddings[i], 
            maxdim=maxdim
        )

        diagrams_H0.append(H0)
        diagrams_H1.append(H1)  

    return diagrams_H0, diagrams_H1


def pad_diagrams(
        diagrams: list, 
        pad_value = np.nan
): 
    """Pad sobre diagramas de persistencia a tamaño fijo."""

    max_points = max(len(dgm) for dgm in diagrams)

    padded = np.full(
        (
            len(diagrams),
            max_points,
            2,
        ),
        pad_value,
        dtype=np.float64,
    )

    for i, dgm in enumerate(diagrams):
        padded[i, :len(dgm)] = dgm

    return padded


def build_persistence_dataset(
        h5_path: Path, 
        window_size: int, 
        dimension: int, 
        delay: int, 
        maxdim: int, 
        signal_name: str
): 
    """Construye dataset para una signal."""

    embeddings = load_takens_embeddings(
        h5_path=h5_path, 
        signal_name=signal_name, 
        window_size=window_size, 
        dimension=dimension, 
        delay=delay, 
    )

    diagrams = batch_persistence_diagrams(
        embeddings=embeddings, 
        maxdim=maxdim
    )

    for i, diagram in enumerate(diagrams): 
        
        padded_diagram = pad_diagrams(diagram)

        _ = save_persistence_to_hdf5(
            h5_path=h5_path, 
            diagrams=padded_diagram, 
            homology_dim=i, 
            window_size=window_size, 
            dimension=dimension, 
            delay=delay, 
            signal_name=signal_name
        )


def build_persistence_pipeline(    h5_path: Path, 
        window_size: int, 
        dimension: int, 
        delay: int, 
        maxdim: int, 
        signal_names: list
): 
    
    for signal_name in signal_names: 
        build_persistence_dataset(
            h5_path=h5_path, 
            window_size=window_size, 
            dimension=dimension, 
            delay=delay, 
            maxdim=maxdim, 
            signal_name=signal_name
        )
    
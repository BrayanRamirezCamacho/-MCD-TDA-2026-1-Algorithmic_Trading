from pathlib import Path
from datetime import datetime

import h5py
import numpy as np

def _now(): 
    return datetime.now().isoformat()


def _replace_dataset(group: h5py.Group, name: str): 
    """Elimina dataset previo si existe."""
    if name in group:
        del group[name]


def _write_dataset_to_hdf5(
        group: h5py.Group, 
        name: str, 
        data: np.ndarray,
        astype: str ="f4", 
        compression: str ="gzip"
): 
    """Escribe HDF5 con metadata estándar."""

    # elimina dataset si existe
    _replace_dataset(group, name)

    ds = group.create_dataset(
        name, 
        data=data.astype(astype),
        compression=compression
    )

    # standard metadata
    ds.attrs["dtype"] = astype
    ds.attrs["shape"] = data.shape
    ds.attrs["ndim"] = data.ndim
    ds.attrs["compression"] = compression
    ds.attrs["created_at"] = _now()

    return ds

def init_hdf5(
        h5_path: Path, 
        pair: str, 
        timeframe: str, 
        timerange: str,
        exchange: str
): 
    """Inicializa HDF5 global."""

    with h5py.File(h5_path, "a") as h5: 

        h5.attrs["asset"] = pair
        h5.attrs["timeframe"] = timeframe
        h5.attrs["timeframe"] = timerange
        h5.attrs["exchange"] = exchange
        h5.attrs["created_at"] = _now()


def save_raw_dataset_to_hdf5(
        h5_path: Path, 
        dataset_name: str,
        data: np.ndarray,  
        astype: str = "f4"
): 
    """
    Guarda series base. 
        raw/close
    """
    with h5py.File(h5_path, "a") as h5:
        
        # crea grupo
        group = h5.require_group("raw")
        
        _ = _write_dataset_to_hdf5(
            group, 
            dataset_name, 
            data, 
            astype
        )
        

def save_windows_to_hdf5(
        h5_path: Path, 
        signal_name: str, 
        windows: np.ndarray, 
        window_size: int, 
        stride: int = 1, 
        timestamps: np.ndarray = None,
        astype: str = "f4"
): 
    """
    Guarda ventanas.
        windows/
            w128/
                close
                timestamp
    """

    group_path = f"windows/w{window_size}"

    with h5py.File(h5_path, "a") as h5: 

        group = h5.require_group(group_path)

        # metadata grupo
        group.attrs["window_size"] = window_size
        group.attrs["stride"] = stride
        group.attrs["created_at"] = _now()

        # dataset principal
        ds = _write_dataset_to_hdf5(
            group=group, 
            name=signal_name, 
            data=windows, 
            astype=astype
        )

        # metadata dataset
        ds.attrs["n_windows"] = windows.shape[0]

        # timestamps alineados
        if timestamps is not None: 
            _ = _write_dataset_to_hdf5(
                group=group, 
                name="timestamp", 
                data=timestamps, 
                astype="i8"
            ) 


def save_takens_to_hdf5(
        h5_path: Path, 
        signal_name: str, 
        embeddings: np.ndarray, 
        window_size: int, 
        stride: int = 1,
        timestamps: np.ndarray = None,
        dimension: int = 3, 
        delay: int = 1, 
        astype: str = "f4", 
): 
    """
    Guarda Takens embeddings.
        takens/
            w128/
                d3_t1/
                    close
                    timestamps
    """

    group_path = (
        "takens/"
        f"w{window_size}/"
        f"d{dimension}_t{delay}"
    )

    with h5py.File(h5_path, "a") as h5: 

        group = h5.require_group(group_path)

        # metadata grupo
        group.attrs["window_size"] = window_size
        group.attrs["stride"] = stride
        group.attrs["dimension"] = dimension
        group.attrs["delay"] = delay
        group.attrs["created_at"] = _now()

        ds = _write_dataset_to_hdf5(
            group=group, 
            name=signal_name, 
            data=embeddings, 
            astype=astype
        )

        # embeddings metadata
        ds.attrs["n_windows"] = embeddings.shape[0]
        ds.attrs["n_points"] = embeddings.shape[1]
        ds.attrs["embedding_dimenssion"] = embeddings.shape[2]

        # timestamps alineados
        if timestamps is not None: 
            _ = _write_dataset_to_hdf5(
                group=group, 
                name="timestamp", 
                data=timestamps, 
                astype="i8"
            ) 


def save_persistence_to_hdf5(
        h5_path: Path, 
        diagrams: np.ndarray, 
        homology_dim: int, 
        window_size: int,
        dimension: int, 
        delay: int, 
        signal_name: str,
        filtration: str = "vietoris_rips", 
        astype: str = "f4"
): 
    """
    Guarda peristence diagrams.
        persistence/
            w128/
                d3_t1/
                    h0/
                        close
                    h1/
                        close
                
    """

    group_path = (
        "persistence/"
        f"w{window_size}/"
        f"d{dimension}_t{delay}/"
        f"h{homology_dim}"
    )

    with h5py.File(h5_path, "a") as h5: 
        
        group = h5.require_group(group_path)

        # metadata grupo
        group.attrs["window_size"] = window_size
        group.attrs["dimension"] = dimension
        group.attrs["delay"] = delay
        group.attrs["homology_dim"] = homology_dim
        group.attrs["filtration"] = filtration
        group.attrs["created_at"] = _now()

        ds = _write_dataset_to_hdf5(
            group=group, 
            name=signal_name,
            data=diagrams, 
            astype=astype
        )

        ds.attrs["n_windows"] = diagrams.shape[0]


def save_vectors_to_hdf5(
    h5_path: Path,
    vectors: np.ndarray,
    vectorization_type: str,
    homology_dim: int,
    window_size: int,
    dimension: int,
    delay: int,
    signal_name: str,
    filtration: str = "vietoris_rips",
    astype: str = "f4",
):
    """
    Guarda vectorización de diagramas de persistencia.

    Estructura:
        vectorization/
            persistence_vectors/            
                w128/
                    d3_t1/
                        h0/
                            close
                        h1/
                            close
    """

    group_path = (
        "vectorization/"
        f"{vectorization_type}/"
        f"w{window_size}/"
        f"d{dimension}_t{delay}/"
        f"h{homology_dim}"
    )

    with h5py.File(h5_path, "a") as h5:

        group = h5.require_group(group_path)

        # metadata grupo
        group.attrs["vectorization_type"] = vectorization_type
        group.attrs["window_size"] = window_size
        group.attrs["dimension"] = dimension
        group.attrs["delay"] = delay
        group.attrs["homology_dim"] = homology_dim
        group.attrs["filtration"] = filtration
        group.attrs["created_at"] = _now()

        ds = _write_dataset_to_hdf5(
            group=group,
            name=signal_name,
            data=vectors,
            astype=astype,
        )

        ds.attrs["n_windows"] = vectors.shape[0]
        ds.attrs["vector_size"] = vectors.shape[1]


def save_betti_curves_to_hdf5(
    h5_path: Path,
    curves: np.ndarray,
    t: np.ndarray,
    signal_name: str,
    homology_dim: int,
    window_size: int,
    dimension: int,
    delay: int,
    filtration: str = "vietoris_rips",
    astype: str = "f4",
):
    """
    Guarda Betti curves.

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

    group_path = (
        "vectorization/"
        "betti_curve/"
        f"w{window_size}/"
        f"d{dimension}_t{delay}/"
        f"h{homology_dim}"
    )

    with h5py.File(h5_path, "a") as h5:

        group = h5.require_group(group_path)

        # metadata grupo
        group.attrs["vectorization_type"] = "betti_curve"
        group.attrs["window_size"] = window_size
        group.attrs["dimension"] = dimension
        group.attrs["delay"] = delay
        group.attrs["homology_dim"] = homology_dim
        group.attrs["filtration"] = filtration
        group.attrs["created_at"] = _now()

        if "t" not in group:

            ds_t = _write_dataset_to_hdf5(
                group=group,
                name="t",
                data=t,
                astype=astype,
            )

            ds_t.attrs["n_bins"] = len(t)

        ds_curve = _write_dataset_to_hdf5(
            group=group,
            name=signal_name,
            data=curves,
            astype=astype,
        )

        ds_curve.attrs["n_windows"] = curves.shape[0]
        ds_curve.attrs["n_bins"] = curves.shape[1]


def save_feature_to_hdf5(
        h5_path: Path, 
        feature_name: str, 
        signal_name: str, 
        feature: np.ndarray, 
        window_size: int, 
        dimension: int, 
        delay: int, 
        homology_dim: int,
        column_names: list = None, 
        astype: str = "f4"
): 
    """
    Guarda features topológicas.
        features/
            w128/
                d3_t1/
                    persistence_entropy/
                        h0/
                            close
                        h1/
                            close
    """

    group_path = (
        "features/"
        f"w{window_size}/"
        f"d{dimension}_t{delay}/"
        f"{feature_name}/"
        f"h{homology_dim}"
    )

    with h5py.File(h5_path, "a") as h5: 

        group = h5.require_group(group_path)

        # metadata grupo
        group.attrs["window_size"] = window_size
        group.attrs["dimension"] = dimension
        group.attrs["delay"] = delay
        group.attrs["created_at"] = _now()

        ds = _write_dataset_to_hdf5(
            group=group, 
            name=signal_name, 
            data=feature,
            astype=astype 
        )

        ds.attrs["n_sanples"] = feature.shape[0]

        if column_names is not None:

            ds.attrs["column_names"] = np.array(
                column_names,
                dtype="S"
            )

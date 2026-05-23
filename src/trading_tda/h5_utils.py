import h5py


def print_hdf5_structure(h5_path):
    """
    Imprime estructura completa del HDF5.
    """

    def visitor(name, obj):

        indent = "    " * name.count("/")

        if isinstance(obj, h5py.Group):
            print(f"{indent}[GROUP] {name}")

        elif isinstance(obj, h5py.Dataset):
            print(
                f"{indent}[DATASET] {name} "
                f"shape={obj.shape}"
            )

    with h5py.File(h5_path, "r") as h5:
        h5.visititems(visitor)

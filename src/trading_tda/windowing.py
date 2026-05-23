import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def create_data_windows(
        series: np.ndarray, 
        size: int, 
        stride: int = 1, 
):
    """
    Crea sliding windows a partir de una serie de tiempo 1D.
    """ 
    windows = sliding_window_view(series, window_shape=size)

    windows = windows[::stride]

    return windows


def create_timestamps_windows(
        timestamps: np.ndarray, 
        size: int, 
        stride: int = 1
): 
    """Alinea timestamps con sliding windows"""
    
    aligned = timestamps[size - 1:]
    
    aligned = aligned[::stride]
    
    return aligned
    
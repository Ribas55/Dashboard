"""
Helper functions for the dashboard application.
"""

import pandas as pd
import numpy as np
from typing import Union, List

def normalize_series(
    data: Union[pd.Series, np.ndarray, List[float]],
    feature_range: tuple = (0, 1)
) -> np.ndarray:
    """
    Normalize a series of values to a specified range (default 0-1).
    
    Parameters:
    -----------
    data : array-like
        The data to normalize
    feature_range : tuple, default=(0, 1)
        The range to normalize to
        
    Returns:
    --------
    np.ndarray
        The normalized data
    """
    data_array = np.asarray(data)
    if len(data_array) == 0:
        return np.array([])
        
    min_val = np.min(data_array)
    max_val = np.max(data_array)
    
    if max_val == min_val:
        # If all values are the same, return middle of the range
        middle = (feature_range[0] + feature_range[1]) / 2
        return np.full_like(data_array, middle)
    
    # Scale to [0, 1]
    scaled = (data_array - min_val) / (max_val - min_val)
    
    # Scale to feature_range
    min_out, max_out = feature_range
    scaled = scaled * (max_out - min_out) + min_out
    
    return scaled

def format_number(value: float, precision: int = 2) -> str:
    """
    Format a number with thousands separator and specified precision.
    
    Parameters:
    -----------
    value : float
        The number to format
    precision : int, default=2
        The number of decimal places
        
    Returns:
    --------
    str
        The formatted number
    """
    return f"{value:,.{precision}f}" 
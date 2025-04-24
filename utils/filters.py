"""
Filter functions for the dashboard application.
"""

import pandas as pd
from typing import List, Optional, Union, Dict

def filter_dataframe(
    df: pd.DataFrame,
    filters: Dict[str, Union[str, List[str], tuple]]
) -> pd.DataFrame:
    """
    Filter a DataFrame based on column filters.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to filter
    filters : Dict
        Dictionary mapping column names to filter values
        
    Returns:
    --------
    pd.DataFrame
        The filtered DataFrame
    """
    filtered_df = df.copy()
    
    if filtered_df.empty:
        return filtered_df
    
    for column, filter_value in filters.items():
        if column not in filtered_df.columns:
            continue
            
        if isinstance(filter_value, (list, tuple)):
            # Multi-select filter
            if filter_value:  # Only apply if the list is not empty
                filtered_df = filtered_df[filtered_df[column].isin(filter_value)]
        elif filter_value is not None:
            # Single-select filter
            filtered_df = filtered_df[filtered_df[column] == filter_value]
    
    return filtered_df

def filter_date_range(
    df: pd.DataFrame,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None,
    date_column: str = "invoice_date"
) -> pd.DataFrame:
    """
    Filter a DataFrame based on a date range.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The DataFrame to filter
    start_date : pd.Timestamp, optional
        The start date (inclusive)
    end_date : pd.Timestamp, optional
        The end date (inclusive)
    date_column : str, default="invoice_date"
        The column containing dates
        
    Returns:
    --------
    pd.DataFrame
        The filtered DataFrame
    """
    if df.empty or date_column not in df.columns:
        return df
    
    filtered_df = df.copy()
    
    # Make sure the date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(filtered_df[date_column]):
        filtered_df[date_column] = pd.to_datetime(filtered_df[date_column], errors="coerce")
    
    # Apply start_date filter if provided
    if start_date is not None:
        filtered_df = filtered_df[filtered_df[date_column] >= start_date]
    
    # Apply end_date filter if provided
    if end_date is not None:
        filtered_df = filtered_df[filtered_df[date_column] <= end_date]
    
    return filtered_df 
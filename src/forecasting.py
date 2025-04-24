"""
Module for forecasting methods.
"""

import pandas as pd
import re
from typing import Dict, Any, Optional, Tuple

# Import forecasting libraries as needed, e.g.:
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Import the specific forecasting calculation functions
from .analysis.arima_forecast import calculate_arima
from .analysis.ses_forecast import calculate_ses
from .analysis.sma_forecast import calculate_sma

# Define default column names, can be overridden if needed
DEFAULT_SKU_COL = 'sku' # Corrected from 'Material Mapeado'
DEFAULT_DATE_COL = 'Period' # Assumed correct as it's generated
DEFAULT_VALUE_COL = 'sales_value' # Corrected from 'Valor Faturado Liquido'

def apply_arima(series: pd.Series, order: tuple):
    """Applies ARIMA model to a time series."""
    # TODO: Implement ARIMA logic
    print(f"Applying ARIMA with order {order} to series (length {len(series)})...")
    # Example:
    # model = ARIMA(series, order=order)
    # model_fit = model.fit()
    # forecast = model_fit.predict(start=len(series), end=len(series) + 11) # Forecast next 12 steps
    # return forecast
    return None

def apply_ses(series: pd.Series, seasonal_periods: int):
    """Applies Simple Exponential Smoothing (SES) to a time series."""
    # TODO: Implement SES logic
    print(f"Applying SES with seasonal_periods={seasonal_periods} to series (length {len(series)})...")
    # Example:
    # model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=seasonal_periods)
    # model_fit = model.fit()
    # forecast = model_fit.predict(start=len(series), end=len(series) + 11)
    # return forecast
    return None

def _parse_method_string(method_str: str) -> Tuple[Optional[str], Dict[str, Any]]:
    """Parses the method string from the UI to extract method name and parameters."""
    params = {}
    method_name = None

    # ARIMA pattern: e.g., "ARIMA (p,d,q)"
    arima_match = re.match(r'ARIMA\s*\((\d+,\d+,\d+)\)', method_str, re.IGNORECASE)
    if arima_match:
        method_name = 'ARIMA'
        try:
            # Extract the tuple string and convert to tuple of ints
            order_str = arima_match.group(1)
            params['order'] = tuple(map(int, order_str.split(',')))
        except Exception:
            print(f"Error parsing ARIMA order from string: {method_str}")
            return None, {} # Return None if parsing fails
        return method_name, params

    # SES pattern: e.g., "SES (alpha=0.1)"
    ses_match = re.match(r'SES\s*\(alpha=(\d+(\.\d+)?)\)', method_str, re.IGNORECASE)
    if ses_match:
        method_name = 'SES'
        try:
            params['alpha'] = float(ses_match.group(1))
        except Exception:
            print(f"Error parsing SES alpha from string: {method_str}")
            return None, {} # Return None if parsing fails
        return method_name, params

    # SMA pattern: e.g., "SMA (N=3)"
    sma_match = re.match(r'SMA\s*\(N=(\d+)\)', method_str, re.IGNORECASE)
    if sma_match:
        method_name = 'SMA'
        try:
            params['n'] = int(sma_match.group(1))
        except Exception:
            print(f"Error parsing SMA N from string: {method_str}")
            return None, {} # Return None if parsing fails
        return method_name, params

    print(f"Warning: Could not parse method string: {method_str}")
    return None, {} # Return None if no match

def generate_forecasts(
    filtered_df: pd.DataFrame,
    selected_method_str: str,
    start_period: pd.Period,
    end_period: pd.Period,
    sku_col: str = DEFAULT_SKU_COL,
    date_col: str = DEFAULT_DATE_COL,
    value_col: str = DEFAULT_VALUE_COL
) -> pd.DataFrame:
    """
    Generates forecasts for selected SKUs using the chosen method and horizon.

    Args:
        filtered_df (pd.DataFrame): DataFrame pre-filtered for the relevant SKUs.
                                     Must contain sku_col, date_col, value_col.
        selected_method_str (str): The forecasting method string from the UI.
        start_period (pd.Period): The first period in the forecast horizon.
        end_period (pd.Period): The last period in the forecast horizon.
        sku_col (str): Name of the SKU identifier column.
        date_col (str): Name of the Period column (e.g., 'Period').
        value_col (str): Name of the column containing the sales value.

    Returns:
        pd.DataFrame: A DataFrame with columns [sku_col, date_col, 'Valor Previsto'],
                      containing the forecasts for each SKU within the horizon.
                      Returns an empty DataFrame if forecasting fails or no SKUs exist.
    """
    method_name, params = _parse_method_string(selected_method_str)

    if not method_name:
        print(f"Failed to identify forecasting method from: {selected_method_str}")
        return pd.DataFrame(columns=[sku_col, date_col, 'Valor Previsto'])

    if sku_col not in filtered_df.columns or date_col not in filtered_df.columns or value_col not in filtered_df.columns:
        print(f"Error: Required columns ({sku_col}, {date_col}, {value_col}) not found in DataFrame.")
        return pd.DataFrame(columns=[sku_col, date_col, 'Valor Previsto'])

    # Create the forecast horizon (index for the forecast output)
    try:
        forecast_horizon = pd.period_range(start=start_period, end=end_period, freq='M')
    except Exception as e:
        print(f"Error creating forecast horizon: {e}")
        return pd.DataFrame(columns=[sku_col, date_col, 'Valor Previsto'])

    all_forecasts = []
    unique_skus = filtered_df[sku_col].unique()

    print(f"Starting forecast generation for {len(unique_skus)} SKUs using {method_name} with params {params}...")

    for i, sku in enumerate(unique_skus):
        if (i + 1) % 50 == 0: # Print progress periodically
            print(f"Processing SKU {i+1}/{len(unique_skus)}: {sku}")

        # Extract time series for the current SKU
        sku_data = filtered_df[filtered_df[sku_col] == sku]
        
        # Aggregate sales per period for the SKU
        sku_series = sku_data.groupby(date_col)[value_col].sum()

        # Ensure data is sorted by period (index is now Period after groupby)
        sku_series = sku_series.sort_index()

        # Ensure the index is a PeriodIndex (should be after groupby, but good practice)
        if not isinstance(sku_series.index, pd.PeriodIndex):
            # This might happen if date_col wasn't Period type before groupby
            # Try converting
            try:
                sku_series.index = pd.to_datetime(sku_series.index.astype(str)).to_period('M')
            except Exception as idx_e:
                print(f"Warning: Could not convert index to PeriodIndex for SKU {sku}: {idx_e}")
                continue # Skip this SKU if index conversion fails

        # Reindex to ensure all periods are present (fill missing with 0 for calculation)
        # Limit reindexing to the historical data range for this SKU + forecast horizon
        # This prevents excessive memory usage if data is sparse over long time
        min_hist_period = sku_series.index.min()
        full_range_index = pd.period_range(start=min_hist_period, end=end_period, freq='M')
        sku_series = sku_series.reindex(full_range_index, fill_value=0)

        forecast_result = None
        try:
            if method_name == 'ARIMA':
                forecast_result = calculate_arima(sku_series, params['order'], forecast_horizon)
            elif method_name == 'SES':
                forecast_result = calculate_ses(sku_series, params['alpha'], forecast_horizon)
            elif method_name == 'SMA':
                forecast_result = calculate_sma(sku_series, params['n'], forecast_horizon)

            if forecast_result is not None and not forecast_result.isna().all():
                # Convert the result Series to a DataFrame and add SKU column
                forecast_df = forecast_result.reset_index()
                forecast_df.columns = [date_col, 'Valor Previsto']
                forecast_df[sku_col] = sku
                all_forecasts.append(forecast_df[[sku_col, date_col, 'Valor Previsto']])
            # else: # Optional: Log if a forecast is empty/NaN
                # print(f"Note: Forecast for SKU {sku} resulted in all NaNs or was None.")

        except Exception as e:
            print(f"Error forecasting SKU {sku} with {method_name}: {e}")
            # Continue to the next SKU

    print("Finished forecast generation. Aggregating results...")

    if not all_forecasts:
        print("No successful forecasts were generated.")
        return pd.DataFrame(columns=[sku_col, date_col, 'Valor Previsto'])

    # Concatenate all individual forecasts
    final_forecast_df = pd.concat(all_forecasts, ignore_index=True)

    # Ensure correct data types
    final_forecast_df[date_col] = pd.to_datetime(final_forecast_df[date_col].astype(str)).dt.to_period('M')
    final_forecast_df['Valor Previsto'] = pd.to_numeric(final_forecast_df['Valor Previsto'], errors='coerce')

    print(f"Generated forecast DataFrame with shape: {final_forecast_df.shape}")

    return final_forecast_df

# Add other forecasting methods as needed... 
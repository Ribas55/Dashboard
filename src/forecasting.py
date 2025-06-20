"""
Module for forecasting methods.
"""

import pandas as pd
import re
from typing import Dict, Any, Optional, Tuple, List
import os # Added for budget file path

# Import forecasting libraries as needed, e.g.:
# from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Import the specific forecasting calculation functions
from .analysis.arima_forecast import calculate_arima
from .analysis.ses_forecast import calculate_ses
from .analysis.sma_forecast import calculate_sma
from .analysis.tsb_forecast import calculate_tsb
# Import the XGBoost forecasting function
from .analysis.xgboost_forecast import calculate_xgboost
# Import the Linear Regression forecasting function
from .analysis.linear_regression_forecast import calculate_linear_regression
# Import the new SKU ponderation function
from .analysis.sku_ponderation import calculate_sku_ponderation
# Import budget loader (assuming it exists and handles errors)
from .data_loader import load_budget_data

# Define default column names, can be overridden if needed
DEFAULT_SKU_COL = 'sku' # Corrected from 'Material Mapeado'
DEFAULT_DATE_COL = 'Period' # Assumed correct as it's generated
DEFAULT_VALUE_COL = 'sales_value' # Corrected from 'Valor Faturado Liquido'
DEFAULT_FAMILY_COL = 'family' # Added default family column
DEFAULT_MANAGER_COL = 'commercial_manager' # Added default manager column
DEFAULT_BUDGET_PATH = os.path.join('data', 'Orçamento2022-2025.xlsx') # Added default budget path

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

    # TSB pattern: e.g., "TSB (alpha_d=0.1, alpha_p=0.2)"
    # Allows for spaces around commas and equals signs
    tsb_match = re.match(r'TSB\s*\(alpha_d\s*=\s*(\d+(\.\d+)?)\s*,\s*alpha_p\s*=\s*(\d+(\.\d+)?)\)', method_str, re.IGNORECASE)
    if tsb_match:
        method_name = 'TSB'
        try:
            params['alpha_d'] = float(tsb_match.group(1))
            params['alpha_p'] = float(tsb_match.group(3)) # Group 3 captures the second float
        except Exception as e:
            print(f"Error parsing TSB parameters from string: {method_str}. Error: {e}")
            return None, {} # Return None if parsing fails
        return method_name, params

    # XGBoost pattern: e.g., "XGBoost (lags=3)"
    xgboost_match = re.match(r'XGBoost\s*\(lags=(\d+)\)', method_str, re.IGNORECASE)
    if xgboost_match:
        method_name = 'XGBoost'
        try:
            params['n_lags'] = int(xgboost_match.group(1))
        except Exception:
            print(f"Error parsing XGBoost lags from string: {method_str}")
            return None, {} # Return None if parsing fails
        return method_name, params

    # Linear Regression pattern: e.g., "Linear Regression (lags=3)"
    linear_regression_match = re.match(r'Linear\s*Regression\s*\(lags=(\d+)\)', method_str, re.IGNORECASE)
    if linear_regression_match:
        method_name = 'LinearRegression'
        try:
            params['n_lags'] = int(linear_regression_match.group(1))
        except Exception:
            print(f"Error parsing Linear Regression lags from string: {method_str}")
            return None, {} # Return None if parsing fails
        return method_name, params

    # Check for SKU Ponderation explicitly (no parameters in the string)
    if method_str == "Previsão Ponderada por SKU":
        method_name = 'PonderadaSKU' # Internal identifier
        return method_name, params # No parameters needed from string

    print(f"Warning: Could not parse method string: {method_str}")
    return None, {} # Return None if no match

# --- Helper Function to Load Budget Data ---
# Cache decorator could be added if this function is called multiple times
# with the same path in a single run, but likely not needed here as it's
# called once per generate_forecasts run if method is PonderadaSKU.
def _load_budget_for_forecast(budget_path: str = DEFAULT_BUDGET_PATH) -> Optional[pd.DataFrame]:
    """Loads budget data from the specified path, handling errors."""
    try:
        print(f"Attempting to load budget data from: {budget_path}")
        # Assuming load_budget_data returns None on error or if file not found
        df_budget = load_budget_data(budget_path)
        if df_budget is None:
            print("Warning: load_budget_data returned None. Budget data might be missing or invalid.")
            return None
        elif df_budget.empty:
            print("Warning: Budget data file loaded but is empty.")
            return None
        else:
            print("Budget data loaded successfully for forecasting.")
            return df_budget
    except FileNotFoundError:
        print(f"Warning: Budget file not found at {budget_path}. Proceeding without budget data.")
        return None
    except Exception as e:
        print(f"Error loading budget data from {budget_path}: {e}")
        return None

def generate_forecasts(
    filtered_df: pd.DataFrame,
    selected_method_str: str,
    start_period: pd.Period,
    end_period: pd.Period,
    sku_col: str = DEFAULT_SKU_COL,
    date_col: str = DEFAULT_DATE_COL,
    value_col: str = DEFAULT_VALUE_COL,
    family_col: str = DEFAULT_FAMILY_COL,    # Added family col
    manager_col: str = DEFAULT_MANAGER_COL, # Added manager col
    weights: Optional[Dict[str, float]] = None, # Added optional weights arg
    target_skus: Optional[List[str]] = None # Added: Specific SKUs to forecast for
) -> pd.DataFrame:
    """
    Generates forecasts for selected SKUs using the chosen method and horizon.

    Args:
        filtered_df (pd.DataFrame): DataFrame pre-filtered for the relevant SKUs.
                                     Must contain sku_col, date_col, value_col.
                                     For 'PonderadaSKU', must also contain family_col, manager_col.
        selected_method_str (str): The forecasting method string from the UI.
        start_period (pd.Period): The first period in the forecast horizon.
        end_period (pd.Period): The last period in the forecast horizon.
        sku_col (str): Name of the SKU identifier column.
        date_col (str): Name of the Period column (e.g., 'Period').
        value_col (str): Name of the column containing the sales value.
        family_col (str): Name of the product family column.
        manager_col (str): Name of the commercial manager column.
        weights (Optional[Dict[str, float]]): Dictionary of weights (0.0-1.0)
                                             required for 'PonderadaSKU'.
        target_skus (Optional[List[str]]): If provided, calculate forecast only for
                                         these specific SKUs (used by PonderadaSKU
                                         to filter final output).

    Returns:
        pd.DataFrame: A DataFrame with columns [sku_col, date_col, 'Valor Previsto'],
                      containing the forecasts for each SKU within the horizon.
                      Returns an empty DataFrame if forecasting fails or no SKUs exist.
    """
    method_name, params = _parse_method_string(selected_method_str)

    if not method_name:
        print(f"Failed to identify forecasting method from: {selected_method_str}")
        return pd.DataFrame(columns=[sku_col, date_col, 'Valor Previsto'])

    # --- Validate Required Columns Based on Method ---
    required_cols = [sku_col, date_col, value_col]
    if method_name == 'PonderadaSKU':
        required_cols.extend([family_col, manager_col])
        if weights is None:
            print(f"Error: Weights dictionary is required for method '{selected_method_str}' but was not provided.")
            return pd.DataFrame(columns=[sku_col, date_col, 'Valor Previsto'])
        # PonderadaSKU also needs target_skus to know which ones to return
        if target_skus is None or not target_skus:
             print(f"Error: target_skus list is required for method '{selected_method_str}' but was not provided or is empty.")
             return pd.DataFrame(columns=[sku_col, date_col, 'Valor Previsto'])

    if not all(col in filtered_df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in filtered_df.columns]
        print(f"Error: Required columns ({missing_cols}) for method '{selected_method_str}' not found in DataFrame.")
        return pd.DataFrame(columns=[sku_col, date_col, 'Valor Previsto'])

    # --- Load Budget Data (only if needed) ---
    df_budget = None
    if method_name == 'PonderadaSKU':
        df_budget = _load_budget_for_forecast() # Uses default path
        # The calculate_sku_ponderation function handles the case where df_budget is None

    # Create the forecast horizon (index for the forecast output)
    try:
        forecast_horizon = pd.period_range(start=start_period, end=end_period, freq='M')
    except Exception as e:
        print(f"Error creating forecast horizon: {e}")
        return pd.DataFrame(columns=[sku_col, date_col, 'Valor Previsto'])

    all_forecasts = []
    unique_skus = filtered_df[sku_col].unique()

    print(f"Starting forecast generation for {len(unique_skus)} SKUs using {method_name} with params {params}...")

    # --- Main Processing Logic ---
    # For PonderadaSKU, we process all SKUs together
    if method_name == 'PonderadaSKU':
        try:
            print("Calculating SKU Ponderation forecast for all selected SKUs...")
            # Ensure weights are not None before calling (validated earlier)
            assert weights is not None, "Weights should not be None for PonderadaSKU"
            # Ensure target_skus is not None before calling (validated earlier)
            assert target_skus is not None, "Target SKUs should not be None for PonderadaSKU"

            forecast_result_df = calculate_sku_ponderation(
                df_sales_hist=filtered_df, # Pass the entire filtered df
                df_budget=df_budget,
                weights=weights,
                forecast_horizon=forecast_horizon,
                sku_col=sku_col,
                date_col=date_col,
                value_col=value_col,
                family_col=family_col,
                manager_col=manager_col,
                target_skus=target_skus # Pass the target SKUs list
            )
            if forecast_result_df is not None and not forecast_result_df.empty:
                # <<< Ensure the result only contains the target SKUs >>>
                # <<< Ensure consistent string types for filtering >>>
                target_skus_str = [str(s) for s in target_skus]
                forecast_result_df[sku_col] = forecast_result_df[sku_col].astype(str)

                forecast_result_df = forecast_result_df[forecast_result_df[sku_col].isin(target_skus_str)]
                all_forecasts.append(forecast_result_df)
            else:
                print("Warning: calculate_sku_ponderation returned empty or None DataFrame.")

        except Exception as e:
            print(f"Error during PonderadaSKU calculation: {e}")
            # If the combined calculation fails, return empty
            return pd.DataFrame(columns=[sku_col, date_col, 'Valor Previsto'])

    # For other methods, process SKU by SKU
    else:
        for i, sku in enumerate(unique_skus):
            if (i + 1) % 50 == 0: # Print progress periodically
                print(f"Processing SKU {i+1}/{len(unique_skus)}: {sku}")

            # Extract time series for the current SKU
            sku_data = filtered_df[filtered_df[sku_col] == sku]

            # Aggregate sales per period for the SKU
            sku_series = sku_data.groupby(date_col)[value_col].sum()

            # Ensure data is sorted by period (index is now Period after groupby)
            sku_series = sku_series.sort_index()

            # Ensure the index is a PeriodIndex
            if not isinstance(sku_series.index, pd.PeriodIndex):
                try:
                    sku_series.index = pd.to_datetime(sku_series.index.astype(str)).to_period('M')
                except Exception as idx_e:
                    print(f"Warning: Could not convert index to PeriodIndex for SKU {sku}: {idx_e}")
                    continue # Skip this SKU

            # Reindex to ensure all periods are present (fill missing with 0)
            # Use the range from the SKU's actual data min to forecast end
            try:
                min_hist_period = sku_series.index.min()
                full_range_index = pd.period_range(start=min_hist_period, end=forecast_horizon.max(), freq='M')
                sku_series = sku_series.reindex(full_range_index, fill_value=0)
            except Exception as reindex_e:
                 print(f"Warning: Could not reindex series for SKU {sku}: {reindex_e}")
                 continue # Skip this SKU

            forecast_result = None
            try:
                if method_name == 'ARIMA':
                    forecast_result = calculate_arima(sku_series, params['order'], forecast_horizon)
                elif method_name == 'SES':
                    forecast_result = calculate_ses(sku_series, params['alpha'], forecast_horizon)
                elif method_name == 'SMA':
                    forecast_result = calculate_sma(sku_series, params['n'], forecast_horizon)
                elif method_name == 'TSB':
                    forecast_result = calculate_tsb(sku_series, params['alpha_d'], params['alpha_p'], forecast_horizon)
                elif method_name == 'XGBoost':
                    forecast_result = calculate_xgboost(sku_series, params['n_lags'], forecast_horizon)
                elif method_name == 'LinearRegression':
                    forecast_result = calculate_linear_regression(sku_series, params['n_lags'], forecast_horizon)
                # PonderadaSKU is handled outside this loop

                if forecast_result is not None and not forecast_result.empty and not forecast_result.isna().all():
                    # Convert the result Series to a DataFrame and add SKU column
                    forecast_df = forecast_result.reset_index()
                    forecast_df.columns = [date_col, 'Valor Previsto']
                    forecast_df[sku_col] = sku
                    all_forecasts.append(forecast_df[[sku_col, date_col, 'Valor Previsto']])
                # else: # Optional: Log if forecast is empty/NaN
                    # print(f"Note: Forecast for SKU {sku} resulted in all NaNs or was None/empty.")

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
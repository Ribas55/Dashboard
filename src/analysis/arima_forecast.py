"""
Module for ARIMA forecasting.
"""

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings

def calculate_arima(series: pd.Series, order: tuple, forecast_horizon: pd.PeriodIndex) -> pd.Series:
    """
    Calculates iterative ARIMA forecasts.

    Args:
        series (pd.Series): Input time series data indexed by period (e.g., monthly).
                               Must have enough data points for the specified order.
        order (tuple): The (p, d, q) order of the ARIMA model.
        forecast_horizon (pd.PeriodIndex): The periods for which to generate forecasts.

    Returns:
        pd.Series: A series containing the forecasted values, indexed by period.
                   Returns NaN for periods where forecasting failed.
    """
    forecasts = {}

    # Ensure the series index is a PeriodIndex for comparison
    if not isinstance(series.index, pd.PeriodIndex):
        series.index = pd.to_datetime(series.index).to_period('M')

    # Convert PeriodIndex to DatetimeIndex for statsmodels compatibility
    series_dt_index = series.copy()
    if isinstance(series.index, pd.PeriodIndex):
        series_dt_index.index = series.index.to_timestamp()

    # Set minimum observations needed to 2
    min_obs_needed = 2

    for target_period in forecast_horizon:
        target_timestamp = target_period.to_timestamp()
        # Use original series for the zero check
        training_data_for_check = series[series.index < target_period]

        # --- New Check ---
        if len(training_data_for_check) >= 2 and training_data_for_check.iloc[-1] == 0 and training_data_for_check.iloc[-2] == 0:
            forecasts[target_period] = 0
            continue # Skip ARIMA calculation for this period
        # --- End New Check ---

        # Prepare data with DatetimeIndex for ARIMA model
        training_data = series_dt_index[series_dt_index.index < target_timestamp]

        if len(training_data) >= min_obs_needed:
            try:
                with warnings.catch_warnings():
                    # Suppress common warnings during ARIMA fitting
                    warnings.simplefilter("ignore")
                    # Use enforce_stationarity=False and enforce_invertibility=False
                    # for robustness, as finding optimal parameters is hard automatically.
                    model = ARIMA(training_data, order=order, enforce_stationarity=False, enforce_invertibility=False)
                    fit = model.fit()
                    # Forecast 1 step ahead
                    forecast_value = fit.forecast(1).iloc[0]
                    # --- Ensure forecast is not negative ---
                    if forecast_value < 0:
                        forecast_value = 0
                    # --- End Ensure non-negative ---
                    forecasts[target_period] = forecast_value
            except Exception as e:
                # Handle potential errors (e.g., convergence, matrix singularity)
                print(f"Warning: ARIMA {order} failed for SKU (period {target_period}). Error: {e}")
                forecasts[target_period] = pd.NA
        else:
            # Not enough data
            # print(f"Warning: Insufficient data ({len(training_data)}<{min_obs_needed}) for ARIMA {order} for SKU (period {target_period})")
            forecasts[target_period] = pd.NA

    return pd.Series(forecasts, index=forecast_horizon, name="arima_forecast") 
"""
Module for Simple Moving Average (SMA) forecasting.
"""

import pandas as pd

def calculate_sma(series: pd.Series, n: int, forecast_horizon: pd.PeriodIndex) -> pd.Series:
    """
    Calculates iterative Simple Moving Average forecasts.

    Args:
        series (pd.Series): Input time series data indexed by period (e.g., monthly).
        n (int): The window size for the moving average.
        forecast_horizon (pd.PeriodIndex): The periods for which to generate forecasts.

    Returns:
        pd.Series: A series containing the forecasted values, indexed by period.
    """
    forecasts = {}

    # Ensure the series index is a PeriodIndex for comparison
    if not isinstance(series.index, pd.PeriodIndex):
        series.index = pd.to_datetime(series.index).to_period('M')

    for target_period in forecast_horizon:
        # Data available up to the period *before* the target period
        training_data = series[series.index < target_period]

        if len(training_data) >= 2 and training_data.iloc[-1] == 0 and training_data.iloc[-2] == 0:
            forecasts[target_period] = 0
            continue # Skip SMA calculation for this period

        if len(training_data) >= n:
            # Calculate the average of the last N periods
            forecast_value = training_data.iloc[-n:].mean()
            # --- Ensure forecast is not negative ---
            if forecast_value < 0:
                forecast_value = 0
            # --- End Ensure non-negative ---
            forecasts[target_period] = forecast_value
        else:
            # Not enough data for SMA, forecast is NaN or 0?
            # For simplicity, let's use NaN, can be adjusted.
            forecasts[target_period] = pd.NA

    return pd.Series(forecasts, index=forecast_horizon, name="sma_forecast") 
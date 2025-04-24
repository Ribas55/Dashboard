"""
Module for Simple Exponential Smoothing (SES) forecasting.
"""

import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import warnings

def calculate_ses(series: pd.Series, alpha: float, forecast_horizon: pd.PeriodIndex) -> pd.Series:
    """
    Calculates iterative Simple Exponential Smoothing (SES) forecasts.

    Args:
        series (pd.Series): Input time series data indexed by period (e.g., monthly).
                               Must contain at least 2 data points.
        alpha (float): The smoothing level parameter (0 < alpha <= 1).
        forecast_horizon (pd.PeriodIndex): The periods for which to generate forecasts.

    Returns:
        pd.Series: A series containing the forecasted values, indexed by period.
                   Returns NaN for periods where forecasting failed (e.g., insufficient data).
    """
    forecasts = {}

    # Ensure the series index is a PeriodIndex for comparison
    if not isinstance(series.index, pd.PeriodIndex):
        series.index = pd.to_datetime(series.index).to_period('M')

    # Convert PeriodIndex to DatetimeIndex for statsmodels compatibility if needed
    series_dt_index = series.copy()
    if isinstance(series.index, pd.PeriodIndex):
         # Keep original periods for indexing forecasts
        #original_periods = series.index # Not strictly needed here
        series_dt_index.index = series.index.to_timestamp()


    for target_period in forecast_horizon:
        # Convert target_period to timestamp to compare with series_dt_index
        target_timestamp = target_period.to_timestamp()

        # Use original series for the zero check
        training_data_for_check = series[series.index < target_period]

        # --- New Check ---
        if len(training_data_for_check) >= 2 and training_data_for_check.iloc[-1] == 0 and training_data_for_check.iloc[-2] == 0:
            forecasts[target_period] = 0
            continue # Skip SES calculation for this period
        # --- End New Check ---

        # Data available up to the period *before* the target period (for SES model)
        training_data = series_dt_index[series_dt_index.index < target_timestamp]

        # SES requires at least 2 data points to estimate initial level
        if len(training_data) >= 2:
            try:
                with warnings.catch_warnings():
                    # Suppress expected ConvergenceWarning & other potential warnings
                    warnings.simplefilter("ignore")
                    model = SimpleExpSmoothing(training_data, initialization_method='estimated')
                    # Use alpha directly as smoothing_level
                    fit = model.fit(smoothing_level=alpha, optimized=False)
                    # Forecast 1 step ahead from the end of training data
                    forecast_value = fit.forecast(1).iloc[0]
                    # --- Ensure forecast is not negative ---
                    if forecast_value < 0:
                        forecast_value = 0
                    # --- End Ensure non-negative ---
                    forecasts[target_period] = forecast_value
            except Exception as e:
                # Handle potential errors during fitting/forecasting
                print(f"Warning: SES failed for SKU (period {target_period}). Error: {e}") # Added SKU context placeholder
                forecasts[target_period] = pd.NA
        else:
            # Not enough data to fit SES model
            # print(f"Warning: Insufficient data for SES for SKU (period {target_period})") # Optional warning
            forecasts[target_period] = pd.NA

    return pd.Series(forecasts, index=forecast_horizon, name="ses_forecast") 
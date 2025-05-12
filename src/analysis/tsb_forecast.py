"""
Module for Teunter-Syntetos-Babai (TSB) forecasting for intermittent demand.
"""

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import TSB
import warnings
import numpy as np

def calculate_tsb(series: pd.Series, alpha_d: float, alpha_p: float, forecast_horizon: pd.PeriodIndex) -> pd.Series:
    """
    Calculates iterative TSB forecasts using the statsforecast library.

    Args:
        series (pd.Series): Input time series data indexed by period (monthly).
                           Must contain at least 2 data points.
        alpha_d (float): Smoothing parameter for demand level.
        alpha_p (float): Smoothing parameter for demand probability.
        forecast_horizon (pd.PeriodIndex): The periods for which to generate forecasts.

    Returns:
        pd.Series: A series containing the forecasted values, indexed by period.
                   Returns NaN for periods where forecasting failed.
    """
    forecasts = {}
    # Use a fixed unique_id since we process one series at a time
    unique_id = "sku"

    # --- Data Preparation for StatsForecast --- 
    # Ensure index is PeriodIndex
    if not isinstance(series.index, pd.PeriodIndex):
        try:
            series.index = pd.to_datetime(series.index.astype(str)).to_period('M')
        except Exception as e:
            print(f"Warning: Could not convert index to PeriodIndex for TSB. Error: {e}")
            # Return NaNs for the entire horizon if index conversion fails
            return pd.Series([pd.NA] * len(forecast_horizon), index=forecast_horizon, name="tsb_forecast")

    # Drop NA values before processing, but keep track of original horizon
    series_cleaned = series.dropna()

    # Check for the zero condition (last two observed points are zero)
    if len(series_cleaned) >= 2 and series_cleaned.iloc[-1] == 0 and series_cleaned.iloc[-2] == 0:
        # If last two are zero, TSB typically forecasts zero
        return pd.Series(0, index=forecast_horizon, name="tsb_forecast")

    # Convert to DataFrame format required by StatsForecast: [unique_id, ds, y]
    df = series_cleaned.reset_index()
    df.columns = ['ds', 'y']
    df['unique_id'] = unique_id
    # Convert PeriodIndex 'ds' to Timestamps (start of the month)
    df['ds'] = df['ds'].dt.to_timestamp()

    # TSB might need a minimum number of points, let's assume 2 for basic run
    min_obs_needed = 2
    if len(df) < min_obs_needed:
        print(f"Warning: Insufficient data ({len(df)}<{min_obs_needed}) for TSB.")
        return pd.Series([pd.NA] * len(forecast_horizon), index=forecast_horizon, name="tsb_forecast")

    # --- TSB Model Fitting and Prediction --- 
    try:
        # Define the model
        model = StatsForecast(
            models=[TSB(alpha_d=alpha_d, alpha_p=alpha_p)],
            freq='MS', # 'MS' for Month Start frequency, matching the timestamp conversion
            n_jobs=1 # Run sequentially for safety within iterative calls
        )

        # Fit the model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore") # Suppress potential statsforecast warnings
            model.fit(df)

        # Determine forecast horizon length
        h = len(forecast_horizon)

        # Predict
        forecast_df = model.predict(h=h)

        # Check the type and emptiness
        if not isinstance(forecast_df, pd.DataFrame) or forecast_df.empty:
             print(f"Warning: TSB prediction did not return a valid Pandas DataFrame.")
             return pd.Series([pd.NA] * len(forecast_horizon), index=forecast_horizon, name="tsb_forecast")

        # Extract the forecast values using pandas column access
        if 'TSB' not in forecast_df.columns:
            print(f"Warning: 'TSB' column not found in prediction results.")
            return pd.Series([pd.NA] * len(forecast_horizon), index=forecast_horizon, name="tsb_forecast")

        predicted_values_pd = forecast_df['TSB']
        # Convert to NumPy array for numerical operations, ensuring float type
        predicted_values = predicted_values_pd.to_numpy(dtype=float, na_value=np.nan) # Use np.nan for clarity

        # Ensure forecasts are not negative (operating on NumPy array)
        # Use np.nan_to_num to handle potential NaNs before comparison
        predicted_values = np.nan_to_num(predicted_values, nan=0.0) # Replace NaN with 0 before comparison
        predicted_values[predicted_values < 0] = 0

        # Create the result Series with the original PeriodIndex horizon
        results = pd.Series(predicted_values, index=forecast_horizon, name="tsb_forecast")
        return results
    except Exception as e:
        print(f"Error during TSB calculation for SKU. Error: {e}")
        # Return NaNs if any error occurs
        return pd.Series([pd.NA] * len(forecast_horizon), index=forecast_horizon, name="tsb_forecast") 
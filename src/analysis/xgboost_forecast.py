"""
Module for XGBoost forecasting.
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import warnings
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

def calculate_xgboost(series: pd.Series, n_lags: int, forecast_horizon: pd.PeriodIndex) -> pd.Series:
    """
    Calculates iterative XGBoost forecasts using lagged features.

    Args:
        series (pd.Series): Input time series data indexed by period (e.g., monthly).
                            Must have enough data points for the specified lags.
        n_lags (int): Number of lagged values to use as features.
        forecast_horizon (pd.PeriodIndex): The periods for which to generate forecasts.

    Returns:
        pd.Series: A series containing the forecasted values, indexed by period.
                   Returns NaN for periods where forecasting failed.
    """
    forecasts = {}

    # Ensure the series index is a PeriodIndex for comparison
    if not isinstance(series.index, pd.PeriodIndex):
        series.index = pd.to_datetime(series.index).to_period('M')

    # Min observations needed for training is n_lags + 1
    min_obs_needed = n_lags + 1

    # Create a DataFrame with lagged features from the series
    def create_features(ts: pd.Series, n_lags: int) -> pd.DataFrame:
        df = pd.DataFrame({'y': ts})
        for i in range(1, n_lags + 1):
            df[f'lag_{i}'] = df['y'].shift(i)
        return df.dropna()

    # Get the last n values for prediction
    def get_last_n_values(ts: pd.Series, n: int) -> np.ndarray:
        values = ts.values[-n:]
        return values[::-1]  # Reverse to match the lag ordering

    for target_period in forecast_horizon:
        # Use all data prior to the target period
        training_data = series[series.index < target_period]

        if len(training_data) >= min_obs_needed:
            try:
                # Check for consecutive zeros
                if len(training_data) >= 2 and training_data.iloc[-1] == 0 and training_data.iloc[-2] == 0:
                    forecasts[target_period] = 0
                    continue

                # Create features dataset
                features_df = create_features(training_data, n_lags)
                
                if features_df.empty:
                    forecasts[target_period] = pd.NA
                    continue
                
                # Split into X and y
                X = features_df.drop('y', axis=1)
                y = features_df['y']
                
                # Normalize features for better performance
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Train XGBoost model
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = XGBRegressor(
                        n_estimators=100, 
                        learning_rate=0.1,
                        max_depth=3,
                        random_state=42
                    )
                    model.fit(X_scaled, y)
                
                # Prepare input for prediction
                last_values = get_last_n_values(training_data, n_lags)
                if len(last_values) < n_lags:
                    # Pad with zeros if not enough values
                    last_values = np.pad(last_values, (n_lags - len(last_values), 0), 'constant')
                
                # Scale input
                last_values_scaled = scaler.transform(last_values.reshape(1, -1))
                
                # Predict
                forecast_value = model.predict(last_values_scaled)[0]
                
                # Ensure forecast is not negative
                if forecast_value < 0:
                    forecast_value = 0
                
                forecasts[target_period] = forecast_value
                
            except Exception as e:
                print(f"Warning: XGBoost failed for period {target_period}. Error: {e}")
                forecasts[target_period] = pd.NA
        else:
            # Not enough data
            forecasts[target_period] = pd.NA

    return pd.Series(forecasts, index=forecast_horizon, name="xgboost_forecast") 
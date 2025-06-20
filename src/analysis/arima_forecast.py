"""
Module for ARIMA forecasting with AIC optimization.
"""

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import warnings
import itertools
import numpy as np

def find_best_arima_order(series: pd.Series, max_p: int = 3, max_d: int = 2, max_q: int = 3, 
                         seasonal: bool = False, max_P: int = 2, max_D: int = 1, max_Q: int = 2, 
                         s: int = 12) -> tuple:
    """
    Find the best ARIMA order using AIC optimization.
    
    Args:
        series (pd.Series): Input time series data
        max_p (int): Maximum value for p parameter
        max_d (int): Maximum value for d parameter  
        max_q (int): Maximum value for q parameter
        seasonal (bool): Whether to include seasonal parameters
        max_P (int): Maximum value for seasonal P parameter
        max_D (int): Maximum value for seasonal D parameter
        max_Q (int): Maximum value for seasonal Q parameter
        s (int): Seasonal period (12 for monthly data)
    
    Returns:
        tuple: Best order (p,d,q) or ((p,d,q), (P,D,Q,s)) if seasonal
    """
    best_aic = np.inf
    best_order = None
    best_seasonal_order = None
    
    # Generate parameter combinations
    if seasonal:
        # For seasonal ARIMA
        p_values = range(0, max_p + 1)
        d_values = range(0, max_d + 1)
        q_values = range(0, max_q + 1)
        P_values = range(0, max_P + 1)
        D_values = range(0, max_D + 1)
        Q_values = range(0, max_Q + 1)
        
        for p, d, q in itertools.product(p_values, d_values, q_values):
            for P, D, Q in itertools.product(P_values, D_values, Q_values):
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        model = ARIMA(series, order=(p, d, q), 
                                    seasonal_order=(P, D, Q, s),
                                    enforce_stationarity=False, 
                                    enforce_invertibility=False)
                        fit = model.fit()
                        
                        if fit.aic < best_aic:
                            best_aic = fit.aic
                            best_order = (p, d, q)
                            best_seasonal_order = (P, D, Q, s)
                            
                except Exception:
                    continue
        
        return (best_order, best_seasonal_order) if best_order else ((1, 1, 1), (0, 0, 0, s))
    
    else:
        # For non-seasonal ARIMA
        p_values = range(0, max_p + 1)
        d_values = range(0, max_d + 1)
        q_values = range(0, max_q + 1)
        
        for p, d, q in itertools.product(p_values, d_values, q_values):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    model = ARIMA(series, order=(p, d, q),
                                enforce_stationarity=False, 
                                enforce_invertibility=False)
                    fit = model.fit()
                    
                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_order = (p, d, q)
                        
            except Exception:
                continue
        
        return best_order if best_order else (1, 1, 1)

def calculate_arima_with_aic_optimization(series: pd.Series, forecast_horizon: pd.PeriodIndex,
                                        optimize_params: bool = True, default_order: tuple = (1, 1, 1),
                                        seasonal: bool = False, seasonal_period: int = 12) -> pd.Series:
    """
    Calculates iterative ARIMA forecasts with optional AIC optimization.

    Args:
        series (pd.Series): Input time series data indexed by period (e.g., monthly).
        forecast_horizon (pd.PeriodIndex): The periods for which to generate forecasts.
        optimize_params (bool): Whether to optimize ARIMA parameters using AIC.
        default_order (tuple): Default ARIMA order if optimization is disabled.
        seasonal (bool): Whether to use seasonal ARIMA.
        seasonal_period (int): Seasonal period (12 for monthly data).

    Returns:
        pd.Series: A series containing the forecasted values, indexed by period.
    """
    forecasts = {}

    # Ensure the series index is a PeriodIndex for comparison
    if not isinstance(series.index, pd.PeriodIndex):
        series.index = pd.to_datetime(series.index).to_period('M')

    # Convert PeriodIndex to DatetimeIndex for statsmodels compatibility
    series_dt_index = series.copy()
    if isinstance(series.index, pd.PeriodIndex):
        series_dt_index.index = series.index.to_timestamp()

    # Set minimum observations needed
    min_obs_needed = 10 if optimize_params else 2  # Need more data for optimization

    # If optimizing, find best parameters once using all available data
    best_order = None
    best_seasonal_order = None
    
    if optimize_params and len(series_dt_index) >= min_obs_needed:
        try:
            if seasonal:
                best_order, best_seasonal_order = find_best_arima_order(
                    series_dt_index, seasonal=True, s=seasonal_period
                )
            else:
                best_order = find_best_arima_order(series_dt_index, seasonal=False)
            print(f"Optimized ARIMA order: {best_order}" + 
                  (f", seasonal: {best_seasonal_order}" if seasonal else ""))
        except Exception as e:
            print(f"AIC optimization failed: {e}. Using default order.")
            best_order = default_order
            best_seasonal_order = (0, 0, 0, seasonal_period) if seasonal else None
    else:
        best_order = default_order
        best_seasonal_order = (0, 0, 0, seasonal_period) if seasonal else None

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
                    warnings.simplefilter("ignore")
                    
                    if seasonal and best_seasonal_order:
                        model = ARIMA(training_data, order=best_order, 
                                    seasonal_order=best_seasonal_order,
                                    enforce_stationarity=False, 
                                    enforce_invertibility=False)
                    else:
                        model = ARIMA(training_data, order=best_order, 
                                    enforce_stationarity=False, 
                                    enforce_invertibility=False)
                    
                    fit = model.fit()
                    forecast_value = fit.forecast(1).iloc[0]
                    
                    # Ensure forecast is not negative
                    if forecast_value < 0:
                        forecast_value = 0
                    
                    forecasts[target_period] = forecast_value
                    
            except Exception as e:
                print(f"Warning: ARIMA {best_order} failed for period {target_period}. Error: {e}")
                forecasts[target_period] = pd.NA
        else:
            forecasts[target_period] = pd.NA

    return pd.Series(forecasts, index=forecast_horizon, name="arima_forecast")

def calculate_arima(series: pd.Series, order: tuple, forecast_horizon: pd.PeriodIndex) -> pd.Series:
    """
    Calculates iterative ARIMA forecasts (original function for backward compatibility).

    Args:
        series (pd.Series): Input time series data indexed by period (e.g., monthly).
                               Must have enough data points for the specified order.
        order (tuple): The (p, d, q) order of the ARIMA model.
        forecast_horizon (pd.PeriodIndex): The periods for which to generate forecasts.

    Returns:
        pd.Series: A series containing the forecasted values, indexed by period.
                   Returns NaN for periods where forecasting failed.
    """
    return calculate_arima_with_aic_optimization(
        series=series, 
        forecast_horizon=forecast_horizon,
        optimize_params=False,
        default_order=order,
        seasonal=False
    ) 
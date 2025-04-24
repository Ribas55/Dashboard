# Placeholder for custom weighted forecast logic 

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from dateutil.relativedelta import relativedelta
import calendar # Added for month end calculation

def calculate_weighted_forecast(
    df_sales: pd.DataFrame,
    df_budget: Optional[pd.DataFrame],
    weights: Dict[str, float],
    forecast_start_date: pd.Timestamp,
    forecast_end_date: pd.Timestamp, # Changed from forecast_periods
    manager_filter: Optional[str] = None,
    family_filter: Optional[str] = None
) -> pd.DataFrame:
    """
    Calculates a weighted monthly sales forecast using a rolling window approach.

    Parameters:
    -----------
    df_sales : pd.DataFrame
        DataFrame containing historical sales data with columns like
        'invoice_date', 'commercial_manager', 'family', 'sales_value'.
        'invoice_date' MUST be datetime.
    df_budget : Optional[pd.DataFrame]
        DataFrame containing aggregated monthly budget data with columns like
        'commercial_manager', 'family', 'year', 'month', 'KG'. Can be None.
    weights : Dict[str, float]
        Dictionary containing the percentage weights for each component:
        'avg_2023', 'avg_ytd', 'last_3_months', 'last_12_months', 'budget'.
        Weights should sum to 1.0.
    forecast_start_date : pd.Timestamp
        The first month (start of the month) for which to generate a forecast.
    forecast_end_date : pd.Timestamp
        The last month (start of the month) for which to generate a forecast.
    manager_filter : Optional[str]
        Filter forecast by a specific commercial manager.
    family_filter : Optional[str]
        Filter forecast by a specific family.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing the forecast with columns:
        'commercial_manager', 'family', 'year', 'month', 'forecast_value'.
        Returns an empty DataFrame if input data is insufficient or filtered out.
    """
    # --- Input Validation and Preparation ---
    required_sales_cols = ['invoice_date', 'commercial_manager', 'family', 'sales_value']
    if not all(col in df_sales.columns for col in required_sales_cols):
        print(f"Error: Sales data missing required columns: {required_sales_cols}")
        return pd.DataFrame(columns=['commercial_manager', 'family', 'year', 'month', 'forecast_value'])
    if not pd.api.types.is_datetime64_any_dtype(df_sales['invoice_date']):
        print("Error: Sales data 'invoice_date' column must be datetime.")
        return pd.DataFrame(columns=['commercial_manager', 'family', 'year', 'month', 'forecast_value'])

    # Ensure weights sum to 1.0
    total_weight = sum(weights.values())
    if not np.isclose(total_weight, 1.0):
       print(f"Warning: Weights sum to {total_weight}, normalizing to 1.0")
       if total_weight != 0:
           weights = {k: v / total_weight for k, v in weights.items()}
       else:
           print("Error: Weights sum to zero, cannot normalize.")
           return pd.DataFrame(columns=['commercial_manager', 'family', 'year', 'month', 'forecast_value'])

    df_filtered = df_sales.copy()
    if manager_filter and manager_filter != "Todos":
        df_filtered = df_filtered[df_filtered['commercial_manager'] == manager_filter]
    if family_filter and family_filter != "Todas":
        df_filtered = df_filtered[df_filtered['family'] == family_filter]

    if df_filtered.empty:
        print("Warning: No sales data remains after filtering.")
        return pd.DataFrame(columns=['commercial_manager', 'family', 'year', 'month', 'forecast_value'])

    # Pre-aggregate sales data monthly for efficiency
    df_filtered['year_month_period'] = df_filtered['invoice_date'].dt.to_period('M')
    monthly_sales_agg = df_filtered.groupby(
        ['commercial_manager', 'family', 'year_month_period']
    )['sales_value'].sum().reset_index()
    # Convert period back to timestamp (start of month) for easier comparison
    monthly_sales_agg['month_start_date'] = monthly_sales_agg['year_month_period'].dt.to_timestamp()
    monthly_sales_agg = monthly_sales_agg.sort_values(['commercial_manager', 'family', 'month_start_date'])

    # --- Prepare Budget Lookup --- 
    budget_lookup = {}
    if df_budget is not None:
        required_budget_cols = ['commercial_manager', 'family', 'year', 'month', 'KG']
        if all(col in df_budget.columns for col in required_budget_cols):
             budget_lookup = df_budget.set_index(['commercial_manager', 'family', 'year', 'month'])['KG'].to_dict()
        else:
            print(f"Warning: Budget data missing required columns: {required_budget_cols}. Budget component will be ignored.")
            if weights.get('budget', 0) > 0:
                 weights['budget'] = 0.0 # Effectively disable budget if columns missing
                 # Re-normalize if budget weight was > 0
                 total_weight = sum(weights.values())
                 if not np.isclose(total_weight, 1.0) and total_weight != 0:
                     weights = {k: v / total_weight for k, v in weights.items()}

    # --- Generate Forecast Month by Month --- 
    forecast_results = []
    # Generate date range for forecast months
    forecast_dates = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='MS') # MS for Month Start

    # Filter out excluded managers if 'Todos' is selected
    managers_to_exclude = ['Concurso', 'Outros']
    if manager_filter is None: # Corresponds to "Todos"
        sales_data_for_groups = monthly_sales_agg[
            ~monthly_sales_agg['commercial_manager'].isin(managers_to_exclude)
        ]
        budget_data_for_groups = df_budget[~df_budget['commercial_manager'].isin(managers_to_exclude)] if df_budget is not None else None
    else: # Specific manager selected
        sales_data_for_groups = monthly_sales_agg # Use original if specific manager
        budget_data_for_groups = df_budget # Use original if specific manager

    # --- Update Budget Lookup based on potentially filtered budget data ---
    budget_lookup = {}
    if budget_data_for_groups is not None:
        required_budget_cols = ['commercial_manager', 'family', 'year', 'month', 'KG']
        if all(col in budget_data_for_groups.columns for col in required_budget_cols):
             budget_lookup = budget_data_for_groups.set_index(['commercial_manager', 'family', 'year', 'month'])['KG'].to_dict()
        # Keep original weight warning logic as is
        elif df_budget is not None: # Check original df_budget for warning consistency
             print(f"Warning: Budget data missing required columns: {required_budget_cols}. Budget component will be ignored.")
             if weights.get('budget', 0) > 0:
                 weights['budget'] = 0.0
                 total_weight = sum(weights.values())
                 if not np.isclose(total_weight, 1.0) and total_weight != 0:
                     weights = {k: v / total_weight for k, v in weights.items()}
    # -------------------------------------------------------------------

    # Get unique manager/family combinations to forecast for (from potentially filtered data)
    manager_family_groups = sales_data_for_groups[['commercial_manager', 'family']].drop_duplicates().to_records(index=False)

    # Pre-calculate components for the first time? Optimization? No, needs recalculation each month.

    for forecast_date in forecast_dates:
        f_year = forecast_date.year
        f_month = forecast_date.month
        # Cutoff is the end of the *previous* month
        cutoff_date = forecast_date - pd.Timedelta(days=1)
        cutoff_period = pd.Period(cutoff_date, freq='M')

        # Filter historical sales data available up to the cutoff date (from potentially filtered data)
        historical_sales_for_calc = sales_data_for_groups[sales_data_for_groups['year_month_period'] <= cutoff_period]

        if historical_sales_for_calc.empty:
            print(f"Warning: No historical sales data available before {cutoff_date.strftime('%Y-%m-%d')} for forecasting {forecast_date.strftime('%Y-%m')}. Skipping component calculation for this month.")
            # Still need to generate a forecast row, potentially with only budget?
            # Let's calculate components as 0 if no history.

        for manager, family in manager_family_groups:
            # Filter history further for the specific manager/family
            group_hist = historical_sales_for_calc[
                (historical_sales_for_calc['commercial_manager'] == manager) & 
                (historical_sales_for_calc['family'] == family)
            ].set_index('year_month_period') # Use period index for easier slicing

            # --- Calculate Components Dynamically for this forecast_date ---
            avg_2023 = 0
            avg_ytd = 0
            avg_last_3 = 0
            avg_last_12 = 0
            cagr_component_value = 0 # Initialize CAGR component

            if not group_hist.empty:
                # Component: Average Sales of 2023 (up to cutoff)
                sales_2023 = group_hist[group_hist.index.year == 2023]['sales_value']
                avg_2023 = sales_2023.mean() if not sales_2023.empty else 0

                # Component: YTD Average Sales of Current Year (year of cutoff_date)
                current_calc_year = cutoff_period.year
                ytd_sales = group_hist[group_hist.index.year == current_calc_year]['sales_value']
                avg_ytd = ytd_sales.mean() if not ytd_sales.empty else 0

                # Component: Last 3 Months Average (ending at cutoff_period)
                last_3_period_start = cutoff_period - 2 
                last_3_sales = group_hist[(group_hist.index >= last_3_period_start)]['sales_value']
                avg_last_3 = last_3_sales.mean() if not last_3_sales.empty else 0

                # Component: Last 12 Months Average (ending at cutoff_period)
                last_12_period_start = cutoff_period - 11 
                last_12_sales = group_hist[(group_hist.index >= last_12_period_start)]['sales_value']
                avg_last_12 = last_12_sales.mean() if not last_12_sales.empty else 0

                # --- CAGR Component Value Calculation (User Formula) ---
                cagr = 0.0
                # Calculate ratio only if avg_2023 is positive
                if avg_2023 > 0:
                    ratio = avg_ytd / avg_2023
                    # Calculate CAGR only if ratio is non-negative
                    if ratio >= 0:
                         try:
                             # Using 0.5 as exponent per user request
                             cagr = (ratio ** 0.5) - 1 
                         except Exception as e:
                             print(f"Warning: CAGR calculation error for {manager}/{family} on {forecast_date.strftime('%Y-%m')}: {e}")
                             cagr = 0.0 # Default to no growth on error
                    else:
                         # If ratio is negative, implies negative avg_ytd which doesn't make sense for CAGR base
                         cagr = 0.0 # Default to no growth
                
                # Calculate the component value based on avg_2023 and CAGR
                cagr_component_value = avg_2023 * (1 + cagr)

            # --- Budget Component ---
            budget_value = 0
            if weights.get('budget', 0) > 0 and budget_lookup:
                 budget_key = (manager, family, f_year, f_month)
                 budget_value = budget_lookup.get(budget_key, 0) # Default to 0 if not found

            # --- Weighted Calculation ---
            forecast_value = (
                avg_2023 * weights.get('avg_2023', 0) +
                avg_ytd * weights.get('avg_ytd', 0) +
                avg_last_3 * weights.get('last_3_months', 0) +
                avg_last_12 * weights.get('last_12_months', 0) +
                budget_value * weights.get('budget', 0) +
                cagr_component_value * weights.get('cagr', 0) # Added CAGR component value
            )

            forecast_results.append({
                'commercial_manager': manager,
                'family': family,
                'year': f_year,
                'month': f_month,
                'forecast_value': forecast_value
            })

    if not forecast_results:
         print("Warning: No forecast results generated.")
         return pd.DataFrame(columns=['commercial_manager', 'family', 'year', 'month', 'forecast_value'])

    # Aggregate results if 'Todos' was selected
    final_forecast_df = pd.DataFrame(forecast_results)
    if manager_filter is None and not final_forecast_df.empty:
        print("Aggregating forecast results for 'Todos'...")
        agg_cols = ['family', 'year', 'month']
        aggregated_df = final_forecast_df.groupby(agg_cols)['forecast_value'].sum().reset_index()
        # Add a placeholder manager column for potential consistency?
        # aggregated_df['commercial_manager'] = "Total (Excl. Concurso/Outros)"
        return aggregated_df
    else:
        return final_forecast_df # Return detailed if specific manager

# Example Usage (can be removed or kept for testing):
if __name__ == '__main__':
    # Create dummy sales data
    # Ensure invoice_date covers a longer period for testing past forecasts
    sales_dates = pd.date_range('2022-01-01', '2024-06-30', freq='D') # Daily data initially
    managers = ['MDD', 'Exportação']
    families = ['Cream Cracker', 'Maria']
    data = []
    np.random.seed(42)
    for date in sales_dates:
        # Simulate some days having no sales
        if np.random.rand() > 0.3:
            for manager in managers:
                for family in families:
                     # Simulate some seasonality and trend
                     month_factor = 1 + np.sin((date.month - 1) * np.pi / 6) * 0.2
                     year_trend = 1 + (date.year - 2022) * 0.1
                     base_sales = np.random.uniform(10, 50) # Lower daily sales
                     sales = base_sales * month_factor * year_trend * (1 + (managers.index(manager) * 0.1)) * (1 + (families.index(family) * 0.05))
                     data.append({
                         'invoice_date': date,
                         'commercial_manager': manager,
                         'family': family,
                         'sales_value': sales
                     })
    dummy_sales_df = pd.DataFrame(data)
    dummy_sales_df['invoice_date'] = pd.to_datetime(dummy_sales_df['invoice_date'])

    # Create dummy budget data for past and future
    budget_dates = pd.date_range('2023-01-01', '2025-12-31', freq='M') # Wider budget range
    budget_data = []
    for date in budget_dates:
         for manager in managers:
             for family in families:
                 budget_val = np.random.uniform(800, 1800) # Monthly budget
                 budget_data.append({
                     'commercial_manager': manager,
                     'family': family,
                     'year': date.year,
                     'month': date.month,
                     'KG': budget_val
                 })
    dummy_budget_df = pd.DataFrame(budget_data)

    # Define weights
    forecast_weights = {
        'avg_2023': 0.10,
        'avg_ytd': 0.10,
        'last_3_months': 0.25,
        'last_12_months': 0.25,
        'budget': 0.30,
        'cagr': 0.00 # Added CAGR weight for testing
    }

    # --- Test Case 1: Future Forecast ---
    print("--- Test Case 1: Future Forecast (Jul 2024 - Dec 2024) ---")
    start_forecast_1 = pd.Timestamp('2024-07-01')
    end_forecast_1 = pd.Timestamp('2024-12-01')
    forecast_df_1 = calculate_weighted_forecast(
        dummy_sales_df,
        dummy_budget_df,
        forecast_weights,
        start_forecast_1,
        end_forecast_1
    )
    print(forecast_df_1)

    # --- Test Case 2: Past Forecast ---
    print("\n--- Test Case 2: Past Forecast (Apr 2023 - Jun 2023) ---")
    start_forecast_2 = pd.Timestamp('2023-04-01')
    end_forecast_2 = pd.Timestamp('2023-06-01')
    forecast_df_2 = calculate_weighted_forecast(
        dummy_sales_df,
        dummy_budget_df,
        forecast_weights,
        start_forecast_2,
        end_forecast_2,
        family_filter='Maria' # Filter example
    )
    print(forecast_df_2)

    # --- Test Case 3: Mixed Past/Future Forecast ---
    print("\n--- Test Case 3: Mixed Forecast (Dec 2023 - Mar 2024) ---")
    start_forecast_3 = pd.Timestamp('2023-12-01')
    end_forecast_3 = pd.Timestamp('2024-03-01')
    forecast_df_3 = calculate_weighted_forecast(
        dummy_sales_df,
        dummy_budget_df,
        forecast_weights,
        start_forecast_3,
        end_forecast_3
    )
    print(forecast_df_3)

    # --- Test Case 4: Forecast start where no prior data exists ---
    print("\n--- Test Case 4: Forecast Start Before Data (Jan 2022 - Mar 2022) ---")
    start_forecast_4 = pd.Timestamp('2022-01-01')
    end_forecast_4 = pd.Timestamp('2022-03-01')
    forecast_df_4 = calculate_weighted_forecast(
        dummy_sales_df,
        dummy_budget_df,
        forecast_weights,
        start_forecast_4,
        end_forecast_4
    )
    print(forecast_df_4) 
"""
Module for SKU-level weighted forecasting.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple


def calculate_sku_ponderation(
    df_sales_hist: pd.DataFrame,
    df_budget: Optional[pd.DataFrame],
    weights: Dict[str, float],
    forecast_horizon: pd.PeriodIndex,
    sku_col: str,
    date_col: str, # Should be 'Period'
    value_col: str,
    family_col: str,
    manager_col: str,
    target_skus: List[str] # Added: List of SKUs to return forecasts for
) -> pd.DataFrame:
    """
    Calculates a weighted monthly sales forecast disaggregated to the SKU level.

    Args:
        df_sales_hist (pd.DataFrame): DataFrame with historical sales data, including
            sku_col, date_col (as Period), value_col, family_col, manager_col.
        df_budget (Optional[pd.DataFrame]): DataFrame with budget data (KG),
            including manager_col, family_col, year, month, KG.
        weights (Dict[str, float]): Dictionary of component weights (must sum to 1.0).
            Keys: 'avg_2023', 'avg_ytd', 'last_3_months', 'last_12_months', 'budget', 'cagr'.
        forecast_horizon (pd.PeriodIndex): The periods (e.g., '2024-07', '2024-08')
            for which to generate forecasts.
        sku_col (str): Name of the SKU column.
        date_col (str): Name of the Period column.
        value_col (str): Name of the sales value column.
        family_col (str): Name of the product family column.
        manager_col (str): Name of the commercial manager column.
        target_skus (List[str]): List of specific SKU identifiers for which to
                                return the final forecast.

    Returns:
        pd.DataFrame: DataFrame with columns [sku_col, date_col, 'Valor Previsto']
            containing the final aggregated forecast for each SKU in the horizon.
            Returns an empty DataFrame if errors occur or no forecast is generated.
    """
    # --- Input Validation and Preparation ---
    required_sales_cols = [sku_col, date_col, value_col, family_col, manager_col]
    if not all(col in df_sales_hist.columns for col in required_sales_cols):
        print(f"Error: Sales data missing required columns: {required_sales_cols}")
        return pd.DataFrame(columns=[sku_col, date_col, 'forecast_value'])

    if not isinstance(forecast_horizon, pd.PeriodIndex):
        print("Error: forecast_horizon must be a PeriodIndex.")
        return pd.DataFrame(columns=[sku_col, date_col, 'forecast_value'])

    # Ensure date_col is Period type
    if not isinstance(df_sales_hist[date_col].dtype, pd.PeriodDtype):
         try:
             # Attempt conversion if not already Period
             df_sales_hist[date_col] = pd.to_datetime(df_sales_hist[date_col]).dt.to_period('M')
         except Exception as e:
             print(f"Error converting date column '{date_col}' to Period: {e}")
             return pd.DataFrame(columns=[sku_col, date_col, 'forecast_value'])

    # Normalize weights (if necessary)
    total_weight = sum(weights.values())
    if not np.isclose(total_weight, 1.0):
        print(f"Warning: Weights sum to {total_weight:.4f}, normalizing to 1.0")
        if total_weight != 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
            print("Error: Weights sum to zero, cannot proceed.")
            return pd.DataFrame(columns=[sku_col, date_col, 'forecast_value'])

    # --- Budget Preparation (similar to custom_weighted_forecast.py) ---
    budget_lookup = {}
    budget_available = False
    if df_budget is not None:
        required_budget_cols = [manager_col, family_col, 'year', 'month', 'KG'] # Assuming 'KG' is the budget value column
        if all(col in df_budget.columns for col in required_budget_cols):
            try:
                budget_lookup = df_budget.set_index([manager_col, family_col, 'year', 'month'])['KG'].to_dict()
                budget_available = True
                print("Budget data loaded successfully.")
            except KeyError as e:
                 print(f"Error creating budget lookup: Missing key {e}")
            except Exception as e:
                print(f"Error processing budget data: {e}")
        else:
            print(f"Warning: Budget data missing required columns: {required_budget_cols}. Budget component will be ignored.")

    if not budget_available and weights.get('budget', 0) > 0:
        print("Warning: Budget component weight > 0 but budget data is unavailable or invalid. Setting budget weight to 0.")
        weights['budget'] = 0.0
        # Re-normalize weights if budget weight was removed
        total_weight = sum(weights.values())
        if total_weight != 0:
            weights = {k: v / total_weight for k, v in weights.items()}
        else:
             print("Error: Weights sum to zero after removing budget weight.")
             return pd.DataFrame(columns=[sku_col, date_col, 'forecast_value'])

    # --- Pre-aggregation --- 
    # Monthly sales aggregated by manager and family (used for FG forecast calculation)
    fg_monthly_sales = df_sales_hist.groupby(
        [manager_col, family_col, date_col]
    )[value_col].sum().reset_index()
    fg_monthly_sales = fg_monthly_sales.sort_values(date_col)

    # Monthly sales aggregated also by SKU (used for share calculation)
    sku_monthly_sales = df_sales_hist.groupby(
        [manager_col, family_col, sku_col, date_col]
    )[value_col].sum().reset_index()
    sku_monthly_sales = sku_monthly_sales.sort_values(date_col)

    all_sku_forecasts = [] # List to store results for each period

    # --- Main Loop: Iterate through Forecast Horizon ---
    for forecast_period in forecast_horizon:
        f_year = forecast_period.year
        f_month = forecast_period.month
        cutoff_period = forecast_period - 1 # Last period of actuals available

        print(f"\nCalculating forecast for: {forecast_period}")

        # --- 1. Calculate Família-Gestor (FG) Forecasts for forecast_period ---
        fg_forecasts_this_month = {}
        historical_fg_sales = fg_monthly_sales[fg_monthly_sales[date_col] <= cutoff_period]
        manager_family_groups = fg_monthly_sales[[manager_col, family_col]].drop_duplicates().to_records(index=False)

        if historical_fg_sales.empty:
             print(f"Warning: No historical FG sales data available up to {cutoff_period} to calculate FG forecasts for {forecast_period}. Skipping FG calc.")
             # Continue to next forecast period? Or try to use only budget?
             # For now, we can't calculate shares either, so skip month.
             continue

        print(f"  Calculating FG forecasts for {len(manager_family_groups)} groups...")
        for manager, family in manager_family_groups:
            group_hist = historical_fg_sales[
                (historical_fg_sales[manager_col] == manager) &
                (historical_fg_sales[family_col] == family)
            ].set_index(date_col)[value_col]

            # Calculate components based on group_hist up to cutoff_period
            avg_2023 = 0
            avg_ytd = 0
            avg_last_3 = 0
            avg_last_12 = 0
            cagr_component_value = 0

            if not group_hist.empty:
                # Ensure index is PeriodIndex before filtering by year
                if not isinstance(group_hist.index, pd.PeriodIndex):
                    group_hist.index = group_hist.index.to_period('M')

                sales_2023 = group_hist[group_hist.index.year == 2023]
                avg_2023 = sales_2023.mean() if not sales_2023.empty else 0

                current_calc_year = cutoff_period.year
                ytd_sales = group_hist[group_hist.index.year == current_calc_year]
                avg_ytd = ytd_sales.mean() if not ytd_sales.empty else 0

                last_3_period_start = cutoff_period - 2
                last_3_sales = group_hist[group_hist.index >= last_3_period_start]
                avg_last_3 = last_3_sales.mean() if not last_3_sales.empty else 0

                last_12_period_start = cutoff_period - 11
                last_12_sales = group_hist[group_hist.index >= last_12_period_start]
                avg_last_12 = last_12_sales.mean() if not last_12_sales.empty else 0

                # CAGR Component Value Calculation
                cagr = 0.0
                if avg_2023 > 0:
                    ratio = avg_ytd / avg_2023
                    if ratio >= 0:
                        try:
                            cagr = (ratio ** 0.5) - 1 # User Formula
                        except Exception:
                            cagr = 0.0
                    else:
                        cagr = 0.0
                cagr_component_value = avg_2023 * (1 + cagr)

            # Budget Component
            budget_value = 0
            if budget_available:
                budget_key = (manager, family, f_year, f_month)
                budget_value = budget_lookup.get(budget_key, 0)

            # Weighted FG Forecast Calculation
            fg_forecast_value = (
                avg_2023 * weights.get('avg_2023', 0) +
                avg_ytd * weights.get('avg_ytd', 0) +
                avg_last_3 * weights.get('last_3_months', 0) +
                avg_last_12 * weights.get('last_12_months', 0) +
                budget_value * weights.get('budget', 0) +
                cagr_component_value * weights.get('cagr', 0)
            )

            # Ensure forecast is not negative
            fg_forecasts_this_month[(manager, family)] = max(0, fg_forecast_value)

        # --- 2. Calculate SKU Shares based on Total Sales in the 3-Month Window --- (MODIFIED LOGIC)
        share_calculation_periods = pd.period_range(end=cutoff_period, periods=3, freq='M')
        print(f"  Calculating SKU shares based on total sales in periods: {share_calculation_periods.tolist()}")

        # Filter historical SKU sales data for the 3-month window
        share_sales_window = sku_monthly_sales[
            (sku_monthly_sales[date_col].isin(share_calculation_periods))
        ]

        # Calculate Total Sales per SKU-FG over the 3-month window
        sku_totals_3m = share_sales_window.groupby(
            [manager_col, family_col, sku_col]
        )[value_col].sum()

        # Calculate Total Sales per FG over the 3-month window
        fg_totals_3m = share_sales_window.groupby(
            [manager_col, family_col]
        )[value_col].sum()

        # Calculate the share based on totals
        calculated_sku_shares = {} # {(manager, family, sku): share}
        if not fg_totals_3m.empty:
            # Iterate through SKUs that had sales in the window
            # Iterate over the index directly and use .loc
            for index_tuple in sku_totals_3m.index:
                manager, family, sku = index_tuple # Unpack the MultiIndex tuple
                sku_total = sku_totals_3m.loc[index_tuple] # Get value using loc

                # Get the corresponding FG total
                # Note: fg_totals_3m index is (manager, family)
                fg_total = fg_totals_3m.get((manager, family), 0)
                if fg_total > 0:
                    share = sku_total / fg_total
                    calculated_sku_shares[(manager, family, sku)] = share
                # else: share is implicitly 0 if fg_total is 0

        print(f"  Calculated shares for {len(calculated_sku_shares)} SKU-FG combinations based on 3-month totals.")

        # --- 3. Disaggregate FG Forecasts and Aggregate by SKU ---
        sku_forecasts_aggregated = {} # {sku: total_forecast_value}
        print(f"  Disaggregating {len(fg_forecasts_this_month)} FG forecasts...")

        # <<< START DEBUG SKU 50000067 for July 2024 >>>
        target_sku_debug = '50000067'
        target_period_debug = pd.Period('2024-07', freq='M')
        # <<< END DEBUG SKU 50000067 >>>

        for (manager, family), fg_forecast in fg_forecasts_this_month.items():
            if fg_forecast == 0:
                continue # No forecast value to disaggregate

            # Inner loop: Find SKUs and their calculated shares corresponding to the current FG
            # Uses the new 'calculated_sku_shares' dictionary
            for (m_key, f_key, sku_key), share in calculated_sku_shares.items():
                # Check if the share belongs to the current FG group
                if m_key == manager and f_key == family:
                    if share > 0: # Use the calculated share
                        # Disaggregate: Multiply FG forecast by the SKU's share of 3m totals
                        disaggregated_value = fg_forecast * share

                        # Aggregate: Add this value to the SKU's total forecast for the period
                        sku_forecasts_aggregated[sku_key] = sku_forecasts_aggregated.get(sku_key, 0) + disaggregated_value

                        # <<< START DEBUG SKU 50000067 for July 2024 >>>
                        if forecast_period == target_period_debug and sku_key == target_sku_debug:
                            print(f"    DEBUG [{target_sku_debug} @ {forecast_period}]:")
                            print(f"      - Combo FG: ({manager}, {family})")
                            print(f"      - Previsão FG: {fg_forecast:.2f}")
                            print(f"      - Share SKU (média Abr-Jun): {share:.4f}")
                            print(f"      - Valor Desagregado Contribuído: {disaggregated_value:.2f}")
                        # <<< END DEBUG SKU 50000067 >>>

        print(f"  Aggregated forecasts for {len(sku_forecasts_aggregated)} unique SKUs for {forecast_period}.")

        # --- Explicitly Add Zero Forecasts (MODIFIED LOGIC) ---
        # Identify all SKUs that belong to the FG groups processed in this period
        # Use sku_monthly_sales which retains the SKU column
        relevant_fg_tuples = list(fg_forecasts_this_month.keys())
        relevant_sku_data = sku_monthly_sales[
            sku_monthly_sales[[manager_col, family_col]].apply(tuple, axis=1).isin(relevant_fg_tuples)
        ]
        active_skus_in_period_fgs = relevant_sku_data[sku_col].unique()

        zero_forecast_skus_added = 0
        for sku in active_skus_in_period_fgs:
            if sku not in sku_forecasts_aggregated:
                # This SKU belongs to a relevant FG but didn't get a value
                # (likely due to zero share in the 3m window or zero sales in 2024 filter)
                sku_forecasts_aggregated[sku] = 0.0
                zero_forecast_skus_added += 1

        if zero_forecast_skus_added > 0:
            print(f"  Added {zero_forecast_skus_added} SKUs with zero forecast for {forecast_period}.")

        # <<< START DEBUG SKU 50000067 for July 2024 >>>
        if forecast_period == target_period_debug and target_sku_debug in sku_forecasts_aggregated:
            final_agg_value = sku_forecasts_aggregated[target_sku_debug]
            print(f"    DEBUG [{target_sku_debug} @ {forecast_period}]: Valor Final Agregado = {final_agg_value:.2f}")
            print("    ---------------------------------------------------") # Separator
        # <<< END DEBUG SKU 50000067 >>>

        # --- 4. Store Results for this Period ---
        if sku_forecasts_aggregated:
            period_results = pd.DataFrame({
                sku_col: list(sku_forecasts_aggregated.keys()),
                'forecast_value': list(sku_forecasts_aggregated.values())
            })
            period_results[date_col] = forecast_period
            all_sku_forecasts.append(period_results[[sku_col, date_col, 'forecast_value']])
        else:
             print(f"  No final SKU forecasts generated for {forecast_period}.")

    # --- Final Output ---
    if not all_sku_forecasts:
        print("\nWarning: No SKU forecasts were generated across the entire horizon.")
        return pd.DataFrame(columns=[sku_col, date_col, 'forecast_value'])

    final_df = pd.concat(all_sku_forecasts, ignore_index=True)

    # Ensure correct types
    final_df[date_col] = pd.to_datetime(final_df[date_col].astype(str)).dt.to_period('M')
    final_df['forecast_value'] = pd.to_numeric(final_df['forecast_value'], errors='coerce').fillna(0)

    # Rename column to match expectation in generate_forecasts
    final_df.rename(columns={'forecast_value': 'Valor Previsto'}, inplace=True)

    # --- Filter final results to only include target SKUs ---
    if target_skus:
        print(f"Filtering final results to keep only {len(target_skus)} target SKUs.")
        # Ensure SKU column type matches target list type for filtering
        if not final_df.empty and target_skus and final_df[sku_col].dtype != type(target_skus[0]):
             try:
                 print(f"Aligning final DataFrame SKU column type ({final_df[sku_col].dtype}) with target SKU list type ({type(target_skus[0])}).")
                 # Convert target_skus list elements to match DataFrame column type
                 col_type = final_df[sku_col].dtype
                 target_skus_typed = [col_type.type(sku) for sku in target_skus]
             except Exception as e:
                 print(f"Warning: Failed to align target SKU list type for final filtering: {e}. Filter might fail.")
                 target_skus_typed = target_skus # Proceed with original list
        else:
            target_skus_typed = target_skus

        final_df = final_df[final_df[sku_col].isin(target_skus_typed)]
        print(f"Filtered DataFrame shape: {final_df.shape}")
    else:
         print("Warning: No target SKUs provided for final filtering. Returning all calculated forecasts.")

    print(f"\nFinished SKU Ponderation. Generated forecast shape: {final_df.shape}")
    return final_df


# Example Usage (for potential testing)
if __name__ == '__main__':
    # Create more realistic dummy sales data including SKUs
    sales_dates = pd.date_range('2022-01-01', '2024-06-30', freq='M') # Monthly data
    managers = ['MDD', 'Exportação', 'Outros']
    families = ['Cream Cracker', 'Maria']
    skus_per_fg = {
        ('MDD', 'Cream Cracker'): ['SKU001', 'SKU002', 'SKU003'],
        ('MDD', 'Maria'): ['SKU004'],
        ('Exportação', 'Cream Cracker'): ['SKU001', 'SKU005'], # SKU001 in multiple
        ('Exportação', 'Maria'): ['SKU006', 'SKU007'],
        ('Outros', 'Cream Cracker'): ['SKU008'],
        ('Outros', 'Maria'): ['SKU004', 'SKU009'] # SKU004 in multiple
    }

    data = []
    np.random.seed(42)
    for date_period in sales_dates:
        for (manager, family), skus in skus_per_fg.items():
            fg_monthly_total = np.random.uniform(500, 5000) # Simulate FG total for month
            # Distribute total among SKUs somewhat randomly
            sku_weights = np.random.rand(len(skus))
            sku_weights /= sku_weights.sum()
            for i, sku in enumerate(skus):
                sales = fg_monthly_total * sku_weights[i]
                if np.random.rand() > 0.1: # Simulate occasional zero sales for an SKU
                    data.append({
                        'Period': date_period,
                        'commercial_manager': manager,
                        'family': family,
                        'sku': sku,
                        'sales_value': sales
                    })

    dummy_sales_df = pd.DataFrame(data)

    # Create dummy budget data
    budget_dates = pd.date_range('2023-01-01', '2025-12-31', freq='M')
    budget_data = []
    for date in budget_dates:
        for manager in managers:
            for family in families:
                budget_val = np.random.uniform(1000, 6000)
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
        'cagr': 0.00
    }

    # Define forecast horizon
    horizon = pd.period_range(start='2024-07', end='2024-09', freq='M')

    print("--- Running SKU Ponderation Example ---")
    forecast_df = calculate_sku_ponderation(
        df_sales_hist=dummy_sales_df,
        df_budget=dummy_budget_df,
        weights=forecast_weights,
        forecast_horizon=horizon,
        sku_col='sku',
        date_col='Period',
        value_col='sales_value',
        family_col='family',
        manager_col='commercial_manager',
        target_skus=['SKU001', 'SKU004']
    )

    print("\n--- Forecast Results ---")
    print(forecast_df)

    # Check a specific SKU that was in multiple groups
    print("\n--- Forecast for SKU001 ---")
    print(forecast_df[forecast_df['sku'] == 'SKU001'])
    print("\n--- Forecast for SKU004 ---")
    print(forecast_df[forecast_df['sku'] == 'SKU004']) 
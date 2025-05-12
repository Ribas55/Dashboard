import streamlit as st
import pandas as pd
import numpy as np # Added for calculations
import plotly.graph_objects as go # Keep for potential future comparisons
import plotly.express as px # Keep for potential future comparisons
from collections import Counter # For summary counting
# Import any specific functions needed from src or utils if necessary
# from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error # Optional: could use library

# --- Constants ---
SKU_COL = 'sku'
PERIOD_COL = 'Period'
FORECAST_COL = 'Valor Previsto'
ACTUAL_COL = 'Valor Real de Vendas'
METRICS_LOWER_BETTER = ["MSE", "MAE", "MAPE"]
METRICS_HIGHER_BETTER = ["Assertividade"]
ALL_COMPARISON_METRICS = METRICS_LOWER_BETTER + METRICS_HIGHER_BETTER
CLASSIFICATION_DF_KEY = 'classification_results_df' # Key for classification data in session state
CATEGORY_ABC_COL = 'ABC' # Assumed column name for ABC
CATEGORY_XYZ_COL = 'XYZ' # Assumed column name for XYZ
CATEGORY_ABCXYZ_COL = 'ABC_XYZ' # Assumed column name for combined ABC/XYZ

# --- Helper Functions for Metric Calculation ---

def calculate_mse(actual, predicted):
    """Calculates Mean Squared Error."""
    actual = pd.to_numeric(actual, errors='coerce').fillna(0)
    predicted = pd.to_numeric(predicted, errors='coerce').fillna(0)
    return (actual - predicted) ** 2

def calculate_mae(actual, predicted):
    """Calculates Mean Absolute Error."""
    actual = pd.to_numeric(actual, errors='coerce').fillna(0)
    predicted = pd.to_numeric(predicted, errors='coerce').fillna(0)
    return abs(actual - predicted)

def calculate_mape(actual, predicted):
    """Calculates Mean Absolute Percentage Error. Handles division by zero."""
    actual = pd.to_numeric(actual, errors='coerce').fillna(0)
    predicted = pd.to_numeric(predicted, errors='coerce').fillna(0)
    # Avoid division by zero: if actual is 0, MAPE is 0 if predicted is also 0, otherwise infinity (or large number).
    # We'll return 0 if actual is 0 for simplicity in weighted average.
    return np.where(actual == 0, 0, abs((actual - predicted) / actual)) * 100

def calculate_assertividade(actual, predicted):
    """Calculates Assertividade (Accuracy). Handles division by zero/max."""
    actual = pd.to_numeric(actual, errors='coerce').fillna(0)
    predicted = pd.to_numeric(predicted, errors='coerce').fillna(0)
    abs_error = abs(actual - predicted)
    max_val = np.maximum(actual, predicted)
    # If max_val is 0, accuracy is 1 if error is 0, else 0.
    accuracy = np.where(max_val == 0, np.where(abs_error == 0, 1.0, 0.0), 1 - (abs_error / max_val))
    return np.nan_to_num(accuracy, nan=0.0) # Ensure NaNs become 0

def weighted_average(values, weights):
    """Calculates weighted average, handling zero weights sum."""
    values = pd.to_numeric(values, errors='coerce').fillna(0)
    weights = pd.to_numeric(weights, errors='coerce').fillna(0)
    total_weight = weights.sum()
    if total_weight == 0:
        return 0 # Or np.nan, depending on desired behavior
    return (values * weights).sum() / total_weight

# --- Formatting Function ---
def format_metric_value(metric_name, value):
    """Formats the metric value for display."""
    if pd.isna(value):
        return "N/A" # Handle potential NaNs before formatting
    if metric_name in METRICS_HIGHER_BETTER + METRICS_LOWER_BETTER:
        if metric_name in ["Assertividade", "MAPE"]:
            # Check if value is already percentage (e.g., from Assertividade calc)
            if isinstance(value, (int, float)) and value > 1: # Heuristic: assume it's already percentage
                return f"{value:.2f}%"
            elif isinstance(value, (int, float)): # Convert to percentage if it's a fraction
                 return f"{value * 100:.2f}%"
            else: # Fallback if not numeric
                return str(value)
        else:
            return f"{value:.2f}"
    else: # Default formatting if new metrics are added
        try:
             return f"{value:.2f}"
        except (TypeError, ValueError):
             return str(value) # Fallback to string

# --- Helper Function to Calculate Monthly Metrics for a Set ---
def calculate_monthly_metrics_for_set(set_data, metrics_to_calculate):
    """Combines set data, calculates SELECTED monthly metrics, returns DataFrame."""
    all_results_dfs = []
    for forecast_key, saved_forecast in set_data.items():
        if 'results' in saved_forecast and isinstance(saved_forecast['results'], pd.DataFrame):
            all_results_dfs.append(saved_forecast['results'].copy())

    if not all_results_dfs:
        return pd.DataFrame() # Return empty if no valid data

    combined_df = pd.concat(all_results_dfs, ignore_index=True)

    # --- Basic Validation ---
    required_cols = [SKU_COL, PERIOD_COL, FORECAST_COL, ACTUAL_COL]
    if not all(col in combined_df.columns for col in required_cols):
        print(f"Warning: Missing required columns in set data. Cannot calculate metrics")
        return pd.DataFrame()

    # --- Data Preparation ---
    if not isinstance(combined_df[PERIOD_COL].dtype, pd.PeriodDtype):
        try:
            combined_df[PERIOD_COL] = pd.to_datetime(combined_df[PERIOD_COL].astype(str)).dt.to_period('M')
        except Exception as e:
            print(f"Warning: Could not convert Period column to PeriodDtype: {e}")
            return pd.DataFrame()

    combined_df.drop_duplicates(subset=[SKU_COL, PERIOD_COL], keep='first', inplace=True)
    combined_df[ACTUAL_COL] = pd.to_numeric(combined_df[ACTUAL_COL], errors='coerce').fillna(0)
    combined_df[FORECAST_COL] = pd.to_numeric(combined_df[FORECAST_COL], errors='coerce').fillna(0)

    # --- Monthly Calculation ---
    monthly_results = []
    unique_months = sorted(combined_df[PERIOD_COL].unique())

    for month_period in unique_months:
        month_df = combined_df[combined_df[PERIOD_COL] == month_period].copy()
        if month_df.empty: continue

        weights_actual = month_df[ACTUAL_COL]
        weights_forecast = month_df[FORECAST_COL]
        month_str = month_period.strftime('%Y-%m')

        # Calculate ONLY selected metrics
        if "MSE" in metrics_to_calculate:
            month_df['mse'] = calculate_mse(month_df[ACTUAL_COL], month_df[FORECAST_COL])
            value = weighted_average(month_df['mse'], weights_actual)
            monthly_results.append({"Mês": month_str, "Métrica": "MSE", "Resultados": value})

        if "MAE" in metrics_to_calculate:
            month_df['mae'] = calculate_mae(month_df[ACTUAL_COL], month_df[FORECAST_COL])
            value = weighted_average(month_df['mae'], weights_actual)
            monthly_results.append({"Mês": month_str, "Métrica": "MAE", "Resultados": value})

        if "MAPE" in metrics_to_calculate:
            month_df['mape'] = calculate_mape(month_df[ACTUAL_COL], month_df[FORECAST_COL])
            value = weighted_average(month_df['mape'], weights_actual)
            monthly_results.append({"Mês": month_str, "Métrica": "MAPE", "Resultados": value})

        if "Assertividade" in metrics_to_calculate:
            month_df['assertividade'] = calculate_assertividade(month_df[ACTUAL_COL], month_df[FORECAST_COL])
            value = weighted_average(month_df['assertividade'], weights_forecast) * 100 # Percentage
            monthly_results.append({"Mês": month_str, "Métrica": "Assertividade", "Resultados": value})

    return pd.DataFrame(monthly_results)

# --- NEW Helper Function to Calculate CATEGORIZED Monthly Metrics ---
def calculate_categorized_monthly_metrics(set_data, metrics_to_calculate, classification_df, category_col, category_col_name):
    """Calculates monthly metrics broken down by a specified category."""
    all_results_dfs = []
    for forecast_key, saved_forecast in set_data.items():
        if 'results' in saved_forecast and isinstance(saved_forecast['results'], pd.DataFrame):
            all_results_dfs.append(saved_forecast['results'].copy())

    if not all_results_dfs:
        return pd.DataFrame()

    combined_df = pd.concat(all_results_dfs, ignore_index=True)

    # --- Validation ---
    required_cols = [SKU_COL, PERIOD_COL, FORECAST_COL, ACTUAL_COL]
    if not all(col in combined_df.columns for col in required_cols):
        st.warning(f"Faltam colunas necessárias nos dados do conjunto. Não é possível calcular métricas categorizadas.")
        return pd.DataFrame()
    if classification_df is None or classification_df.empty:
        st.warning("Dados de classificação não encontrados ou vazios. Não é possível calcular métricas categorizadas.")
        return pd.DataFrame()
    if category_col not in classification_df.columns:
         st.warning(f"Coluna de categoria '{category_col}' não encontrada nos dados de classificação.")
         return pd.DataFrame()
    if SKU_COL not in classification_df.columns:
         st.warning(f"Coluna SKU '{SKU_COL}' não encontrada nos dados de classificação para junção.")
         return pd.DataFrame()

    # --- Data Preparation ---
    if not isinstance(combined_df[PERIOD_COL].dtype, pd.PeriodDtype):
        try:
            combined_df[PERIOD_COL] = pd.to_datetime(combined_df[PERIOD_COL].astype(str)).dt.to_period('M')
        except Exception as e:
            st.warning(f"Não foi possível converter a coluna Período para PeriodDtype: {e}")
            return pd.DataFrame()

    combined_df.drop_duplicates(subset=[SKU_COL, PERIOD_COL], keep='first', inplace=True)
    combined_df[ACTUAL_COL] = pd.to_numeric(combined_df[ACTUAL_COL], errors='coerce').fillna(0)
    combined_df[FORECAST_COL] = pd.to_numeric(combined_df[FORECAST_COL], errors='coerce').fillna(0)

    # Merge with classification data
    try:
        # Ensure SKU columns are of compatible types before merge
        combined_df[SKU_COL] = combined_df[SKU_COL].astype(str)
        classification_df[SKU_COL] = classification_df[SKU_COL].astype(str)
        merged_df = pd.merge(combined_df, classification_df[[SKU_COL, category_col]], on=SKU_COL, how='left')
        # Handle SKUs potentially not in classification_df
        merged_df[category_col].fillna('Não Classificado', inplace=True)
    except Exception as e:
        st.error(f"Erro ao juntar dados de resultados com dados de classificação: {e}")
        return pd.DataFrame()

    # --- Monthly Calculation per Category ---
    monthly_results = []
    # Group by month and the chosen category
    grouped = merged_df.groupby([PERIOD_COL, category_col])

    for (month_period, category_value), group_df in grouped:
        if group_df.empty: continue

        weights_actual = group_df[ACTUAL_COL]
        weights_forecast = group_df[FORECAST_COL]
        month_str = month_period.strftime('%Y-%m')

        if "MSE" in metrics_to_calculate:
            group_df['mse'] = calculate_mse(group_df[ACTUAL_COL], group_df[FORECAST_COL])
            value = weighted_average(group_df['mse'], weights_actual)
            monthly_results.append({"Mês": month_str, "Métrica": "MSE", category_col_name: category_value, "Resultados": value})

        if "MAE" in metrics_to_calculate:
            group_df['mae'] = calculate_mae(group_df[ACTUAL_COL], group_df[FORECAST_COL])
            value = weighted_average(group_df['mae'], weights_actual)
            monthly_results.append({"Mês": month_str, "Métrica": "MAE", category_col_name: category_value, "Resultados": value})

        if "MAPE" in metrics_to_calculate:
            group_df['mape'] = calculate_mape(group_df[ACTUAL_COL], group_df[FORECAST_COL])
            value = weighted_average(group_df['mape'], weights_actual)
            monthly_results.append({"Mês": month_str, "Métrica": "MAPE", category_col_name: category_value, "Resultados": value})

        if "Assertividade" in metrics_to_calculate:
            group_df['assertividade'] = calculate_assertividade(group_df[ACTUAL_COL], group_df[FORECAST_COL])
            value = weighted_average(group_df['assertividade'], weights_forecast) * 100 # Percentage
            monthly_results.append({"Mês": month_str, "Métrica": "Assertividade", category_col_name: category_value, "Resultados": value})

    return pd.DataFrame(monthly_results)

# --- Helper Function to Determine Best Method --- (REVISED for potential category index)
def determine_best_method(row, set_columns):
    """Determines the best performing set(s) for a given metric row (potentially including category)."""
    # Metric name is always the last level of the index
    metric_name = row.name[-1] # Adapts to index with or without category
    relevant_values = row[set_columns].dropna().astype(float) # Ensure numeric comparison

    if relevant_values.empty:
        return "N/A"

    best_value = None
    best_sets = []

    if metric_name in METRICS_LOWER_BETTER:
        best_value = relevant_values.min()
        best_sets = relevant_values[relevant_values <= best_value].index.tolist()
    elif metric_name in METRICS_HIGHER_BETTER:
        best_value = relevant_values.max()
        best_sets = relevant_values[relevant_values >= best_value].index.tolist()
    else:
        return "N/A" # Unknown metric type

    if len(best_sets) == 1:
        return best_sets[0]
    elif len(best_sets) > 1:
        return "Empate"
    else:
        return "N/A" # Should not happen if relevant_values is not empty

# --- Render Function ---

def render():
    st.title("🆚 Resultados e Comparações")

    # --- Mode Selection ---
    analysis_mode = st.radio(
        "Selecione o Modo de Análise:",
        ("Analisar Resultados", "Comparar Resultados"),
        horizontal=True,
        key="analysis_mode_selector"
    )

    st.divider()

    # ==============================
    # == Analisar Resultados Mode ==
    # ==============================
    if analysis_mode == "Analisar Resultados":
        st.markdown("### Analisar Resultados de um Conjunto de Forecasts")

        # --- Check for completed forecast sets ---
        if 'completed_forecast_sets' not in st.session_state or not st.session_state.completed_forecast_sets:
            st.warning("Não há conjuntos de forecasts guardados para analisar. Por favor, guarde um conjunto na página 'Métodos de Forecasting'.")
            st.stop()

        completed_sets = st.session_state.completed_forecast_sets
        set_names = list(completed_sets.keys())

        # --- Select Forecast Set ---
        selected_set_name = st.selectbox(
            "Selecione o Conjunto de Forecasts:",
            options=set_names,
            index=0 if set_names else None, # Default to first set if available
            key="select_analysis_set"
        )

        if not selected_set_name:
            st.info("Por favor, selecione um conjunto para análise.")
            st.stop()

        # --- Choose Metrics ---
        available_metrics = ["MSE", "MAE", "MAPE", "Assertividade"]
        selected_metrics = st.multiselect(
            "Selecione as Métricas para Calcular:",
            options=available_metrics,
            default=[], # Default to empty list
            key="select_analysis_metrics"
        )

        # --- NEW: Category Breakdown Option ---
        st.markdown("##### Opções de Detalhamento")
        view_by_category = st.toggle("Ver resultados por categoria", key="analysis_view_category_toggle")
        selected_category_type = None
        category_col_map = {
            "Ver por categoria ABC": (CATEGORY_ABC_COL, "Categoria ABC"),
            "Ver por categoria XYZ": (CATEGORY_XYZ_COL, "Categoria XYZ"),
            "Ver por categoria ABC/XYZ": (CATEGORY_ABCXYZ_COL, "Categoria ABC/XYZ")
        }
        if view_by_category:
             # Check if classification data exists
             if CLASSIFICATION_DF_KEY not in st.session_state or st.session_state[CLASSIFICATION_DF_KEY] is None or st.session_state[CLASSIFICATION_DF_KEY].empty:
                 st.warning(f"Para ver resultados por categoria, por favor, execute primeiro a análise ABC/XYZ na página correspondente. Os resultados da classificação não foram encontrados em `st.session_state['{CLASSIFICATION_DF_KEY}']`.")
                 # Don't render the selectbox if data is missing, but leave the toggle as is.
                 # Subsequent logic will handle the case where view_by_category is True but selected_category_type is None.
             else:
                 selected_category_type = st.selectbox(
                     "Escolha o tipo de detalhamento:",
                     options=list(category_col_map.keys()),
                     key="analysis_category_select"
                 )

        # --- Run Analysis Button ---
        run_analysis = st.button("Analisar Resultados", key="run_set_analysis")

        # --- Analysis Execution & Display ---
        # Initialize session state flags/data if they don't exist
        if 'analysis_results_df' not in st.session_state:
             st.session_state.analysis_results_df = None
        if 'last_run_set_analysis' not in st.session_state:
             st.session_state.last_run_set_analysis = None # Track which set was last analyzed via button
        # Store category selection state at the time of button press
        if 'last_run_analysis_category_toggle' not in st.session_state:
            st.session_state.last_run_analysis_category_toggle = False
        if 'last_run_analysis_category_type' not in st.session_state:
            st.session_state.last_run_analysis_category_type = None

        analysis_df_to_display = None # Variable to hold the df for display section

        if run_analysis:
            st.session_state.last_run_set_analysis = selected_set_name # Mark this set as analyzed by button
            st.session_state.last_run_analysis_category_toggle = view_by_category
            st.session_state.last_run_analysis_category_type = selected_category_type

            if not selected_metrics:
                st.warning("Por favor, selecione pelo menos uma métrica para analisar.")
                st.session_state.analysis_results_df = None # Clear previous results
            # Check again if category view is intended but data is missing (might happen if classification ran between toggle and button press)
            elif view_by_category and (CLASSIFICATION_DF_KEY not in st.session_state or st.session_state[CLASSIFICATION_DF_KEY] is None or st.session_state[CLASSIFICATION_DF_KEY].empty):
                 st.error(f"Erro: A opção 'Ver resultados por categoria' está ativa, mas os dados de classificação em `st.session_state['{CLASSIFICATION_DF_KEY}']` não foram encontrados ou estão vazios. Execute a análise ABC/XYZ primeiro.")
                 st.session_state.analysis_results_df = None
            # Also check if category view is selected BUT the category type wasn't selected (e.g. due to missing data)
            elif view_by_category and not selected_category_type:
                 st.warning("A opção 'Ver resultados por categoria' está ativa, mas nenhum tipo de detalhamento foi selecionado (provavelmente devido a dados de classificação em falta). Calculando resultados gerais.")
                 view_by_category = False # Treat as non-categorized for calculation
            else:
                st.markdown("---") # Visual separator
                set_data = completed_sets.get(selected_set_name)
                if not set_data:
                    st.error(f"Erro ao carregar dados para o conjunto '{selected_set_name}'.")
                    st.session_state.analysis_results_df = None # Clear previous results
                    st.stop()

                # --- Calculate Metrics (Overall or Categorized) ---
                results_df = None
                if view_by_category and selected_category_type:
                    with st.spinner(f"Calculando métricas por '{selected_category_type}'..."):
                        try:
                            # Ensure selected_category_type is valid key before lookup
                            if selected_category_type in category_col_map:
                                category_col, category_col_name = category_col_map[selected_category_type]
                                classification_df = st.session_state[CLASSIFICATION_DF_KEY]
                                results_df = calculate_categorized_monthly_metrics(
                                    set_data,
                                    selected_metrics,
                                    classification_df,
                                    category_col,
                                    category_col_name
                                )
                            else:
                                st.warning(f"Tipo de categoria inválido '{selected_category_type}' encontrado. Calculando métricas gerais.")
                                results_df = calculate_monthly_metrics_for_set(set_data, selected_metrics)
                        except Exception as e:
                            st.error(f"Erro durante o cálculo das métricas categorizadas: {e}")
                            results_df = None

                else: # Calculate overall metrics
                     with st.spinner("Calculando métricas gerais mensais..."):
                        try:
                             results_df = calculate_monthly_metrics_for_set(set_data, selected_metrics)
                        except Exception as e:
                             st.error(f"Erro durante o cálculo das métricas gerais: {e}")
                             results_df = None

                # --- Store results in session state ---
                if results_df is not None and not results_df.empty:
                    st.session_state.analysis_results_df = results_df
                else:
                    st.warning("Nenhuma métrica mensal pôde ser calculada para as seleções atuais.")
                    st.session_state.analysis_results_df = None

        # --- Display Area (Filters and Table) --- Triggered based on last run state ---
        # Check if the button was pressed for the currently selected set, matching category toggle state, and results exist
        if st.session_state.last_run_set_analysis == selected_set_name and \
           st.session_state.last_run_analysis_category_toggle == view_by_category and \
           st.session_state.last_run_analysis_category_type == selected_category_type and \
           st.session_state.analysis_results_df is not None and \
           not st.session_state.analysis_results_df.empty:

            current_results_df = st.session_state.analysis_results_df
            is_categorized = view_by_category and selected_category_type is not None
            category_display_name = None
            if is_categorized and selected_category_type and selected_category_type in category_col_map:
                category_display_name = category_col_map[selected_category_type][1]

            st.subheader(f"Análise Mensal para o Conjunto '{selected_set_name}'" + (f" por {category_display_name}" if category_display_name else ""))

            # --- Filters ---
            filter_col1, filter_col2 = st.columns(2)
            # Determine available filter options from the current results
            unique_months_in_results = sorted(current_results_df['Mês'].unique())
            unique_metrics_in_results = sorted(current_results_df['Métrica'].unique())
            unique_categories_in_results = []
            if is_categorized and category_display_name in current_results_df.columns:
                 unique_categories_in_results = sorted(current_results_df[category_display_name].unique())

            with filter_col1:
                 # --- Month Filter ---
                 filter_options_month = ["Todos"] + unique_months_in_results
                 selected_month_filter = st.selectbox(
                     "Filtrar por Mês:",
                     options=filter_options_month,
                     index=0,
                     key="analysis_month_filter"
                 )
            with filter_col2:
                 # --- Metric Filter ---
                 filter_options_metric = ["Todas"] + unique_metrics_in_results
                 selected_metric_filter = st.selectbox(
                      "Filtrar por Métrica:",
                      options=filter_options_metric,
                      index=0,
                      key="analysis_metric_filter"
                 )

            # --- Category Filter (Only if breakdown is active) ---
            selected_category_filter = "Todos"
            if is_categorized and unique_categories_in_results:
                filter_options_category = ["Todos"] + unique_categories_in_results
                selected_category_filter = st.selectbox(
                     f"Filtrar por {category_display_name}:",
                     options=filter_options_category,
                     index=0,
                     key="analysis_category_filter_value"
                 )

            # --- Filter Data ---
            display_df = current_results_df.copy()
            if selected_month_filter != "Todos":
                display_df = display_df[display_df['Mês'] == selected_month_filter]
            if selected_metric_filter != "Todos":
                display_df = display_df[display_df['Métrica'] == selected_metric_filter]
            if is_categorized and selected_category_filter != "Todos" and category_display_name in display_df.columns:
                display_df = display_df[display_df[category_display_name] == selected_category_filter]

            # --- Display Results Table ---
            if not display_df.empty:
                 # Apply formatting
                 formatted_display_df = display_df.copy()
                 formatted_display_df['Resultados'] = formatted_display_df.apply(
                     lambda row: format_metric_value(row['Métrica'], row['Resultados']), axis=1
                 )

                 # Define columns to show
                 cols_to_show = ['Mês', 'Métrica']
                 if is_categorized and category_display_name:
                     cols_to_show.append(category_display_name)
                 cols_to_show.append('Resultados')

                 # Reorder if category column exists
                 if category_display_name and category_display_name not in ['Mês', 'Métrica', 'Resultados']:
                      # Ensure the order is Month, Metric, Category, Result
                      order = ['Mês', 'Métrica']
                      if category_display_name in formatted_display_df.columns:
                          order.append(category_display_name)
                      order.append('Resultados')
                      # Filter out any columns not present (safety check)
                      cols_to_show = [col for col in order if col in formatted_display_df.columns]

                 st.dataframe(
                     formatted_display_df[cols_to_show],
                     use_container_width=True,
                     hide_index=True
                 )
            else:
                 st.info(f"Não há resultados para a combinação de filtros selecionada.")

        elif run_analysis and (st.session_state.analysis_results_df is None or st.session_state.analysis_results_df.empty):
             # Show message if analysis was run but resulted in no data
             st.info(f"Não foram encontrados ou calculados resultados para o conjunto '{selected_set_name}' com as seleções atuais.")

    # ===============================
    # == Comparar Resultados Mode ==
    # ===============================
    elif analysis_mode == "Comparar Resultados":
        st.markdown("### Comparar Desempenho entre Conjuntos de Forecasts")

        # --- Check for necessary data in session state ---
        if 'completed_forecast_sets' not in st.session_state or not st.session_state.completed_forecast_sets:
            st.warning("Não há conjuntos de forecasts guardados para comparar. Por favor, guarde pelo menos dois conjuntos na página 'Métodos de Forecasting'.")
            st.stop()

        completed_sets = st.session_state.completed_forecast_sets
        set_names = list(completed_sets.keys())

        if len(set_names) < 2:
             st.warning("São necessários pelo menos dois conjuntos de forecasts guardados para fazer uma comparação.")
             st.stop()

        # --- Select Sets to Compare ---
        selected_set_keys = st.multiselect(
            "Selecione os Conjuntos de Forecasts para Comparar (mínimo 2):",
            options=set_names,
            default=[], # Default to empty list
            key="select_comparison_sets"
        )

        # --- Select Metrics to Compare ---
        selected_metrics_comp = st.multiselect(
            "Selecione as Métricas para Comparar:",
            options=ALL_COMPARISON_METRICS,
            default=[], # Default to empty list
            key="select_comparison_metrics"
        )

        # --- NEW: Category Breakdown Option ---
        st.markdown("##### Opções de Detalhamento")
        view_by_category_comp = st.toggle("Ver resultados por categoria", key="comp_view_category_toggle")
        selected_category_type_comp = None
        category_col_map_comp = { # Use separate map if needed, but likely the same
            "Ver por categoria ABC": (CATEGORY_ABC_COL, "Categoria ABC"),
            "Ver por categoria XYZ": (CATEGORY_XYZ_COL, "Categoria XYZ"),
            "Ver por categoria ABC/XYZ": (CATEGORY_ABCXYZ_COL, "Categoria ABC/XYZ")
        }
        if view_by_category_comp:
             # Check if classification data exists
            if CLASSIFICATION_DF_KEY not in st.session_state or st.session_state[CLASSIFICATION_DF_KEY] is None or st.session_state[CLASSIFICATION_DF_KEY].empty:
                st.warning(f"Para ver resultados por categoria, por favor, execute primeiro a análise ABC/XYZ na página correspondente. Os resultados da classificação não foram encontrados em `st.session_state['{CLASSIFICATION_DF_KEY}']`.")
                # Don't render the selectbox if data is missing, but leave the toggle as is.
            else:
                selected_category_type_comp = st.selectbox(
                    "Escolha o tipo de detalhamento:",
                    options=list(category_col_map_comp.keys()),
                    key="comp_category_select"
                )

        # --- Comparison Button ---
        run_comparison_button = st.button("Comparar Resultados", key="run_comparison_button")

        # --- Comparison Calculation and Display --- (Only if button clicked with valid selections)
        # Initialize state for comparison results
        if 'comparison_results_pivot' not in st.session_state:
             st.session_state.comparison_results_pivot = None
        if 'last_run_comparison_sets' not in st.session_state:
             st.session_state.last_run_comparison_sets = None
        if 'last_run_comparison_metrics' not in st.session_state:
             st.session_state.last_run_comparison_metrics = None
        # Store category state for comparison run
        if 'last_run_comparison_category_toggle' not in st.session_state:
            st.session_state.last_run_comparison_category_toggle = False
        if 'last_run_comparison_category_type' not in st.session_state:
            st.session_state.last_run_comparison_category_type = None

        if run_comparison_button:
             # Store the selections that triggered this run
             st.session_state.last_run_comparison_sets = selected_set_keys
             st.session_state.last_run_comparison_metrics = selected_metrics_comp
             st.session_state.last_run_comparison_category_toggle = view_by_category_comp
             st.session_state.last_run_comparison_category_type = selected_category_type_comp

             # Validation
             if len(selected_set_keys) < 2:
                  st.warning("Por favor, selecione pelo menos dois conjuntos para comparar.")
                  st.session_state.comparison_results_pivot = None # Clear previous results
             elif not selected_metrics_comp:
                  st.warning("Por favor, selecione pelo menos uma métrica para comparar.")
                  st.session_state.comparison_results_pivot = None # Clear previous results
             # Check again if category view is intended but data is missing
             elif view_by_category_comp and (CLASSIFICATION_DF_KEY not in st.session_state or st.session_state[CLASSIFICATION_DF_KEY] is None or st.session_state[CLASSIFICATION_DF_KEY].empty):
                  st.error(f"Erro: A opção 'Ver resultados por categoria' está ativa, mas os dados de classificação em `st.session_state['{CLASSIFICATION_DF_KEY}']` não foram encontrados ou estão vazios. Execute a análise ABC/XYZ primeiro.")
                  st.session_state.comparison_results_pivot = None
             # Also check if category view is selected BUT the category type wasn't selected (e.g. due to missing data)
             elif view_by_category_comp and not selected_category_type_comp:
                  st.warning("A opção 'Ver resultados por categoria' está ativa, mas nenhum tipo de detalhamento foi selecionado (provavelmente devido a dados de classificação em falta). Calculando resultados gerais.")
                  view_by_category_comp = False # Treat as non-categorized for calculation
             else:
                 st.divider()
                 is_categorized_comp = view_by_category_comp and selected_category_type_comp is not None
                 category_display_name_comp = None
                 if is_categorized_comp and selected_category_type_comp and selected_category_type_comp in category_col_map_comp:
                     category_display_name_comp = category_col_map_comp[selected_category_type_comp][1]
                 st.markdown(f"#### Comparação de Desempenho Detalhada" + (f" por {category_display_name_comp}" if category_display_name_comp else ""))

                 # --- Calculate metrics for each selected set (Overall or Categorized) ---
                 all_set_metrics = []
                 spinner_msg = f"Calculando métricas {('por ' + category_display_name_comp) if category_display_name_comp else 'gerais'} para os conjuntos..."
                 with st.spinner(spinner_msg):
                     calculation_errors = False
                     classification_df_comp = st.session_state.get(CLASSIFICATION_DF_KEY) # Get classification data once

                     for set_name in selected_set_keys:
                         set_data = completed_sets.get(set_name)
                         if not set_data:
                             st.warning(f"Não foi possível carregar dados para o conjunto '{set_name}'.")
                             calculation_errors = True
                             continue

                         # Call appropriate calculation function
                         monthly_metrics_df = None
                         try:
                             if is_categorized_comp and selected_category_type_comp and category_display_name_comp: # Ensure category info is valid
                                 # Ensure selected_category_type_comp is valid key before lookup
                                 if selected_category_type_comp in category_col_map_comp:
                                     category_col, category_col_name = category_col_map_comp[selected_category_type_comp]
                                     monthly_metrics_df = calculate_categorized_monthly_metrics(
                                         set_data, selected_metrics_comp, classification_df_comp, category_col, category_col_name
                                     )
                                 else: # Should not happen if toggle logic is correct, but safe check
                                     st.warning(f"Tipo de categoria inválido '{selected_category_type_comp}' encontrado. Calculando métricas gerais.")
                                     monthly_metrics_df = calculate_monthly_metrics_for_set(set_data, selected_metrics_comp)
                             else:
                                 monthly_metrics_df = calculate_monthly_metrics_for_set(set_data, selected_metrics_comp)

                             if monthly_metrics_df is None or monthly_metrics_df.empty:
                                  st.warning(f"Não foi possível calcular as métricas selecionadas para o conjunto '{set_name}' {'por categoria' if is_categorized_comp else ''}.")
                                  calculation_errors = True # Treat empty result as error for comparison
                                  continue

                             monthly_metrics_df['Set Name'] = set_name # Add identifier
                             all_set_metrics.append(monthly_metrics_df)

                         except Exception as calc_e:
                              st.error(f"Erro ao calcular métricas para o conjunto '{set_name}': {calc_e}")
                              calculation_errors = True
                              continue # Skip this set

                 if calculation_errors:
                      st.error("Ocorreram erros ao calcular métricas para um ou mais conjuntos. A comparação pode estar incompleta.")

                 if not all_set_metrics:
                      st.error("Não foi possível calcular métricas para nenhum dos conjuntos selecionados.")
                      st.session_state.comparison_results_pivot = None
                      st.stop()

                 # --- Combine and Pivot ---
                 try:
                     combined_metrics_df = pd.concat(all_set_metrics, ignore_index=True)
                     # Filter combined df for only selected metrics BEFORE pivoting (redundant if done in calc, but safe)
                     combined_metrics_df = combined_metrics_df[combined_metrics_df['Métrica'].isin(selected_metrics_comp)]

                     if combined_metrics_df.empty:
                          st.warning("Não foram encontrados dados para as métricas selecionadas nos conjuntos escolhidos.")
                          st.session_state.comparison_results_pivot = None
                     else:
                         pivot_index = ['Mês', 'Métrica']
                         if is_categorized_comp and category_display_name_comp in combined_metrics_df.columns:
                              # Add category to the index if breakdown is active
                              # Ensure category_display_name_comp is a string before inserting
                              if isinstance(category_display_name_comp, str):
                                  pivot_index.insert(1, category_display_name_comp) # e.g., ['Mês', 'Categoria ABC', 'Métrica']

                         comparison_pivot_df = combined_metrics_df.pivot_table(
                             index=pivot_index,
                             columns='Set Name',
                             values='Resultados'
                         )
                         st.session_state.comparison_results_pivot = comparison_pivot_df # Store result
                 except Exception as e:
                     st.error(f"Erro ao combinar e pivotar os resultados dos conjuntos: {e}")
                     st.session_state.comparison_results_pivot = None
                     st.stop()

                 # --- Determine Best Method (Adapts to index structure) ---
                 if st.session_state.comparison_results_pivot is not None:
                     try:
                         comparison_pivot_df = st.session_state.comparison_results_pivot # Retrieve stored df
                         set_columns_in_pivot = comparison_pivot_df.columns.tolist()
                         # Pass only the set columns to the function
                         comparison_pivot_df['Melhor Método'] = comparison_pivot_df.apply(
                            lambda row: determine_best_method(row, set_columns_in_pivot), axis=1
                         )
                         st.session_state.comparison_results_pivot = comparison_pivot_df # Update stored df
                     except Exception as e:
                         st.error(f"Erro ao determinar o melhor método: {e}")
                         # Attempt to remove column if it exists
                         if st.session_state.comparison_results_pivot is not None and 'Melhor Método' in st.session_state.comparison_results_pivot.columns:
                              st.session_state.comparison_results_pivot.drop(columns=['Melhor Método'], inplace=True)


        # --- Display Area (Filters, Table, Summary) - Adapts to categorized comparison ---
        # Check if the button was last clicked with the current selections & results exist
        if st.session_state.last_run_comparison_sets == selected_set_keys and \
           st.session_state.last_run_comparison_metrics == selected_metrics_comp and \
           st.session_state.last_run_comparison_category_toggle == view_by_category_comp and \
           st.session_state.last_run_comparison_category_type == selected_category_type_comp and \
           st.session_state.comparison_results_pivot is not None and \
           not st.session_state.comparison_results_pivot.empty:

            comparison_pivot_df = st.session_state.comparison_results_pivot # Use the calculated/stored df
            is_categorized_display = view_by_category_comp and selected_category_type_comp is not None
            category_display_name_disp = None
            if is_categorized_display and selected_category_type_comp and selected_category_type_comp in category_col_map_comp:
                category_display_name_disp = category_col_map_comp[selected_category_type_comp][1]


            # --- Filters for Comparison Table ---
            st.markdown("##### Filtrar Tabela de Comparação")
            filter_comp_col1, filter_comp_col2 = st.columns(2)
            # Get unique values from index levels
            unique_months_comp = sorted(comparison_pivot_df.index.get_level_values('Mês').unique())
            unique_metrics_comp = sorted(comparison_pivot_df.index.get_level_values('Métrica').unique())
            unique_categories_comp = []
            if is_categorized_display and category_display_name_disp:
                 try:
                     unique_categories_comp = sorted(comparison_pivot_df.index.get_level_values(category_display_name_disp).unique())
                 except KeyError:
                      st.warning(f"Não foi possível encontrar o nível do índice da categoria '{category_display_name_disp}' para filtrar.")
                      is_categorized_display = False # Fallback if index is wrong


            with filter_comp_col1:
                filter_options_month_comp = ["Todos"] + unique_months_comp
                selected_month_filter_comp = st.selectbox(
                    "Filtrar por Mês:",
                    options=filter_options_month_comp,
                    index=0,
                    key="comparison_month_filter"
                )
            with filter_comp_col2:
                # Filter metrics based on the ones calculated (in the index)
                filter_options_metric_comp = ["Todos"] + unique_metrics_comp
                selected_metric_filter_comp = st.selectbox(
                     "Filtrar por Métrica:",
                     options=filter_options_metric_comp,
                     index=0,
                     key="comparison_metric_filter"
                )

            # --- Category Filter for Comparison Table ---
            selected_category_filter_comp = "Todos"
            if is_categorized_display and unique_categories_comp:
                # Ensure category_display_name_disp is valid before using in f-string
                if category_display_name_disp:
                    filter_options_category_comp = ["Todos"] + unique_categories_comp
                    selected_category_filter_comp = st.selectbox(
                        f"Filtrar por {category_display_name_disp}:",
                        options=filter_options_category_comp,
                        index=0,
                        key="comparison_category_filter_value"
                    )


            # --- Filter Comparison Data ---
            display_comp_df = comparison_pivot_df.copy()
            if selected_month_filter_comp != "Todos":
                display_comp_df = display_comp_df[display_comp_df.index.get_level_values('Mês') == selected_month_filter_comp]

            # Apply category filter if active and valid
            if is_categorized_display and selected_category_filter_comp != "Todos" and category_display_name_disp:
                try:
                     if selected_category_filter_comp in display_comp_df.index.get_level_values(category_display_name_disp):
                          display_comp_df = display_comp_df[display_comp_df.index.get_level_values(category_display_name_disp) == selected_category_filter_comp]
                     else:
                          st.warning(f"Valor de categoria '{selected_category_filter_comp}' não encontrado nos resultados filtrados por mês/métrica.")
                except KeyError:
                     st.warning(f"Não foi possível aplicar filtro para a categoria '{category_display_name_disp}'.")


            if selected_metric_filter_comp != "Todos":
                # Ensure the selected metric filter exists in the index before filtering
                if selected_metric_filter_comp in display_comp_df.index.get_level_values('Métrica'):
                    display_comp_df = display_comp_df[display_comp_df.index.get_level_values('Métrica') == selected_metric_filter_comp]
                else:
                    st.warning(f"Métrica selecionada '{selected_metric_filter_comp}' não encontrada nos resultados atuais. Mostrando todas as métricas disponíveis.")
                    # Optionally reset the display_comp_df or just don't apply the metric filter
                    # display_comp_df = comparison_pivot_df.copy() # Reset if needed
                    pass # Or just ignore the filter if the metric isn't there

            # --- Display Comparison Table ---
            if not display_comp_df.empty:
                formatted_comp_df = display_comp_df.reset_index()
                # Get metric column name - it's always 'Métrica' after reset_index
                metric_col = 'Métrica'
                # Get category column name if applicable
                category_col_for_format = category_display_name_disp if is_categorized_display else None

                for col in selected_set_keys: # Format only the set columns
                    if col in formatted_comp_df.columns:
                        try:
                            # Ensure the column is numeric before formatting (handle N/A, strings)
                            numeric_col = pd.to_numeric(formatted_comp_df[col], errors='coerce')
                            formatted_comp_df[col] = formatted_comp_df.apply(
                               lambda row: format_metric_value(row[metric_col], numeric_col.loc[row.name]), axis=1 # Use numeric version for format
                            )
                        except Exception as format_e:
                            st.warning(f"Erro ao formatar coluna '{col}': {format_e}. Mostrando valores originais.")
                            formatted_comp_df[col] = display_comp_df[col].astype(str).values # Show original as string

                # Define columns to display based on whether it's categorized
                cols_to_display = ['Mês']
                if is_categorized_display and category_col_for_format and category_col_for_format in formatted_comp_df.columns:
                    cols_to_display.append(category_col_for_format)
                cols_to_display.append('Métrica')
                cols_to_display.extend(selected_set_keys) # Add the set columns

                if 'Melhor Método' in formatted_comp_df.columns:
                     cols_to_display.append('Melhor Método')

                # Ensure only existing columns are selected
                final_display_comp_df = formatted_comp_df[[col for col in cols_to_display if col in formatted_comp_df.columns]]
                st.dataframe(final_display_comp_df, use_container_width=True, hide_index=True)
            else:
                st.info("Não há resultados para a combinação de filtros selecionada.")

            # --- Summary Section (Adapts to categorized comparison) ---
            # Summary calculation uses the unfiltered pivot df stored in session state
            if 'Melhor Método' in comparison_pivot_df.columns:
                st.markdown(f"#### Resumo dos Resultados (Baseado nas Métricas/Categorias Selecionadas)")
                win_counts = Counter(comparison_pivot_df['Melhor Método'].dropna())
                total_comparisons = len(comparison_pivot_df['Melhor Método'].dropna())

                if total_comparisons > 0:
                    for set_name in selected_set_keys:
                        st.markdown(f"- **{set_name}** superior em **{win_counts.get(set_name, 0)}** métricas/{'categorias/' if is_categorized_display else ''}meses.")
                    if win_counts.get('Empate', 0) > 0:
                        st.markdown(f"- Empates em **{win_counts['Empate']}** métricas/{'categorias/' if is_categorized_display else ''}meses.")

                    winner = None
                    max_wins = -1
                    tie_in_wins = False
                    set_wins = {k: v for k, v in win_counts.items() if k != 'Empate'}

                    if set_wins:
                        max_wins = max(set_wins.values())
                        winners = [k for k, v in set_wins.items() if v == max_wins]
                        if len(winners) == 1:
                            winner = winners[0]
                        else:
                            tie_in_wins = True

                    if winner:
                        st.markdown(f"**Conclusão:** `{winner}` demonstra melhor desempenho geral nesta comparação.")
                    elif tie_in_wins:
                        st.markdown(f"**Conclusão:** Empate no desempenho geral entre: `{', '.join(winners)}`.")
                    elif win_counts.get('Empate', 0) == total_comparisons and total_comparisons > 0:
                         st.markdown("**Conclusão:** Empate em todas as métricas comparadas.")
                    elif total_comparisons > 0 : # Avoid saying no winner if there were simply no comparisons
                         st.markdown("**Conclusão:** Não foi possível determinar um vencedor claro com base nas vitórias.")
                    else: # total_comparisons == 0
                         st.markdown("**Conclusão:** Não há comparações válidas para determinar um vencedor.")
                else:
                    st.markdown("Não foi possível gerar um resumo (sem coluna 'Melhor Método' ou comparações válidas).")

        # Message if button was clicked but conditions weren't met or calculation failed
        elif run_comparison_button:
             if len(selected_set_keys) < 2:
                 pass # Warning already shown
             elif not selected_metrics_comp:
                 pass # Warning already shown
             elif st.session_state.comparison_results_pivot is None:
                 # This covers cases where calculation failed after warnings/errors shown during calculation
                 st.info("Não foi possível gerar a comparação devido a erros anteriores ou falta de dados.")

# Optional: Add other helper functions specific to this page below render()

# Example direct run check (for basic testing, might need session state mocking)
# if __name__ == "__main__":
#     # Mock session state if needed for direct execution
#     if 'completed_forecast_sets' not in st.session_state:
#          st.session_state.completed_forecast_sets = {} # Initialize empty
#     if 'saved_forecasts' not in st.session_state:
#          st.session_state.saved_forecasts = {} # Initialize empty
#     render() 
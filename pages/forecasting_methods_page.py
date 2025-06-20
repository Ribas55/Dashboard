import streamlit as st
import pandas as pd
import numpy as np # Added for accuracy calculation
import datetime # Added for date input default
import plotly.graph_objects as go # Added for Gauge and Bar charts
import plotly.express as px # Added for line chart
from itertools import combinations # Added for category combinations
# from src.visualizations import plot_aggregated_series  # Example import, adjust as needed
# from src.abc_xyz import categorize_abc_xyz  # Example import, adjust as needed
# Import the forecasting orchestrator
from src.forecasting import generate_forecasts, DEFAULT_SKU_COL, DEFAULT_DATE_COL, DEFAULT_VALUE_COL

# --- Helper function for gauge chart color ---
# (Could be placed elsewhere, e.g., in visualizations.py)
def get_gauge_color(value):
    if value >= 0.9:
        return "#28a745" # Green
    elif value >= 0.75:
        return "#17a2b8" # Teal (Adjusted from image - green used above)
    elif value >= 0.6:
        return "#ffc107" # Yellow
    else:
        return "#dc3545" # Red

# --- Helper function for category overlap --- (UPDATED)
def category_overlap(cat1, cat2):
    """Checks if two category selections overlap.
    Handles single chars ('A'), multi-chars ('AB'), and 'Todas'.
    Returns True if they share any specific category or if either is 'Todas'.
    Handles None inputs safely.
    """
    # Handle None inputs safely at the beginning
    if cat1 is None or cat2 is None:
        # Print a warning to help debug if this happens unexpectedly
        print(f"Warning: category_overlap called with None value(s): cat1={cat1}, cat2={cat2}")
        # Define behavior: None probably shouldn't cause an overlap.
        return False

    if cat1 == "Todas" or cat2 == "Todas":
        return True
    # Convert to sets of individual categories
    set1 = set(list(cat1))
    set2 = set(list(cat2))
    # Check for intersection
    return not set1.isdisjoint(set2)

# --- Helper function to parse saved key --- (UPDATED)
def parse_combination_key(key):
    """Parses 'ABC-XYZ' key into a tuple ('ABC', 'XYZ'). Handles potential errors."""
    if not isinstance(key, str):
        return (None, None) # Handle non-string keys gracefully
    parts = key.split('-')
    if len(parts) == 2:
        return (parts[0], parts[1])
    else:
        print(f"Warning: Could not parse combination key: {key}")
        return (None, None) # Or raise an error

# --- Helper function to get covered single combinations --- (NEW)
ALL_SINGLE_ABC = ['A', 'B', 'C']
ALL_SINGLE_XYZ = ['X', 'Y', 'Z']

def get_covered_singles(saved_combinations_set):
    """Calculates the set of single 'A-X' style keys covered by a set of saved combination keys."""
    covered_singles = set()
    for saved_key in saved_combinations_set:
        saved_abc, saved_xyz = parse_combination_key(saved_key)
        if saved_abc is None or saved_xyz is None: continue # Skip if key is invalid

        # Determine individual ABCs covered by saved_abc
        if saved_abc == "Todas":
            abc_covered = ALL_SINGLE_ABC
        else:
            abc_covered = list(saved_abc)

        # Determine individual XYZs covered by saved_xyz
        if saved_xyz == "Todas":
            xyz_covered = ALL_SINGLE_XYZ
        else:
            xyz_covered = list(saved_xyz)

        # Add all combinations of covered singles
        for a in abc_covered:
            for x in xyz_covered:
                # Ensure only valid single letters are considered (in case of unexpected input)
                if a in ALL_SINGLE_ABC and x in ALL_SINGLE_XYZ:
                    covered_singles.add(f"{a}-{x}")
    return covered_singles

# --- End Helpers ---

def render(): # Wrap existing code in a render function
    # --- Page Configuration --- (Removed - handled by app.py)
    # st.set_page_config(page_title="Métodos de Forecasting", layout="wide")

    st.title("🔮 Métodos de Forecasting - Visualização") # Added icon to title

    # --- Define Key Column Names --- # Moved earlier
    # Adjust these if your column names are different in the main dataframe
    DATE_COL_RAW = 'invoice_date' # Original date column before converting to period
    SKU_COL = 'sku'             # Corrected based on actual DataFrame columns
    VALUE_COL = 'sales_value'
    FAMILY_COL = 'family'
    MANAGER_COL = 'commercial_manager'
    PERIOD_COL = DEFAULT_DATE_COL # Defined in forecasting.py, 'Period' - Assumed correct
    abc_col = 'abc_class'         # Defined here for consistency
    xyz_col = 'xyz_class'         # Defined here for consistency

    # --- Initialize Session State for Results --- (Added)
    if 'results_table' not in st.session_state:
        st.session_state.results_table = None
    if 'analysis_context' not in st.session_state:
        st.session_state.analysis_context = None
    # Add state for saved forecasts
    if 'saved_forecasts' not in st.session_state:
        st.session_state.saved_forecasts = {} # Dict: key="ABC-XYZ", value={results: df, context: dict}
    if 'saved_combinations' not in st.session_state:
        # Stores keys like "A-X", "AB-Y", "Todas-Z"
        st.session_state.saved_combinations = set() # Set: {"ABC-XYZ"}
    # Add state for SKU ponderation weights
    if 'ponderation_sku_weights' not in st.session_state:
        st.session_state.ponderation_sku_weights = {
            'avg_2023': 10, 'avg_ytd': 10, 'last_3_months': 25,
            'last_12_months': 25, 'budget': 30, 'cagr': 0
        }
    # Temporary state for loading
    if '_load_forecast_data' not in st.session_state:
        st.session_state._load_forecast_data = None
    if '_load_context' not in st.session_state:
        st.session_state._load_context = None
    if '_load_weights' not in st.session_state:
        st.session_state._load_weights = None
    # State variable to track pending actions from buttons
    if '_pending_action' not in st.session_state:
        st.session_state._pending_action = None
    # Added completed_forecast_sets
    if 'completed_forecast_sets' not in st.session_state:
        st.session_state.completed_forecast_sets = {} # Dict: key=set_name, value={dict_of_saved_forecasts}

    # --- Handle Pending Actions (Load/Remove Forecasts/Sets) ---
    pending_action = st.session_state._pending_action
    if pending_action:
        action_type = pending_action.get('type')
        action_key = pending_action.get('key') # Key is "ABC-XYZ" format for forecasts, set_name for sets

        # --- Handle Individual Forecast Actions ---
        if action_type == 'load' and action_key:
            st.info(f"Carregando forecast '{action_key}'...")
            saved_item = st.session_state.saved_forecasts.get(action_key)
            if saved_item:
                # Store data in temporary state variables for widget defaults
                st.session_state._load_forecast_data = saved_item.get('results')
                st.session_state._load_context = saved_item.get('context')
                st.session_state._load_weights = saved_item.get('weights')
            st.session_state._pending_action = None # Clear the action
            st.rerun()

        elif action_type == 'remove' and action_key:
            if action_key in st.session_state.saved_forecasts:
                del st.session_state.saved_forecasts[action_key]
            if action_key in st.session_state.saved_combinations:
                st.session_state.saved_combinations.remove(action_key)
            st.success(f"Forecast '{action_key}' removido.")
            # --- Clear current results when removing ---
            st.session_state.results_table = None
            st.session_state.analysis_context = None
            # --- End Clear ---
            st.session_state._pending_action = None # Clear the action
            st.rerun()

        # --- Handle Forecast Set Actions ---
        elif action_type == 'load_set' and action_key:
            st.info(f"Carregando conjunto de forecasts '{action_key}'...")
            set_data = st.session_state.completed_forecast_sets.get(action_key)
            if set_data:
                all_results_dfs = []
                # Retrieve the abc_xyz_df for merging base classifications
                abc_xyz_data_for_merge = st.session_state.get('abc_xyz_data')
                if abc_xyz_data_for_merge is None or 'sku' not in abc_xyz_data_for_merge.columns:
                     st.error("Resultados ABC/XYZ não encontrados ou inválidos. Impossível carregar o conjunto.")
                else:
                    for forecast_key, saved_forecast in set_data.items():
                        if 'results' in saved_forecast:
                            all_results_dfs.append(saved_forecast['results'].copy()) # Append a copy

                    if not all_results_dfs:
                        st.warning(f"Nenhum dado de forecast encontrado no conjunto '{action_key}'.")
                        st.session_state.results_table = None
                        st.session_state.analysis_context = None
                    else:
                        try:
                            combined_df = pd.concat(all_results_dfs, ignore_index=True)

                            # Ensure necessary columns exist before deduplication
                            required_cols = [SKU_COL, PERIOD_COL]
                            if not all(col in combined_df.columns for col in required_cols):
                                st.error(f"Erro: Colunas essenciais ({', '.join(required_cols)}) em falta nos dados combinados do conjunto.")
                            else:
                                # Add base ABC/XYZ classification
                                combined_df = pd.merge(combined_df,
                                                       abc_xyz_data_for_merge[[SKU_COL, abc_col, xyz_col]],
                                                       on=SKU_COL,
                                                       how='left')

                                # Deduplicate based on SKU and Period, keeping the first entry encountered
                                combined_df.drop_duplicates(subset=[SKU_COL, PERIOD_COL], keep='first', inplace=True)

                                # Store combined results
                                st.session_state.results_table = combined_df
                                st.session_state.analysis_context = {'type': 'set', 'name': action_key} # Mark as loaded set
                                st.success(f"Conjunto '{action_key}' carregado com sucesso.")

                        except Exception as e:
                            st.error(f"Erro ao combinar os dados do conjunto '{action_key}': {e}")
                            st.session_state.results_table = None
                            st.session_state.analysis_context = None
            else:
                 st.warning(f"Conjunto '{action_key}' não encontrado.")

            st.session_state._pending_action = None # Clear the action
            st.rerun()

        elif action_type == 'remove_set' and action_key:
            if action_key in st.session_state.completed_forecast_sets:
                del st.session_state.completed_forecast_sets[action_key]
                st.success(f"Conjunto '{action_key}' removido.")
                # If the removed set was the one currently loaded, clear the results
                if st.session_state.analysis_context and st.session_state.analysis_context.get('type') == 'set' and st.session_state.analysis_context.get('name') == action_key:
                    st.session_state.results_table = None
                    st.session_state.analysis_context = None
            st.session_state._pending_action = None # Clear the action
            st.rerun()
        else:
             # Clear invalid action
             st.session_state._pending_action = None

    # --- Load Data --- # Corrected session state keys
    if 'sales_data' not in st.session_state or 'abc_xyz_data' not in st.session_state:
        st.warning("Por favor, carregue os dados e execute a análise ABC/XYZ primeiro.")
        st.stop()

    df = st.session_state.get('sales_data') # Use 'sales_data'
    abc_xyz_df = st.session_state.get('abc_xyz_data') # Use 'abc_xyz_data'

    if df is None or df.empty or abc_xyz_df is None or abc_xyz_df.empty:
         st.warning("Dados de vendas ou resultados ABC/XYZ não encontrados ou vazios.") # Updated warning message slightly
         st.stop()
    # Add assertion for df after checks
    assert df is not None, "DataFrame 'df' should not be None here."
    assert abc_xyz_df is not None, "DataFrame 'abc_xyz_df' should not be None here."

    # Ensure date column is datetime and get available months
    date_col = DATE_COL_RAW
    if date_col not in df.columns:
        st.error(f"Coluna de data '{date_col}' não encontrada no DataFrame.")
        st.stop()
    if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df.dropna(subset=[date_col], inplace=True) # Remove rows where date conversion failed

    # Add a 'Period' column if it doesn't exist
    if PERIOD_COL not in df.columns:
        df[PERIOD_COL] = df[date_col].dt.to_period('M')

    all_periods = sorted(df[PERIOD_COL].unique())
    all_periods_str = [p.strftime('%Y-%m') for p in all_periods] # Format for display


    # --- Main Area Filters ---
    st.markdown("### Configurações do Forecasting")

    # --- Determine if Horizon is Locked ---
    horizon_locked = bool(st.session_state.saved_forecasts)
    locked_start_str = None
    locked_end_str = None
    lock_tooltip = ""
    if horizon_locked:
        # Get the first saved forecast key and its context
        first_saved_key = next(iter(st.session_state.saved_forecasts))
        first_saved_context = st.session_state.saved_forecasts[first_saved_key].get('context', {})
        locked_start_str = first_saved_context.get('start')
        locked_end_str = first_saved_context.get('end')
        lock_tooltip = f"Horizonte bloqueado pelo primeiro forecast guardado ({locked_start_str} - {locked_end_str}). Remova todos os forecasts para alterar."
    # --- End Horizon Lock Logic ---

    col1, col2 = st.columns(2)

    with col1:
        # 1. Select Forecasting Method (Updated)
        method_options = [
            "Selecione...",
            "ARIMA (0,1,1)", "ARIMA (1,1,1)", "ARIMA (2,1,2)",
            "SES (alpha=0.1)", "SES (alpha=0.3)",
            "SMA (N=1)", "SMA (N=2)", "SMA (N=3)", "SMA (N=6)", "SMA (N=9)",
            "TSB (alpha_d=0.1, alpha_p=0.1)", # Added TSB option
            "XGBoost (lags=2)", "XGBoost (lags=3)", "XGBoost (lags=6)", # Added XGBoost options
            "Linear Regression (lags=2)", "Linear Regression (lags=3)", "Linear Regression (lags=6)", # Added Linear Regression options
            "Previsão Ponderada por SKU" # Added SKU Weighted Forecast
        ]
        # Get default method index from temporary load state if available
        default_method_index = 0
        if st.session_state._load_context:
            try:
                default_method_index = method_options.index(st.session_state._load_context['method'])
            except ValueError:
                pass # Keep 0 if loaded value not in options

        selected_method = st.selectbox(
            "Selecionar Método de Forecasting:",
            method_options,
            index=default_method_index, # Set index based on loaded state
            key='forecast_method_selector' # Keep the key
        )

        # 2. Time Horizon (New)
        st.markdown("Horizonte de Previsão:")
        if all_periods_str:
            # --- Calculate default/locked indices --- #
            start_idx = 0
            end_idx = len(all_periods_str) - 1

            if horizon_locked and locked_start_str and locked_end_str:
                # If locked, use the locked dates
                try:
                    start_idx = all_periods_str.index(locked_start_str)
                    end_idx = all_periods_str.index(locked_end_str)
                except ValueError:
                    st.warning("Datas do forecast guardado não encontradas nas opções atuais. Usando defaults.")
                    # Keep default 0 and len-1 if locked dates not found
                    pass
            elif st.session_state._load_context:
                # If not locked, but loading, use loaded context
                try:
                    start_idx = all_periods_str.index(st.session_state._load_context['start'])
                    end_idx = all_periods_str.index(st.session_state._load_context['end'])
                except ValueError:
                    pass # Keep default if saved context not in current options
            # --- End Index Calculation --- #

            start_period_str = st.selectbox(
                "De:",
                options=all_periods_str,
                index=start_idx,
                key='start_period',
                disabled=horizon_locked, # Disable if locked
                help=lock_tooltip if horizon_locked else "Selecione o mês inicial para a previsão." # Add tooltip
            )
            end_period_str = st.selectbox(
                "Até:",
                options=all_periods_str,
                index=end_idx,
                key='end_period',
                disabled=horizon_locked, # Disable if locked
                help=lock_tooltip if horizon_locked else "Selecione o mês final para a previsão." # Add tooltip
            )
            # Convert selected strings back to Period objects for filtering
            start_period = pd.Period(start_period_str, freq='M') if start_period_str else None
            end_period = pd.Period(end_period_str, freq='M') if end_period_str else None

            # Validate period selection
            if start_period and end_period and start_period > end_period:
                st.warning("O período 'De' não pode ser posterior ao período 'Até'.")
                valid_periods = False
            else:
                valid_periods = True
        else:
            st.warning("Não há períodos disponíveis nos dados carregados.")
            start_period = None
            end_period = None
            valid_periods = False


    with col2:
        # 3. Select Combination Type (Revised Filtering)
        st.markdown("Filtro de SKUs por Categoria ABC/XYZ (Baseado em 2024):")

        # --- Generate Full Option Lists --- Always generate full lists
        all_abc_single = sorted(abc_xyz_df[abc_col].unique())
        all_xyz_single = sorted(abc_xyz_df[xyz_col].unique())

        abc_combinations = []
        for i in range(2, len(all_abc_single) + 1):
            for combo in combinations(all_abc_single, i):
                abc_combinations.append("".join(sorted(combo)))
        xyz_combinations = []
        for i in range(2, len(all_xyz_single) + 1):
            for combo in combinations(all_xyz_single, i):
                xyz_combinations.append("".join(sorted(combo)))

        full_abc_options = ["Todas"] + all_abc_single + sorted(abc_combinations)
        full_xyz_options = ["Todas"] + all_xyz_single + sorted(xyz_combinations)
        # --- End Option List Generation ---

        # --- Get current selections (from previous run state) --- # Renamed comment
        current_abc_selection = st.session_state.get("select_abc_filter", "Todas")
        current_xyz_selection = st.session_state.get("select_xyz_filter", "Todas")

        # --- Filter options based on saved combinations and the *other* selection --- # Reworked logic
        saved_combinations_set = st.session_state.get('saved_combinations', set())
        parsed_saved_combos = [(s_abc, s_xyz) for s_abc, s_xyz in [parse_combination_key(k) for k in saved_combinations_set] if s_abc is not None]

        # Filter ABC options based on current_xyz_selection and saved combos
        filtered_abc_options = []
        for potential_abc in full_abc_options:
            is_problematic = False
            for saved_abc, saved_xyz in parsed_saved_combos:
                # Check if choosing potential_abc AND current_xyz overlaps with a saved combo
                if category_overlap(potential_abc, saved_abc) and category_overlap(current_xyz_selection, saved_xyz):
                    is_problematic = True
                    break # Found an overlap, no need to check other saved combos
            if not is_problematic:
                filtered_abc_options.append(potential_abc)

        # Filter XYZ options based on current_abc_selection and saved combos
        filtered_xyz_options = []
        for potential_xyz in full_xyz_options:
            is_problematic = False
            for saved_abc, saved_xyz in parsed_saved_combos:
                 # Check if choosing current_abc AND potential_xyz overlaps with a saved combo
                if category_overlap(current_abc_selection, saved_abc) and category_overlap(potential_xyz, saved_xyz):
                    is_problematic = True
                    break # Found an overlap
            if not is_problematic:
                filtered_xyz_options.append(potential_xyz)


        # --- Determine final selection and index --- # Recalculated based on filtered lists
        final_abc_selection = current_abc_selection
        final_xyz_selection = current_xyz_selection

        # Adjust if loading a forecast
        if st.session_state._load_context:
            load_context_abc = st.session_state._load_context.get('abc')
            load_context_xyz = st.session_state._load_context.get('xyz')

            # Use loaded value only if it is present in the *filtered* list for this run
            if load_context_abc in filtered_abc_options:
                final_abc_selection = load_context_abc
            elif final_abc_selection not in filtered_abc_options:
                 final_abc_selection = "Todas"

            if load_context_xyz in filtered_xyz_options:
                final_xyz_selection = load_context_xyz
            elif final_xyz_selection not in filtered_xyz_options:
                 final_xyz_selection = "Todas"
        else:
            # Ensure current selections are valid in the *filtered* lists
            if current_abc_selection not in filtered_abc_options:
                final_abc_selection = "Todas"
            if current_xyz_selection not in filtered_xyz_options:
                final_xyz_selection = "Todas"

        # Calculate index based on the final determined selection values
        try:
            abc_index = filtered_abc_options.index(final_abc_selection)
        except ValueError:
            abc_index = 0 # Default to "Todas"

        try:
            xyz_index = filtered_xyz_options.index(final_xyz_selection)
        except ValueError:
            xyz_index = 0 # Default to "Todas"

        # --- Display Selectboxes ---
        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            selected_abc = st.selectbox(
                "Categoria ABC:",
                filtered_abc_options,
                index=abc_index,
                help="Selecione a(s) categoria(s) ABC. Opções são filtradas baseadas em forecasts já guardados.",
                key="select_abc_filter" # Use a specific key
            )
        with sub_col2:
            selected_xyz = st.selectbox(
                "Categoria XYZ:",
                filtered_xyz_options,
                index=xyz_index,
                help="Selecione a(s) categoria(s) XYZ. Opções são filtradas baseadas em forecasts já guardados.",
                key="select_xyz_filter" # Use a specific key
            )

    # --- Apply Loaded State (After Widgets) ---
    if st.session_state._load_context:
        st.session_state.results_table = st.session_state._load_forecast_data
        st.session_state.analysis_context = st.session_state._load_context
        if st.session_state._load_weights:
            st.session_state.ponderation_sku_weights = st.session_state._load_weights

        # Clean up temporary state
        st.session_state._load_forecast_data = None
        st.session_state._load_context = None
        st.session_state._load_weights = None


    # --- Conditionally Display Weights for SKU Ponderation ---
    weights_valid = True # Assume weights are valid unless proven otherwise
    ponderation_weights = None # Initialize to None
    if selected_method == "Previsão Ponderada por SKU":
        st.markdown("#### Pesos dos Componentes (%)")
        with st.form("ponderation_sku_weights_form"):
            weight_keys = list(st.session_state.ponderation_sku_weights.keys())
            num_weight_cols = len(weight_keys)
            weight_cols = st.columns(num_weight_cols)
            current_weights_input = {}

            for idx, key in enumerate(weight_keys):
                with weight_cols[idx]:
                    label = key.replace('_', ' ').title()
                    if key == 'cagr':
                        label = "Cresc. (YTD vs 23)"
                    min_val, max_val, step = 0, 100, 1
                    default_val = st.session_state.ponderation_sku_weights.get(key, 0)
                    is_disabled = False # Budget slider always enabled for now

                    current_weights_input[key] = st.number_input(
                        label=label,
                        min_value=min_val, max_value=max_val,
                        value=default_val, step=step,
                        key=f"ponderation_sku_{key}",
                        disabled=is_disabled,
                        help=f"Percentagem de peso para o componente '{label}'."
                    )

            submitted = st.form_submit_button("Aplicar Pesos")
            if submitted:
                total_weight_input = sum(current_weights_input.values())
                if total_weight_input == 100:
                    st.session_state.ponderation_sku_weights = current_weights_input
                    st.success("Pesos atualizados com sucesso!")
                else:
                    st.warning(f"A soma dos pesos ({total_weight_input}%) deve ser 100%. Pesos não foram aplicados.")

        current_total_weight_state = sum(st.session_state.ponderation_sku_weights.values())
        color = "green" if current_total_weight_state == 100 else "red"
        st.markdown(f"**Soma Atual dos Pesos:** <span style='color:{color};'>{current_total_weight_state}%</span>", unsafe_allow_html=True)

        if current_total_weight_state != 100:
            weights_valid = False
            st.warning("A soma dos pesos deve ser 100% para gerar a Previsão Ponderada.")
        else:
            ponderation_weights = st.session_state.ponderation_sku_weights.copy()

    # --- Check if combination overlaps with a saved forecast --- # Uses selection from widgets
    apply_disabled = False
    conflict_message = ""
    # Get selections made in *this* run from the widgets
    actual_selected_abc = selected_abc
    actual_selected_xyz = selected_xyz

    for saved_key in saved_combinations_set:
        saved_abc, saved_xyz = parse_combination_key(saved_key)
        if saved_abc is None: continue
        # Check overlap between the actual current selection and saved combinations
        if category_overlap(actual_selected_abc, saved_abc) and category_overlap(actual_selected_xyz, saved_xyz):
            apply_disabled = True
            conflict_message = f"A seleção atual ({actual_selected_abc} / {actual_selected_xyz}) conflita com um forecast já guardado ('{saved_key}'). Remova o forecast guardado ou altere a seleção."
            break # Found a conflict

    if apply_disabled:
         st.warning(conflict_message)

    # --- Apply Button --- (Added disabled state)
    # Also disable if Ponderada SKU is selected and weights are invalid
    apply_button_disabled = apply_disabled or (selected_method == "Previsão Ponderada por SKU" and not weights_valid)
    apply_forecast = st.button("Aplicar Método e Gerar Previsão", disabled=apply_button_disabled)

    # --- Display Saved Forecasts ---
    st.markdown("### Forecasts Guardados")
    if not st.session_state.saved_forecasts:
        st.info("Nenhum forecast guardado ainda.")
    else:
        st.markdown("Clique no nome para carregar, ou em ❌ para excluir:")
        saved_keys_list = sorted(list(st.session_state.saved_forecasts.keys())) # Get updated keys

        num_cols = 4
        item_cols = st.columns(num_cols)
        col_idx = 0
        action_set_this_run = False

        for combination_key in saved_keys_list: # combination_key is "ABC-XYZ"
            if action_set_this_run: break

            with item_cols[col_idx % num_cols]:
                if combination_key in st.session_state.saved_forecasts:
                    context = st.session_state.saved_forecasts[combination_key]['context']
                    button_cols = st.columns([5, 1])

                    with button_cols[0]:
                        # Use the combination_key directly which is "ABC-XYZ" format
                        display_text = f"{combination_key} ({context['method']}, {context['start']}-{context['end']})"
                        load_button_key = f"load_{combination_key}" # Unique key for button
                        if st.button(display_text, key=load_button_key, help=f"Carregar forecast: {display_text}", use_container_width=True):
                           if not action_set_this_run:
                               st.session_state._pending_action = {'type': 'load', 'key': combination_key}
                               action_set_this_run = True
                               st.rerun()

                    with button_cols[1]:
                        remove_button_key = f"remove_{combination_key}" # Unique key for button
                        if st.button("❌", key=remove_button_key, help=f"Remover forecast {combination_key}", use_container_width=True):
                            if not action_set_this_run:
                                st.session_state._pending_action = {'type': 'remove', 'key': combination_key}
                                action_set_this_run = True
                                st.rerun()

            col_idx += 1

    # --- Display Saved Forecast Sets --- (NEW SECTION)
    st.markdown("### Conjuntos de Forecasts Guardados")
    if not st.session_state.completed_forecast_sets:
        st.info("Nenhum conjunto completo de forecasts guardado ainda.")
    else:
        st.markdown("Clique no nome para carregar, ou em ❌ para excluir:")
        saved_set_keys_list = sorted(list(st.session_state.completed_forecast_sets.keys()))

        num_set_cols = 4 # Reuse or define a different number of columns if desired
        set_item_cols = st.columns(num_set_cols)
        set_col_idx = 0
        action_set_this_run_sets = False # Separate flag for set actions

        for set_name in saved_set_keys_list:
            if action_set_this_run_sets: break

            with set_item_cols[set_col_idx % num_set_cols]:
                if set_name in st.session_state.completed_forecast_sets:
                    set_button_cols = st.columns([5, 1])

                    with set_button_cols[0]:
                        load_set_button_key = f"load_set_{set_name}" # Unique key
                        if st.button(f"📂 {set_name}", key=load_set_button_key, help=f"Carregar conjunto: {set_name}", use_container_width=True):
                           if not action_set_this_run_sets:
                               st.session_state._pending_action = {'type': 'load_set', 'key': set_name}
                               action_set_this_run_sets = True
                               st.rerun()

                    with set_button_cols[1]:
                        remove_set_button_key = f"remove_set_{set_name}" # Unique key
                        if st.button("❌", key=remove_set_button_key, help=f"Remover conjunto {set_name}", use_container_width=True):
                            if not action_set_this_run_sets:
                                st.session_state._pending_action = {'type': 'remove_set', 'key': set_name}
                                action_set_this_run_sets = True
                                st.rerun()

            set_col_idx += 1


    # --- Check for Completion and Offer "Save All" --- # Reworked Logic
    st.divider()
    target_singles = { # The 9 fundamental combinations we need covered
        f"{a}-{x}"
        for a in ALL_SINGLE_ABC
        for x in ALL_SINGLE_XYZ
    }
    current_saved_keys = st.session_state.get('saved_combinations', set())
    covered_singles = get_covered_singles(current_saved_keys)

    all_singles_covered = (covered_singles == target_singles)

    if all_singles_covered:
        st.markdown("### Finalizar e Guardar Conjunto Completo")
        st.success("Todas as 9 combinações individuais (AX, AY... CZ) estão cobertas pelos forecasts guardados!")
        set_name = st.text_input("Nome para este conjunto de forecasts:", key="save_all_set_name", help="Escolha um nome único para o conjunto atual de forecasts guardados.")

        if st.button("💾 Salvar Tudo e Recomeçar"):
            if set_name:
                if set_name in st.session_state.completed_forecast_sets:
                    st.warning(f"Já existe um conjunto guardado com o nome '{set_name}'. Escolha outro nome.")
                else:
                    # Store a deep copy of the *current* set of saved forecasts
                    try:
                        from copy import deepcopy
                        # Save the exact set of forecasts that achieved coverage
                        completed_set_data = deepcopy(st.session_state.saved_forecasts)

                        if not completed_set_data: # Check if dictionary is empty (shouldn't happen if all covered)
                             st.error("Erro interno: Não foram encontrados forecasts guardados para salvar o conjunto.")
                        else:
                            st.session_state.completed_forecast_sets[set_name] = completed_set_data
                            st.success(f"Conjunto de forecasts '{set_name}' (com {len(completed_set_data)} item(ns)) guardado com sucesso!")

                            # Clear current saved forecasts and results
                            st.session_state.saved_forecasts = {}
                            st.session_state.saved_combinations = set()
                            st.session_state.results_table = None
                            st.session_state.analysis_context = None
                            st.rerun()

                    except Exception as e:
                        st.error(f"Erro ao copiar dados para guardar o conjunto: {e}")

            else:
                st.warning("Por favor, introduza um nome para o conjunto.")
    else:
        # Show progress based on coverage
        missing_singles = target_singles - covered_singles
        num_missing = len(missing_singles)
        if num_missing < len(target_singles): # Show only if some progress is made
            missing_str = ", ".join(sorted(list(missing_singles)))
            st.info(f"Cobertas {len(covered_singles)} de 9 combinações individuais base. Faltam: {missing_str}.")


    # --- Generation Logic (Inside Button Click) ---
    if apply_forecast and not apply_disabled:
        # --- Clear previous results immediately on button press --- (NEW)
        st.session_state.results_table = None
        st.session_state.analysis_context = None
        # --- End Clear ---

        if selected_method == "Selecione...":
            st.warning("Por favor, selecione um método de forecasting.")
            # st.session_state.results_table = None # Already cleared above
            # st.session_state.analysis_context = None
        elif not valid_periods or not start_period or not end_period:
            st.warning("Por favor, selecione um horizonte de previsão válido.")
            # st.session_state.results_table = None # Already cleared above
            # st.session_state.analysis_context = None
        else:
            run_forecast = True
            current_ponderation_weights_for_run = None
            if selected_method == "Previsão Ponderada por SKU":
                weights_check = st.session_state.ponderation_sku_weights
                if sum(weights_check.values()) != 100:
                    st.error("Erro: A soma dos pesos para a Previsão Ponderada não é 100%. Aplique pesos válidos.")
                    run_forecast = False
                else:
                    current_ponderation_weights_for_run = {k: v / 100.0 for k, v in weights_check.items()}

            if run_forecast:
                # --- Get Target SKU List (handling multi-category) ---
                target_skus_df = abc_xyz_df.copy()

                # Filter by ABC selection
                if selected_abc != "Todas" and selected_abc is not None: # Add check for None
                    target_skus_df = target_skus_df[target_skus_df[abc_col].isin(list(selected_abc))]
                elif selected_abc is None:
                    st.warning("Seleção ABC inválida (None). Tratando como \'Todas\'.") # Optional warning

                # Filter by XYZ selection
                if selected_xyz != "Todas" and selected_xyz is not None: # Add check for None
                    target_skus_df = target_skus_df[target_skus_df[xyz_col].isin(list(selected_xyz))]
                elif selected_xyz is None:
                     st.warning("Seleção XYZ inválida (None). Tratando como \'Todas\'.") # Optional warning

                if 'sku' not in target_skus_df.columns:
                     st.error("Coluna 'sku' não encontrada nos resultados da classificação ABC/XYZ.")
                     st.stop() # Stop execution

                filtered_sku_list = target_skus_df['sku'].tolist()
                if not filtered_sku_list:
                    st.warning(f"Nenhum SKU encontrado para a combinação de filtros ABC='{selected_abc}' e XYZ='{selected_xyz}'. Nenhuma previsão será gerada.")
                    # Clear results and stop
                    st.session_state.results_table = None
                    st.session_state.analysis_context = None
                    st.stop()

                # Align SKU types
                try:
                    if df[SKU_COL].dtype != target_skus_df['sku'].dtype:
                        common_type = str
                        if not pd.api.types.is_string_dtype(df[SKU_COL]):
                             df[SKU_COL] = df[SKU_COL].astype(common_type)
                        filtered_sku_list_typed = [common_type(sku) for sku in filtered_sku_list]
                    else:
                        filtered_sku_list_typed = filtered_sku_list
                except Exception as e:
                    st.error(f"Erro ao alinhar tipos de SKU para filtragem: {e}")
                    st.stop() # Stop execution

                # --- Determine Input DF for Generation ---
                if selected_method == "Previsão Ponderada por SKU":
                    input_df_for_generation = df # Use unfiltered
                    print(f"PonderadaSKU: Usando DataFrame completo (shape: {input_df_for_generation.shape}) para geração.")
                else:
                    input_df_for_generation = df[df[SKU_COL].isin(filtered_sku_list_typed)].copy() # Use filtered
                    if input_df_for_generation.empty:
                        st.warning(f"Método {selected_method}: DataFrame vazio após filtrar por SKUs ({selected_abc}/{selected_xyz}). Nenhuma previsão será gerada.")
                        st.session_state.results_table = None # Clear results
                        st.session_state.analysis_context = None
                        st.stop() # Stop execution
                    print(f"Método {selected_method}: Usando DataFrame filtrado (shape: {input_df_for_generation.shape}) para geração.")

                # --- Generate Forecasts ---
                num_target_skus = len(filtered_sku_list_typed) # Use the count of the *target* list
                spinner_base_text = f"Gerando previsões ({selected_method}) para {num_target_skus} SKUs ({selected_abc} / {selected_xyz})"

                generation_spinner_text = spinner_base_text
                if selected_method == "Previsão Ponderada por SKU":
                    num_processed_skus = len(input_df_for_generation[SKU_COL].unique())
                    generation_spinner_text = f"{spinner_base_text} (processando {num_processed_skus} SKUs)..."
                else:
                     generation_spinner_text = f"{spinner_base_text}..."

                with st.spinner(generation_spinner_text):
                    try:
                        forecast_args = {
                            "filtered_df": input_df_for_generation,
                            "selected_method_str": selected_method,
                            "start_period": start_period,
                            "end_period": end_period,
                            "sku_col": SKU_COL,
                            "date_col": PERIOD_COL,
                            "value_col": VALUE_COL,
                            "family_col": FAMILY_COL,
                            "manager_col": MANAGER_COL,
                            "weights": current_ponderation_weights_for_run,
                            "target_skus": filtered_sku_list_typed # Pass the target list
                        }
                        forecast_results_df = generate_forecasts(**forecast_args)

                    except Exception as e:
                        st.error(f"Ocorreu um erro durante a geração das previsões: {e}")
                        st.session_state.results_table = None
                        st.session_state.analysis_context = None
                        st.stop()

                if forecast_results_df.empty:
                    st.warning("Nenhuma previsão foi gerada com sucesso. Verifique os dados ou o método selecionado.")
                    st.session_state.results_table = None
                    st.session_state.analysis_context = None
                    st.stop()

                # --- Prepare Actual Sales and Details (Based on TARGET SKUs) ---
                try:
                    # Filter main df using the *final* target SKU list
                    df_actuals_details = df[df[SKU_COL].isin(filtered_sku_list_typed)].copy()
                    if df_actuals_details.empty:
                         print("Warning: No actual sales data found for the target SKUs.")
                         actuals_df = pd.DataFrame(columns=[SKU_COL, PERIOD_COL, 'Valor Real de Vendas'])
                         sku_details = pd.DataFrame(columns=[SKU_COL, FAMILY_COL, MANAGER_COL])
                    else:
                        actuals_filtered = df_actuals_details[
                            (df_actuals_details[PERIOD_COL] >= start_period) &
                            (df_actuals_details[PERIOD_COL] <= end_period)
                        ]
                        actuals_df = actuals_filtered.groupby([SKU_COL, PERIOD_COL], observed=False)[VALUE_COL].sum().reset_index()
                        actuals_df.rename(columns={VALUE_COL: 'Valor Real de Vendas'}, inplace=True)
                        sku_details = df_actuals_details[[SKU_COL, FAMILY_COL, MANAGER_COL]].drop_duplicates(subset=[SKU_COL])
                except Exception as e:
                    st.error(f"Erro ao preparar dados reais ou detalhes do SKU para SKUs alvo: {e}")
                    st.stop()


                # --- Merge Forecasts, Actuals, Details ---
                try:
                    results_table = pd.merge(forecast_results_df, actuals_df, on=[SKU_COL, PERIOD_COL], how='left')
                    results_table = pd.merge(results_table, sku_details, on=SKU_COL, how='left')
                except Exception as e:
                    st.error(f"Erro ao combinar resultados das previsões com dados reais: {e}")
                    st.stop()

                # --- Calculate Accuracy (Assertividade) ---
                try:
                    real = results_table['Valor Real de Vendas'].fillna(0)
                    forecast = results_table['Valor Previsto'].fillna(0)
                    abs_error = abs(real - forecast)
                    max_val = np.maximum(real, forecast)
                    accuracy = np.where(max_val == 0, np.where(abs_error == 0, 1.0, 0.0), 1 - (abs_error / max_val))
                    results_table['Assertividade'] = np.nan_to_num(accuracy, nan=0.0)
                except Exception as e:
                    st.error(f"Erro ao calcular a assertividade: {e}")
                    results_table['Assertividade'] = 0.0 # Assign default on error

                # --- Store results in session state ---
                st.session_state.results_table = results_table
                st.session_state.analysis_context = {
                    'method': selected_method,
                    'start': start_period_str,
                    'end': end_period_str,
                    'abc': selected_abc, # Store selected ABC (can be multi-char/Todas)
                    'xyz': selected_xyz  # Store selected XYZ (can be multi-char/Todas)
                }
                st.success("Previsão gerada com sucesso!")


    # --- Display Area (Conditional on Results Existing in Session State) ---
    st.divider()

    if st.session_state.get('results_table') is not None:
        results_table = st.session_state.results_table
        analysis_context = st.session_state.analysis_context
        assert results_table is not None, "results_table should not be None here."
        assert analysis_context is not None, "analysis_context should not be None here."

        # --- Update Subheader based on context ---
        if analysis_context.get('type') == 'set':
            st.subheader(f"Resultados Combinados do Conjunto '{analysis_context['name']}'")
            st.markdown(f"**Período Original dos Forecasts:** (Varia entre os forecasts individuais do conjunto)") # Indicate period might vary
        else:
            st.subheader(f"Resultados do Forecasting ({analysis_context['method']}) para o período {analysis_context['start']} a {analysis_context['end']}")
            st.markdown(f"**Categoria:** {analysis_context['abc']} / {analysis_context['xyz']}")
        # --- End Subheader Update ---

        # --- Display Final Table ---
        st.markdown("### Tabela de Resultados")
        display_rename_map = {
            FAMILY_COL: 'Familia', SKU_COL: 'SKU', 'Valor Previsto': 'Forecast',
            'Valor Real de Vendas': 'Vendas', PERIOD_COL: 'Period', 'Assertividade': 'Assertividade'
        }
        display_order = ['Familia', 'SKU', 'Forecast', 'Vendas', 'Assertividade', 'Period']
        results_table_display = results_table.rename(columns=display_rename_map)
        final_display_columns = [col for col in display_order if col in results_table_display.columns]
        results_table_display = results_table_display[final_display_columns]

        # Formatting (Ensure Period is string for display)
        if 'Period' in results_table_display.columns:
             # Check if it's PeriodDtype before converting
             if isinstance(results_table_display['Period'].dtype, pd.PeriodDtype):
                 results_table_display['Period'] = results_table_display['Period'].astype(str)

        for col in ['Vendas', 'Forecast']:
            if col in results_table_display.columns:
                 results_table_display[col] = pd.to_numeric(results_table_display[col], errors='coerce')
                 results_table_display[col] = results_table_display[col].map('{:,.2f}'.format, na_action='ignore')
        if 'Assertividade' in results_table_display.columns:
             results_table_display['Assertividade'] = pd.to_numeric(results_table_display['Assertividade'], errors='coerce')
             results_table_display['Assertividade'] = results_table_display['Assertividade'].map('{:.2%}'.format, na_action='ignore')

        st.dataframe(results_table_display, use_container_width=True)

        # Download Button
        @st.cache_data
        def convert_df_to_csv(df_to_convert):
            return df_to_convert.to_csv(index=False).encode('utf-8')
        csv = convert_df_to_csv(results_table_display)
        # --- Create Download Key based on context ---
        if analysis_context.get('type') == 'set':
            set_name = analysis_context.get('name', 'unknown_set')
            download_key = f'forecast_set_{set_name}_results.csv'
        else:
            # Ensure all keys exist for individual forecast filename
            method = analysis_context.get('method', 'unknown_method')
            abc = analysis_context.get('abc', 'unknown_abc')
            xyz = analysis_context.get('xyz', 'unknown_xyz')
            start = analysis_context.get('start', 'unknown_start')
            end = analysis_context.get('end', 'unknown_end')
            download_key = f'forecast_results_{method}_{abc}_{xyz}_{start}_to_{end}.csv'
        # --- End Create Download Key ---
        st.download_button(
           label="Download Tabela como CSV",
           data=csv,
           file_name=download_key,
           mime='text/csv',
        )

        # --- Visão Geral de Assertividade Section ---
        st.divider()
        st.markdown("### Visão Geral de Assertividade")

        # Ensure numeric assertividade exists
        if 'NumericAssertividade' not in results_table.columns:
             results_table['NumericAssertividade'] = pd.to_numeric(results_table['Assertividade'], errors='coerce').fillna(0)

        # --- Overall Top 5 SKUs (based on entire period's avg assertividade) ---
        overall_sku_assertividade = results_table.groupby(SKU_COL)['NumericAssertividade'].mean()
        top_5_sku_ids = overall_sku_assertividade.nlargest(5).index.tolist()
        top_5_data = results_table[results_table[SKU_COL].isin(top_5_sku_ids)].copy()

        if not top_5_data.empty:
            # Convert Period to Timestamp for plotting if needed
            if PERIOD_COL in top_5_data.columns:
                if isinstance(top_5_data[PERIOD_COL].dtype, pd.PeriodDtype):
                     top_5_data['Timestamp'] = top_5_data[PERIOD_COL].dt.to_timestamp()
                else: # Attempt conversion if not PeriodDtype
                    try:
                        top_5_data['Timestamp'] = pd.to_datetime(top_5_data[PERIOD_COL].astype(str))
                    except Exception as e:
                        print(f"Warning: Could not convert Period column to Timestamp for top 5 plot. {e}")
                        top_5_data = pd.DataFrame() # Empty df if conversion fails

            if not top_5_data.empty: # Check again after potential conversion failure
                 top_5_data['Assertividade_pct'] = top_5_data['NumericAssertividade'] * 100
                 top_5_data = top_5_data.sort_values('Timestamp')

        # Month Selector
        available_periods_in_results = sorted(results_table[PERIOD_COL].unique())
        month_options = ["Overall"] + [p.strftime('%Y-%m') for p in available_periods_in_results]
        selected_month_str = st.selectbox("Selecionar Mês para Análise:", month_options, index=0, key="month_analysis_selector")

        # Determine the DataFrame to use for calculations
        if selected_month_str == "Overall":
            calc_df = results_table.copy()
            # Determine analysis period string based on context type
            if analysis_context.get('type') == 'set':
                analysis_period_str = f"todos os períodos do conjunto '{analysis_context['name']}'"
            else:
                analysis_period_str = f"período {analysis_context['start']} a {analysis_context['end']}"
        else:
            selected_period_obj = pd.Period(selected_month_str, freq='M')
            assert st.session_state.results_table is not None, "Session state results_table became None unexpectedly."
            calc_df = st.session_state.results_table[st.session_state.results_table[PERIOD_COL] == selected_period_obj].copy()
            analysis_period_str = f"mês {selected_month_str}"

        if calc_df.empty:
            st.warning(f"Não há dados de resultados para {analysis_period_str}.")
        else:
            # Calculations
            try:
                # Ensure numeric columns exist/create them
                for col, base_col in [('NumericForecast', 'Valor Previsto'), ('NumericActual', 'Valor Real de Vendas'), ('NumericAssertividade', 'Assertividade')]:
                     if col not in calc_df.columns:
                         calc_df[col] = pd.to_numeric(calc_df[base_col], errors='coerce').fillna(0)

                # Weighted Average Assertividade
                if selected_month_str == "Overall":
                    # Average of monthly weighted averages
                    monthly_averages = []
                    all_periods_in_calc_df = calc_df[PERIOD_COL].unique()
                    for period in all_periods_in_calc_df:
                        month_df = calc_df[calc_df[PERIOD_COL] == period]
                        # Recalculate numeric cols just in case for month_df
                        if 'NumericForecast' not in month_df.columns: month_df['NumericForecast'] = pd.to_numeric(month_df['Valor Previsto'], errors='coerce').fillna(0)
                        if 'NumericAssertividade' not in month_df.columns: month_df['NumericAssertividade'] = pd.to_numeric(month_df['Assertividade'], errors='coerce').fillna(0)

                        monthly_weighted_sum = (month_df['NumericForecast'] * month_df['NumericAssertividade']).sum()
                        monthly_total_forecast = month_df['NumericForecast'].sum()
                        monthly_avg = monthly_weighted_sum / monthly_total_forecast if monthly_total_forecast > 0 else 0
                        monthly_averages.append(monthly_avg)
                    avg_assertividade = np.mean(monthly_averages) if monthly_averages else 0
                else:
                    # Specific month weighted average
                    weighted_sum = (calc_df['NumericForecast'] * calc_df['NumericAssertividade']).sum()
                    total_forecast_val = calc_df['NumericForecast'].sum() # Rename variable
                    avg_assertividade = weighted_sum / total_forecast_val if total_forecast_val > 0 else 0

                # Monthly/Overall Totals
                total_real_month = calc_df['NumericActual'].sum()
                total_forecast_month = calc_df['NumericForecast'].sum()
                difference_month = abs(total_real_month - total_forecast_month)

                # Use full results_table for overall totals
                if 'NumericActual' not in results_table.columns: results_table['NumericActual'] = pd.to_numeric(results_table['Valor Real de Vendas'], errors='coerce').fillna(0)
                if 'NumericForecast' not in results_table.columns: results_table['NumericForecast'] = pd.to_numeric(results_table['Valor Previsto'], errors='coerce').fillna(0)
                total_real_overall = results_table['NumericActual'].sum()
                total_forecast_overall = results_table['NumericForecast'].sum()
                difference_overall = abs(total_real_overall - total_forecast_overall)

            except Exception as e:
                st.error(f"Erro ao calcular a visão geral da assertividade para {analysis_period_str}: {e}")
                st.stop()

            # Layout and Visualization (Gauge, Bar, Metrics)
            col1, col2 = st.columns([2, 3])
            with col1:
                 gauge_title = f"Assertividade Média ({selected_month_str})" if selected_month_str != "Overall" else "Assertividade Média (Overall)"
                 # Add Set name to title if applicable
                 if analysis_context.get('type') == 'set' and selected_month_str == "Overall":
                      gauge_title = f"Assertividade Média (Conjunto: {analysis_context['name']})"
                 current_value = avg_assertividade * 100
                 gauge_fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = current_value,
                    number = {'suffix': "%", 'font': {'size': 50}},
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': gauge_title, 'font': {'size': 20}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': "rgba(0,0,0,0)"},
                        'bgcolor': "rgba(0,0,0,0)",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps' : [
                            {'range': [0, 60], 'color': '#dc3545'},
                            {'range': [60, 75], 'color': '#ffc107'},
                            {'range': [75, 90], 'color': '#17a2b8'},
                            {'range': [90, 100], 'color': '#28a745'}
                        ],
                        'threshold': {
                            'line': {'color': "white", 'width': 3},
                            'thickness': 0.9,
                            'value': current_value
                        }
                    }
                ))
                 gauge_fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
                 st.plotly_chart(gauge_fig, use_container_width=True)

            with col2:
                st.markdown(f"##### Top 5 Produtos - Assertividade ({selected_month_str})") # Title adapts
                if not top_5_data.empty and 'Timestamp' in top_5_data.columns: # Check Timestamp exists
                    # Filter top_5_data if specific month selected
                    plot_data = top_5_data
                    if selected_month_str != "Overall":
                         selected_ts = pd.Timestamp(selected_month_str)
                         plot_data = top_5_data[top_5_data['Timestamp'].dt.to_period('M') == selected_ts.to_period('M')]

                    # Plotting logic (using plot_data)
                    if not plot_data.empty:
                        # Use bar chart for single month, line for overall
                        if selected_month_str != "Overall":
                            bar_fig = px.bar(
                                plot_data.sort_values('Assertividade_pct', ascending=False), # Sort for bar chart
                                x=SKU_COL,
                                y='Assertividade_pct',
                                color=SKU_COL, # Color by SKU
                                title="",
                                labels={'Assertividade_pct': 'Assertividade (%)', SKU_COL: 'SKU'},
                                text_auto=True, # Use boolean for auto text
                            )
                            # Format the text displayed on bars
                            bar_fig.update_traces(texttemplate='%{y:.1f}', textposition='outside')
                            bar_fig.update_layout(yaxis=dict(range=[0, 105])) # Set y-axis range
                            fig = bar_fig
                        else: # Overall - use line chart
                             line_fig = px.line(
                                plot_data, # Already sorted by Timestamp
                                x='Timestamp',
                                y='Assertividade_pct',
                                color=SKU_COL,
                                title="",
                                labels={'Timestamp': 'Mês', 'Assertividade_pct': 'Assertividade (%)', SKU_COL: 'SKU'},
                                markers=True
                            )
                             line_fig.update_layout(yaxis=dict(range=[0, 105]))
                             line_fig.update_traces(line=dict(width=2))
                             fig = line_fig

                        # Common layout updates for both chart types
                        fig.update_layout(
                            height=250,
                            margin=dict(l=10, r=10, t=20, b=10),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            font=dict(color="white"),
                            legend_title_text='SKU',
                            showlegend=(selected_month_str == "Overall") # Show legend only for line chart
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                         st.info(f"Não há dados do top 5 para o mês {selected_month_str}.")

                else:
                    st.info(f"Não há dados de assertividade suficientes para mostrar o top 5.")


            # Summary Metrics
            st.divider()
            col_sum1, col_sum2, col_sum3 = st.columns(3)

            # --- Determine values and label based on selection and context --- (REVISED LOGIC)
            if selected_month_str == "Overall":
                # Use overall calculated values
                value_real = total_real_overall
                value_forecast = total_forecast_overall
                value_diff = difference_overall
                # Determine label based on context type
                if analysis_context.get('type') == 'set':
                    period_label = f"(Conjunto: {analysis_context.get('name', '?')})"
                else:
                    # Individual forecast - safely get start/end
                    start = analysis_context.get('start', '?')
                    end = analysis_context.get('end', '?')
                    period_label = f"({start} a {end})"
            else:
                # Use monthly calculated values
                value_real = total_real_month
                value_forecast = total_forecast_month
                value_diff = difference_month
                period_label = f"({selected_month_str})"
            # --- End Revised Logic ---

            with col_sum1:
                st.metric(label=f"Total Faturado {period_label}", value=f"{value_real:,.0f}")
            with col_sum2:
                st.metric(label=f"Total Previsto {period_label}", value=f"{value_forecast:,.0f}")
            with col_sum3:
                st.metric(label=f"Diferença {period_label}", value=f"{value_diff:,.0f}")

        # --- Guardar Forecast Button --- # Logic Simplified
        st.divider()
        # Button is never disabled based on selection anymore, only checks for exact duplicates
        save_disabled = False
        save_tooltip = "Guardar este forecast (incluindo seleção atual de ABC/XYZ e período) e limpar a área de resultados."

        # --- Disable Save Forecast if a Set is Loaded --- (NEW)
        if analysis_context.get('type') == 'set':
            save_disabled = True
            save_tooltip = "Não é possível guardar um forecast individual enquanto um conjunto está carregado."

        if st.button("💾 Guardar Forecast", disabled=save_disabled, help=save_tooltip):
            # Construct the key using the current analysis context (which holds the generated forecast's categories)
            combination_key = f"{analysis_context['abc']}-{analysis_context['xyz']}"

            # Check if this *exact* key already exists
            if combination_key not in st.session_state.saved_combinations:
                # Store the forecast data
                saved_data = {
                    'results': results_table.copy(),
                    'context': analysis_context.copy() # Includes potentially combined ABC/XYZ
                }
                # Add weights if applicable
                if analysis_context['method'] == "Previsão Ponderada por SKU":
                    saved_data['weights'] = st.session_state.ponderation_sku_weights.copy()

                # Save the data and add key to the set
                st.session_state.saved_forecasts[combination_key] = saved_data
                st.session_state.saved_combinations.add(combination_key)
                st.success(f"Forecast para '{combination_key}' guardado com sucesso!")

                # Clear current display and rerun
                st.session_state.results_table = None
                st.session_state.analysis_context = None
                st.rerun()
            else:
                # Inform user if the exact combination is already saved
                st.warning(f"Um forecast para a combinação exata '{combination_key}' já está guardado.")

# --- Run Directly Check (Optional) ---
# if __name__ == "__main__":
#     # Basic setup for direct execution (requires manual data loading/mocking)
#     if 'sales_data' not in st.session_state:
#         st.session_state.sales_data = pd.DataFrame() # Load dummy/actual
#     if 'abc_xyz_data' not in st.session_state:
#         st.session_state.abc_xyz_data = pd.DataFrame() # Load dummy/actual
#     render() 
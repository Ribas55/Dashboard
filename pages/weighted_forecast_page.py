# Placeholder for the weighted forecast dashboard page
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import sys
import os

# Ensure imports work when called from app.py
try:
    # Try relative import first
    from ..src.data_loader import load_sales_data, load_budget_data
    from ..src.analysis.custom_weighted_forecast import calculate_weighted_forecast
except (ImportError, ValueError):
    # Fallback for direct execution or different structure
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
    from data_loader import load_sales_data, load_budget_data
    from analysis.custom_weighted_forecast import calculate_weighted_forecast

# --- Page Configuration --- (Removed - Handled by app.py)
# st.set_page_config(layout="wide", page_title="Previsão Ponderada")

# --- Constants ---
# SALES_FILE_PATH = 'data/2022-2025.xlsx' # Path handled by app.py
BUDGET_FILE_PATH = 'data/Orçamento2022-2025.xlsx' # Adjust if necessary
FORECAST_PERIODS = 6 # Number of months to forecast

# --- Data Loading (Budget Only - Sales data comes from app.py session state) ---
# Moved inside render to avoid running st.cache_data on import
# @st.cache_data
# def load_budget():
#     print("Loading budget data (Weighted Forecast Page)...")
#     df_budget = load_budget_data(BUDGET_FILE_PATH)
#     return df_budget

def render():
    """Renders the Weighted Forecast page."""
    
    # Define data loading function here to ensure decorator runs after set_page_config
    @st.cache_data
    def load_budget():
        print("Loading budget data (Weighted Forecast Page)...")
        df_budget = load_budget_data(BUDGET_FILE_PATH)
        return df_budget
    
    st.title("📊 Previsão Ponderada de Vendas Mensais")

    # --- Get Sales Data from Session State --- 
    if 'sales_data' not in st.session_state or st.session_state.sales_data is None:
        st.error("Dados de vendas não encontrados. Por favor, carregue os dados na barra lateral primeiro.")
        st.stop()
    df_sales = st.session_state.sales_data

    # --- Define the 15 specific families ---
    specific_families = [
        "Cream Cracker", "Maria", "Wafer", "Sortido", 
        "Cobertas de Chocolate", "Água e Sal", "Digestiva",
        "Recheada", "Circus", "Tartelete", "Torrada",
        "Flocos de Neve", "Integral", "Mentol", "Aliança"
    ]

    # --- Filter Sales Data for Specific Families ---
    df_sales_filtered = df_sales[df_sales['family'].isin(specific_families)].copy()
    if df_sales_filtered.empty:
        st.warning(f"Nenhum dado encontrado para as famílias específicas: {', '.join(specific_families)}")
        st.stop()

    # --- Load Budget Data --- 
    df_budget = load_budget()
    if df_budget is None:
        st.warning("Não foi possível carregar os dados de orçamento. O componente 'Orçamento' será desativado.")
        budget_available = False
    else:
        budget_available = True

    # --- Filters & Weights (Moved to Main Area) ---
    st.markdown("### Filtros e Pesos da Previsão")
    filter_cols = st.columns(4) # Add columns for date filters
    
    # Get filter options
    managers = ["Todos"] + sorted(df_sales_filtered['commercial_manager'].unique().tolist()) # Use filtered data for managers
    families = ["Todas"] + sorted(specific_families) # Use the defined list for families

    # Prepare date options for dropdowns
    # Ensure invoice_date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_sales_filtered['invoice_date']):
        df_sales_filtered['invoice_date'] = pd.to_datetime(df_sales_filtered['invoice_date'], errors='coerce')

    # Get unique year-month combinations from sales data
    available_dates = df_sales_filtered['invoice_date'].dt.to_period('M').sort_values().unique()
    # Add some future dates (e.g., 12 months beyond last sales date)
    last_sales_period = available_dates.max() if available_dates.size > 0 else pd.Period(pd.Timestamp.now(), freq='M')
    future_periods = pd.period_range(start=last_sales_period + 1, periods=12, freq='M')
    all_periods = available_dates.tolist() + future_periods.tolist()

    # Format dates as "Month YYYY" (e.g., "Junho 2024")
    month_map_pt = {
        1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
        7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"
    }
    date_options = {f"{month_map_pt[p.month]} {p.year}": p.to_timestamp().normalize() for p in all_periods}
    date_options_list = list(date_options.keys())

    # Default start date: 6 months after the last sales date (or current if no data)
    default_start_period = last_sales_period + 1
    default_end_period = default_start_period + (FORECAST_PERIODS - 1)

    default_start_str = f"{month_map_pt[default_start_period.month]} {default_start_period.year}"
    default_end_str = f"{month_map_pt[default_end_period.month]} {default_end_period.year}"

    # Ensure defaults are in the options list
    if default_start_str not in date_options_list:
        default_start_str = date_options_list[-FORECAST_PERIODS] if len(date_options_list) >= FORECAST_PERIODS else date_options_list[0] if date_options_list else None
    if default_end_str not in date_options_list:
        default_end_str = date_options_list[-1] if date_options_list else None

    with filter_cols[0]:
        selected_manager = st.selectbox("Gestor Comercial:", managers, key="forecast_manager_select")
    with filter_cols[1]:
        selected_family = st.selectbox("Família:", families, key="forecast_family_select")
    with filter_cols[2]:
        selected_start_str = st.selectbox("De:", options=date_options_list, 
                                        index=date_options_list.index(default_start_str) if default_start_str and default_start_str in date_options_list else 0,
                                        key="forecast_start_date")
    with filter_cols[3]:
        selected_end_str = st.selectbox("Até:", options=date_options_list, 
                                      index=date_options_list.index(default_end_str) if default_end_str and default_end_str in date_options_list else len(date_options_list)-1,
                                      key="forecast_end_date")

    # Convert selected strings back to dates
    forecast_start_date_selected = date_options.get(selected_start_str)
    forecast_end_date_selected = date_options.get(selected_end_str)

    # Validate date range
    if forecast_start_date_selected and forecast_end_date_selected and forecast_start_date_selected > forecast_end_date_selected:
        st.error("A data 'De:' não pode ser posterior à data 'Até:'.")
        forecast_valid = False
    elif not forecast_start_date_selected or not forecast_end_date_selected:
        st.error("Selecione datas válidas para 'De:' e 'Até:'.")
        forecast_valid = False
    else:
        forecast_valid = True

    st.markdown("#### Pesos dos Componentes (%)")
    
    # Initialize weights in session state if they don't exist
    default_weights = {
        'avg_2023': 10,
        'avg_ytd': 10,
        'last_3_months': 25,
        'last_12_months': 25,
        'budget': 30,
        'cagr': 0 # Added CAGR component
    }
    weight_keys = list(default_weights.keys())

    # Use a unique key for forecast weights to avoid conflicts
    session_key_weights = 'forecast_page_weights'
    if session_key_weights not in st.session_state:
        st.session_state[session_key_weights] = default_weights.copy()
        if not budget_available:
            st.session_state[session_key_weights]['budget'] = 0

    # Weight Sliders in a form
    with st.form("forecast_weights_form"):
        # Create columns for weights for better layout
        weight_cols = st.columns(len(weight_keys)) # Adjust columns based on number of weights
        current_weights = {}
        
        for idx, key in enumerate(weight_keys):
            with weight_cols[idx]:
                # Adjust label for CAGR
                label = key.replace('_', ' ').title()
                if key == 'cagr':
                    label = "Cresc. (YTD vs 23)" 
                
                min_val, max_val, step = 0, 100, 1
                default_val = st.session_state[session_key_weights].get(key, 0)
    
                # Disable budget slider if data not available
                is_disabled = (key == 'budget' and not budget_available)
    
                current_weights[key] = st.number_input(
                    label=label, # Use shorter label
                    min_value=min_val,
                    max_value=max_val,
                    value=default_val,
                    step=step,
                    key=f"forecast_weight_{key}", # Unique key for each input
                    disabled=is_disabled,
                    help=f"Percentagem de peso para o componente '{label}'." + (" Desativado se dados de orçamento não disponíveis." if is_disabled else "")
                )

        # Submit button inside the form
        submitted = st.form_submit_button("Aplicar Pesos e Calcular Previsão")

        # Check and potentially normalize weights on submission
        if submitted:
            total_weight = sum(current_weights.values())
            if total_weight != 100:
                st.warning(f"A soma dos pesos ({total_weight}%) não é 100%. Ajuste os valores.")
                # Update session state, let user fix it
                st.session_state[session_key_weights] = current_weights
            else:
                st.session_state[session_key_weights] = current_weights
                st.success("Pesos válidos! Calculando previsão...")
                # No need to rerun, calculation happens below if weights are valid

    # Display current total weight outside the form
    total_weight_display = sum(st.session_state[session_key_weights].values())
    color = "green" if total_weight_display == 100 else "red"
    st.markdown(f"**Soma Atual dos Pesos:** <span style='color:{color};'>{total_weight_display}%</span>", unsafe_allow_html=True)
    if total_weight_display != 100:
        st.warning("A soma deve ser 100% para uma previsão válida.")

    st.divider() # Add a visual separator
    
    # --- Forecast Calculation ---
    weights_decimal = {k: v / 100.0 for k, v in st.session_state[session_key_weights].items()}

    # Use filters, converting "Todos"/"Todas" to None for the function
    manager_arg = selected_manager if selected_manager != "Todos" else None
    family_arg = selected_family if selected_family != "Todas" else None

    # Execute forecast calculation only if weights sum to 100 and dates are valid
    forecast_df = pd.DataFrame() # Initialize as empty
    if total_weight_display == 100 and forecast_valid:
        # Assert that dates are not None before calling the function
        assert forecast_start_date_selected is not None, "Start date should not be None here"
        assert forecast_end_date_selected is not None, "End date should not be None here"
        
        st.info(f"Gerando previsão de {selected_start_str} até {selected_end_str}...")
        try:
            forecast_df = calculate_weighted_forecast(
                df_sales=df_sales_filtered, # Use filtered sales data
                df_budget=df_budget if budget_available else None,
                weights=weights_decimal,
                forecast_start_date=forecast_start_date_selected,
                forecast_end_date=forecast_end_date_selected,
                manager_filter=manager_arg,
                family_filter=family_arg
            )
        except Exception as e:
            st.error(f"Ocorreu um erro durante o cálculo da previsão: {e}")
            st.stop()
    elif not forecast_valid:
        # Error message already shown above for invalid dates
        pass
    elif total_weight_display != 100:
        st.warning("Ajuste os pesos para que a soma seja 100% para gerar a previsão.")

    # --- Display Results ---
    st.markdown("---")
    st.subheader("Resultados da Previsão Ponderada")

    if forecast_df.empty:
        if total_weight_display == 100: # Only show this if weights are correct but no forecast
            st.warning("Nenhuma previsão gerada com os filtros e dados atuais. Verifique os dados de entrada ou os filtros aplicados.")
    else:
        # Prepare DataFrame for plotting
        forecast_df['date'] = pd.to_datetime(forecast_df[['year', 'month']].assign(day=1))
        forecast_df_display = forecast_df.copy() # Use the potentially aggregated df
        forecast_df_display['forecast_value'] = forecast_df_display['forecast_value'].round(2)
        forecast_df_display['date_str'] = forecast_df_display['date'].dt.strftime('%Y-%m')

        # Adjust display based on whether it's aggregated ('Todos') or specific manager
        is_aggregated_view = manager_arg is None

        # Create Title
        title_parts = ["Previsão Mensal de Vendas (Kg)"]
        if is_aggregated_view:
            title_parts.append("Total (Excl. Concurso/Outros)")
        elif manager_arg:
            title_parts.append(f"Gestor: {manager_arg}")
            
        if family_arg:
            title_parts.append(f"Família: {family_arg}")
        chart_title = " - ".join(title_parts)

        # Determine columns to display and labels
        if is_aggregated_view:
            plot_color_col = 'family'
            table_cols_to_show = ['date_str', 'family', 'forecast_value']
            table_col_rename = {'date_str': 'Mês', 'family': 'Família', 'forecast_value': 'Valor Previsto (Kg)'}
            labels = {'date': 'Mês', 'forecast_value': 'Valor Previsto (Kg)', 'family': 'Família'}
        else:
            # Use family if manager selected but family is 'Todas', else use manager
            plot_color_col = 'family' if not family_arg else 'commercial_manager' 
            table_cols_to_show = ['date_str', 'commercial_manager', 'family', 'forecast_value']
            table_col_rename = {'date_str': 'Mês', 'commercial_manager': 'Gestor Comercial', 'family': 'Família', 'forecast_value': 'Valor Previsto (Kg)'}
            labels = {'date': 'Mês', 'forecast_value': 'Valor Previsto (Kg)', 'family': 'Família', 'commercial_manager': 'Gestor Comercial'}
            # Add commercial_manager to display df if it's missing (shouldn't happen here but safe)
            if 'commercial_manager' not in forecast_df_display.columns:
                 forecast_df_display['commercial_manager'] = manager_arg # Add it back if needed

        # Plotting
        fig = px.line(
            forecast_df_display,
            x='date',
            y='forecast_value',
            color=plot_color_col,
            markers=True,
            labels=labels,
            title=chart_title
        )
        fig.update_layout(hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # Display Table
        st.dataframe(
            forecast_df_display[table_cols_to_show].rename(columns=table_col_rename),
            use_container_width=True
        )

        # Display weights used
        st.subheader("Pesos Atuais Aplicados")
        weights_display = {k.replace('_', ' ').title(): f"{v}%" for k, v in st.session_state[session_key_weights].items()}
        st.json(weights_display)

# This allows the script to be run directly for testing if needed
if __name__ == "__main__":
    # Minimal setup for direct run - assumes data files are accessible
    # In a real direct run, you might load dummy data or require manual loading
    st.session_state.sales_data = load_sales_data('data/2022-2025.xlsx', sheet_name="BD_Vendas")
    # Apply filtering here too for direct run consistency
    specific_families_main = [
        "Cream Cracker", "Maria", "Wafer", "Sortido", 
        "Cobertas de Chocolate", "Água e Sal", "Digestiva",
        "Recheada", "Circus", "Tartelete", "Torrada",
        "Flocos de Neve", "Integral", "Mentol", "Aliança"
    ]
    if st.session_state.sales_data is not None:
         st.session_state.sales_data = st.session_state.sales_data[st.session_state.sales_data['family'].isin(specific_families_main)]
    render() 
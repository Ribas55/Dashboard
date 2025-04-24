import streamlit as st
import pandas as pd
import numpy as np # Added for accuracy calculation
import datetime # Added for date input default
import plotly.graph_objects as go # Added for Gauge and Bar charts
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
# --- End Helper ---

def render(): # Wrap existing code in a render function
    # --- Page Configuration --- (Removed - handled by app.py)
    # st.set_page_config(page_title="Métodos de Forecasting", layout="wide")

    st.title("🔮 Métodos de Forecasting - Visualização") # Added icon to title

    # --- Initialize Session State for Results --- (Added)
    if 'results_table' not in st.session_state:
        st.session_state.results_table = None
    if 'analysis_context' not in st.session_state:
        st.session_state.analysis_context = None
    # --- End Initialization ---

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

    # --- Define Key Column Names --- # Added section for clarity
    # Adjust these if your column names are different in the main dataframe
    DATE_COL_RAW = 'invoice_date' # Original date column before converting to period (Corrected from 'Data')
    # Corrected based on actual DataFrame columns
    SKU_COL = 'sku'
    VALUE_COL = 'sales_value'
    FAMILY_COL = 'family'
    MANAGER_COL = 'commercial_manager'
    PERIOD_COL = DEFAULT_DATE_COL # Defined in forecasting.py, 'Period' - Assumed correct as it's often generated

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

    col1, col2 = st.columns(2)

    with col1:
        # 1. Select Forecasting Method (Updated)
        method_options = [
            "Selecione...",
            "ARIMA (0,1,1)", "ARIMA (1,1,1)", "ARIMA (2,1,2)",
            "SES (alpha=0.1)", "SES (alpha=0.3)",
            "SMA (N=1)", "SMA (N=2)", "SMA (N=3)", "SMA (N=6)", "SMA (N=9)"
        ]
        selected_method = st.selectbox("Selecionar Método de Forecasting:", method_options)

        # 2. Time Horizon (New)
        st.markdown("Horizonte de Previsão:")
        if all_periods_str:
            # Use session state to preserve selection across reruns if context exists
            start_idx = 0
            end_idx = len(all_periods_str) - 1
            if st.session_state.analysis_context:
                 try:
                    start_idx = all_periods_str.index(st.session_state.analysis_context['start'])
                    end_idx = all_periods_str.index(st.session_state.analysis_context['end'])
                 except ValueError:
                      pass # Keep default if saved context not in current options

            start_period_str = st.selectbox(
                "De:",
                options=all_periods_str,
                index=start_idx, # Default to earliest or saved context
                key='start_period'
            )
            end_period_str = st.selectbox(
                "Até:",
                options=all_periods_str,
                index=end_idx, # Default to latest or saved context
                key='end_period'
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
        # 3. Select Combination Type (Moved and Adjusted for ABC/XYZ focus initially)
        st.markdown("Filtro de SKUs por Categoria ABC/XYZ (Baseado em 2024):")

        abc_col = 'abc_class'
        xyz_col = 'xyz_class'

        if abc_col not in abc_xyz_df.columns or xyz_col not in abc_xyz_df.columns:
            st.error(f"Colunas '{abc_col}' ou '{xyz_col}' não encontradas nos resultados ABC/XYZ.")
            st.stop()

        abc_categories = ["Todas"] + sorted(abc_xyz_df[abc_col].unique())
        xyz_categories = ["Todas"] + sorted(abc_xyz_df[xyz_col].unique())

        sub_col1, sub_col2 = st.columns(2)
        with sub_col1:
            selected_abc = st.selectbox("Categoria ABC:", abc_categories, help="Selecione a categoria ABC.")
        with sub_col2:
            selected_xyz = st.selectbox("Categoria XYZ:", xyz_categories, help="Selecione a categoria XYZ.")


    # --- Apply Button ---
    apply_forecast = st.button("Aplicar Método e Gerar Previsão")

    # --- Generation Logic (Inside Button Click) ---
    if apply_forecast:
        if selected_method == "Selecione...":
            st.warning("Por favor, selecione um método de forecasting.")
            # Clear previous results if selection is invalid
            st.session_state.results_table = None
            st.session_state.analysis_context = None
        elif not valid_periods or not start_period or not end_period:
            st.warning("Por favor, selecione um horizonte de previsão válido.")
            # Clear previous results if selection is invalid
            st.session_state.results_table = None
            st.session_state.analysis_context = None
        else:
            # --- Filtering Logic --- (Same as before)
            target_skus_df = abc_xyz_df.copy()
            if selected_abc != "Todas":
                target_skus_df = target_skus_df[target_skus_df[abc_col] == selected_abc]
            if selected_xyz != "Todas":
                 target_skus_df = target_skus_df[target_skus_df[xyz_col] == selected_xyz]
            if 'sku' not in target_skus_df.columns:
                st.error("Coluna 'sku' não encontrada nos resultados da classificação ABC/XYZ.")
                st.stop()
            filtered_sku_list = target_skus_df['sku'].tolist()
            if not filtered_sku_list:
                 st.warning(f"Nenhum SKU encontrado para a combinação de filtros ABC='{selected_abc}' e XYZ='{selected_xyz}'.")
                 # Clear previous results
                 st.session_state.results_table = None
                 st.session_state.analysis_context = None
                 st.stop()
            try:
                 if df[SKU_COL].dtype != target_skus_df['sku'].dtype:
                      common_type = str
                      filtered_sku_list_typed = [common_type(sku) for sku in filtered_sku_list]
                      df[SKU_COL] = df[SKU_COL].astype(common_type)
                 else:
                      filtered_sku_list_typed = filtered_sku_list
            except Exception as e:
                 st.error(f"Erro ao alinhar tipos de SKU para filtragem: {e}")
                 st.stop()
            df_for_forecasting = df[df[SKU_COL].isin(filtered_sku_list_typed)].copy()
            if df_for_forecasting.empty:
                st.warning("DataFrame vazio após filtrar por SKUs selecionados. Nenhuma previsão será gerada.")
                # Clear previous results
                st.session_state.results_table = None
                st.session_state.analysis_context = None
                st.stop()

            # --- Generate Forecasts ---
            with st.spinner(f"Gerando previsões com {selected_method} para {len(filtered_sku_list_typed)} SKUs..."):
                try:
                    forecast_results_df = generate_forecasts(
                        filtered_df=df_for_forecasting,
                        selected_method_str=selected_method,
                        start_period=start_period,
                        end_period=end_period,
                        sku_col=SKU_COL,
                        date_col=PERIOD_COL,
                        value_col=VALUE_COL
                    )
                except Exception as e:
                    st.error(f"Ocorreu um erro durante a geração das previsões: {e}")
                    # Clear previous results on error
                    st.session_state.results_table = None
                    st.session_state.analysis_context = None
                    st.stop()

            if forecast_results_df.empty:
                st.warning("Nenhuma previsão foi gerada com sucesso. Verifique os dados ou o método selecionado.")
                # Clear previous results
                st.session_state.results_table = None
                st.session_state.analysis_context = None
                st.stop()

            # --- Prepare Actual Sales and Details --- (Same as before)
            try:
                actuals_filtered = df_for_forecasting[
                    (df_for_forecasting[PERIOD_COL] >= start_period) &
                    (df_for_forecasting[PERIOD_COL] <= end_period)
                ]
                actuals_df = actuals_filtered.groupby([SKU_COL, PERIOD_COL], observed=False)[VALUE_COL].sum().reset_index()
                actuals_df.rename(columns={VALUE_COL: 'Valor Real de Vendas'}, inplace=True)
                sku_details = df_for_forecasting[[SKU_COL, FAMILY_COL, MANAGER_COL]].drop_duplicates(subset=[SKU_COL])
            except Exception as e:
                st.error(f"Erro ao preparar dados reais ou detalhes do SKU: {e}")
                st.stop()

            # --- Merge Forecasts, Actuals, Details --- (Same as before)
            try:
                results_table = pd.merge(forecast_results_df, actuals_df, on=[SKU_COL, PERIOD_COL], how='left')
                results_table = pd.merge(results_table, sku_details, on=SKU_COL, how='left')
            except Exception as e:
                st.error(f"Erro ao combinar resultados das previsões com dados reais: {e}")
                st.stop()

            # --- Calculate Accuracy (Assertividade) --- (Same as before)
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

            # --- Store results in session state --- (Added)
            st.session_state.results_table = results_table
            st.session_state.analysis_context = {
                'method': selected_method,
                'start': start_period_str,
                'end': end_period_str
            }
            st.success("Previsão gerada com sucesso!") # Give user feedback


    # --- Display Area (Conditional on Results Existing in Session State) --- (Moved and Modified)
    st.divider()

    if st.session_state.get('results_table') is not None:
        # Retrieve results from session state
        results_table = st.session_state.results_table
        analysis_context = st.session_state.analysis_context
        # Add assertions for results_table and analysis_context
        assert results_table is not None, "results_table should not be None here."
        assert analysis_context is not None, "analysis_context should not be None here."

        # Display Subheader using stored context
        st.subheader(f"Resultados do Forecasting ({analysis_context['method']}) para o período {analysis_context['start']} a {analysis_context['end']}")

        # --- Display Final Table ---
        st.markdown("### Tabela de Resultados")
        # (Formatting logic remains the same, operates on retrieved results_table)
        display_rename_map = {
            FAMILY_COL: 'Familia', SKU_COL: 'SKU', 'Valor Previsto': 'Forecast',
            'Valor Real de Vendas': 'Vendas', PERIOD_COL: 'Period', 'Assertividade': 'Assertividade'
        }
        display_order = ['Familia', 'SKU', 'Forecast', 'Vendas', 'Assertividade', 'Period']
        results_table_display = results_table.rename(columns=display_rename_map)
        final_display_columns = [col for col in display_order if col in results_table_display.columns]
        results_table_display = results_table_display[final_display_columns]
        if 'Period' in results_table_display.columns:
             results_table_display['Period'] = results_table_display['Period'].astype(str)
        for col in ['Vendas', 'Forecast']:
                if col in results_table_display.columns:
                 results_table_display[col] = pd.to_numeric(results_table_display[col], errors='coerce')
                 results_table_display[col] = results_table_display[col].map('{:,.2f}'.format, na_action='ignore')
        if 'Assertividade' in results_table_display.columns:
             results_table_display['Assertividade'] = pd.to_numeric(results_table_display['Assertividade'], errors='coerce')
             results_table_display['Assertividade'] = results_table_display['Assertividade'].map('{:.2%}'.format, na_action='ignore')
        st.dataframe(results_table_display, use_container_width=True)

        # Download Button (uses formatted display table)
        @st.cache_data
        def convert_df_to_csv(df_to_convert):
            return df_to_convert.to_csv(index=False).encode('utf-8')
        csv = convert_df_to_csv(results_table_display)
        st.download_button(
           label="Download Tabela como CSV",
           data=csv,
       file_name=f'forecast_results_{analysis_context["method"]}_{analysis_context["start"]}_to_{analysis_context["end"]}.csv',
           mime='text/csv',
        )

        # --- Visão Geral de Assertividade Section ---
        # (Logic remains the same, but now runs whenever results_table is in session state)
        st.divider()
        st.markdown("### Visão Geral de Assertividade")

        # Month Selector
        available_periods_in_results = sorted(results_table[PERIOD_COL].unique())
        month_options = ["Overall"] + [p.strftime('%Y-%m') for p in available_periods_in_results]
        # Use a different key for the selectbox if needed to avoid conflicts, or ensure it's unique
        selected_month_str = st.selectbox("Selecionar Mês para Análise:", month_options, index=0, key="month_analysis_selector")

        # Determine the DataFrame to use for calculations
        if selected_month_str == "Overall":
            calc_df = results_table.copy()
            analysis_period_str = f"período {analysis_context['start']} a {analysis_context['end']}"
        else:
            selected_period_obj = pd.Period(selected_month_str, freq='M')
            # IMPORTANT: Filter the original results_table from session state
            # Add assertion before filtering
            assert st.session_state.results_table is not None, "Session state results_table became None unexpectedly."
            calc_df = st.session_state.results_table[st.session_state.results_table[PERIOD_COL] == selected_period_obj].copy()
            analysis_period_str = f"mês {selected_month_str}"

        if calc_df.empty:
            st.warning(f"Não há dados de resultados para {analysis_period_str}.")
        else:
            # Calculations (Ensure column names match those in the stored results_table)
            try:
                calc_df['NumericForecast'] = pd.to_numeric(calc_df['Valor Previsto'], errors='coerce').fillna(0)
                calc_df['NumericActual'] = pd.to_numeric(calc_df['Valor Real de Vendas'], errors='coerce').fillna(0)
                calc_df['NumericAssertividade'] = pd.to_numeric(calc_df['Assertividade'], errors='coerce').fillna(0)

                # 1. Weighted Average Assertividade
                # --- Modified Calculation for Overall --- #
                if selected_month_str == "Overall":
                    monthly_averages = []
                    all_periods_in_calc_df = calc_df[PERIOD_COL].unique()
                    for period in all_periods_in_calc_df:
                        month_df = calc_df[calc_df[PERIOD_COL] == period]
                        monthly_weighted_sum = (month_df['NumericForecast'] * month_df['NumericAssertividade']).sum()
                        monthly_total_forecast = month_df['NumericForecast'].sum()
                        monthly_avg = monthly_weighted_sum / monthly_total_forecast if monthly_total_forecast > 0 else 0
                        monthly_averages.append(monthly_avg)
                    # Calculate the simple average of the monthly averages
                    avg_assertividade = np.mean(monthly_averages) if monthly_averages else 0
                else:
                    # Keep original calculation for specific month
                    weighted_sum = (calc_df['NumericForecast'] * calc_df['NumericAssertividade']).sum()
                    total_forecast = calc_df['NumericForecast'].sum()
                    avg_assertividade = weighted_sum / total_forecast if total_forecast > 0 else 0
                # --- End Modified Calculation --- #

                # 2. Top 5 SKUs by Average Assertividade (for the selected period)
                # Note: If monthly, this shows top 5 FOR THAT MONTH. If Overall, shows top 5 based on mean over whole period.
                top_5_skus = calc_df.groupby(SKU_COL)['NumericAssertividade'].mean().nlargest(5).reset_index()

                # 3. Totals (for the selected period)
                total_real = calc_df['NumericActual'].sum()
                total_forecast = calc_df['NumericForecast'].sum() # Need total forecast for metrics
                difference = abs(total_real - total_forecast)

            except Exception as e:
                st.error(f"Erro ao calcular a visão geral da assertividade para {analysis_period_str}: {e}")
                st.stop() # Stop if calculation fails

            # Layout and Visualization (Gauge, Bar, Metrics - same as before, uses calculated values)
            col1, col2 = st.columns([2, 3])
            with col1:
                 gauge_title = f"Assertividade Média ({selected_month_str})" if selected_month_str != "Overall" else "Assertividade Média (Overall)"
                 gauge_fig = go.Figure(go.Indicator(
                    mode = "gauge+number", value = avg_assertividade * 100,
                    number = {'suffix': "%", 'font': {'size': 50}}, domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': gauge_title, 'font': {'size': 20}},
                    gauge = {
                        'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                        'bar': {'color': get_gauge_color(avg_assertividade), 'thickness': 0.4},
                        'bgcolor': "rgba(0,0,0,0)", 'borderwidth': 2, 'bordercolor': "gray",
                        'steps' : [ {'range': [0, 60], 'color': '#dc3545'}, {'range': [60, 75], 'color': '#ffc107'},
                                    {'range': [75, 90], 'color': '#17a2b8'}, {'range': [90, 100], 'color': '#28a745'}]
                        }))
                 gauge_fig.update_layout(height=250, margin=dict(l=10, r=10, t=50, b=10), paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
                 st.plotly_chart(gauge_fig, use_container_width=True)

            with col2:
                bar_title = f"Top 5 Produtos ({selected_month_str})" if selected_month_str != "Overall" else "Top 5 Produtos (Overall Média)"
                st.markdown(f"##### {bar_title}")
                if not top_5_skus.empty:
                    top_5_skus['Assertividade_pct'] = top_5_skus['NumericAssertividade'] * 100
                    top_5_skus['Color'] = top_5_skus['NumericAssertividade'].apply(get_gauge_color)
                    top_5_skus = top_5_skus.sort_values('Assertividade_pct', ascending=True)
                    bar_fig = go.Figure(go.Bar(
                        y=top_5_skus[SKU_COL].astype(str), x=top_5_skus['Assertividade_pct'],
                        text=top_5_skus['Assertividade_pct'].map('{:.2f}%'.format), textposition='outside',
                        orientation='h', marker_color=top_5_skus['Color']))
                    bar_fig.update_layout(height=250, margin=dict(l=10, r=10, t=0, b=10), xaxis_title="Assertividade Média (%)", yaxis_title="SKU",
                                          yaxis={'categoryorder':'total ascending'}, xaxis=dict(range=[0, 105]),
                                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
                    bar_fig.update_traces(textfont_color='white', textfont_size=12)
                    st.plotly_chart(bar_fig, use_container_width=True)
                else:
                    st.info(f"Não há dados de assertividade suficientes para mostrar o top 5 para {analysis_period_str}.")

            # Summary Metrics
            st.divider()
            col_sum1, col_sum2, col_sum3 = st.columns(3)
            metric_suffix = f" ({selected_month_str})" if selected_month_str != "Overall" else " (Overall)"
            with col_sum1:
                st.metric(label=f"Total Faturado{metric_suffix}", value=f"{total_real:,.0f}")
            with col_sum2:
                st.metric(label=f"Total Previsto{metric_suffix}", value=f"{total_forecast:,.0f}")
            with col_sum3:
                st.metric(label=f"Diferença{metric_suffix}", value=f"{difference:,.0f}")


    # --- Add some explanation or guidance ---
    st.sidebar.title("Ajuda")
    st.sidebar.info(
        """
        Nesta página, pode aplicar diferentes métodos de forecasting aos dados de vendas.

        1.  **Selecione o Método:** Escolha o algoritmo (e seus parâmetros) que deseja usar.
        2.  **Selecione o Horizonte:** Defina o período (mês inicial e final) para o qual deseja gerar previsões.
        3.  **Filtre por Categoria:** Selecione as categorias ABC e XYZ para focar em SKUs específicos (baseado na classificação de 2024).
        4.  **Aplique:** Clique no botão para executar o forecasting nos SKUs e período selecionados.
        5.  **Resultados:** Uma tabela e uma visão geral da assertividade serão exibidas. Use o seletor de mês na visão geral para analisar meses específicos.
        """ # Updated help text slightly
    )

# Allow running the page directly for testing (optional)
# if __name__ == "__main__":
#     # Basic setup for direct execution
#     # You'd need to load data into session_state manually or use dummy data
#     if 'sales_data' not in st.session_state:
#          st.session_state.sales_data = pd.DataFrame() # Load dummy or actual data
#     if 'abc_xyz_data' not in st.session_state:
#          st.session_state.abc_xyz_data = pd.DataFrame() # Load dummy or actual data
#     render() 
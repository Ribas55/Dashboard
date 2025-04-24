"""
Time series analysis page with cascading filters.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.visualizations import create_aggregated_time_series
from typing import Optional, Dict, Union

def render():
    """Render the time series page."""
    st.subheader("Séries Temporais - Visualização")
    
    if not st.session_state.data_loaded:
        st.info("Por favor, carregue os dados primeiro.")
        return
    
    # Remover o dicionário anterior para inicializar um novo
    if 'time_series_filters' in st.session_state:
        del st.session_state.time_series_filters
        
    # Initialize session state for filters if needed
    if 'time_series_filters' not in st.session_state:
        st.session_state.time_series_filters = {
            'commercial_manager': None,
            'family': None,
            'subfamily': None,
            'format': None,
            'weight': None,
            'sku': None
        }
    
    # Ensure these columns exist in the dataframe
    required_cols = ['family', 'subfamily', 'format', 'weight', 'sku']
    for col in required_cols:
        if col not in st.session_state.sales_data.columns:
            st.error(f"Coluna '{col}' não encontrada nos dados. Por favor, verifique o formato dos dados.")
            return
    
    # Create filters section
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    # Get the unique values for each filter
    df = st.session_state.sales_data
    
    # Lista das 15 famílias específicas
    specific_families = [
        "Cream Cracker", "Maria", "Wafer", "Sortido", 
        "Cobertas de Chocolate", "Água e Sal", "Digestiva",
        "Recheada", "Circus", "Tartelete", "Torrada",
        "Flocos de Neve", "Integral", "Mentol", "Aliança"
    ]
    
    # Gestor Comercial filter
    with col1:
        # Check if commercial_manager column exists
        if 'commercial_manager' in df.columns:
            managers = sorted(df['commercial_manager'].dropna().unique().tolist())
            manager_options = ['Todos'] + managers
            selected_manager = st.selectbox(
                "Gestor Comercial",
                options=manager_options,
                index=0,
                key="manager_selector"
            )
            
            # Update session state
            if selected_manager == 'Todos':
                manager_filter = None
            else:
                manager_filter = selected_manager
        else:
            manager_filter = None
            st.warning("Campo 'Gestor Comercial' não encontrado nos dados.")
    
    # Family filter
    with col2:
        # Apenas as 15 famílias específicas ao invés de todas as famílias
        families = sorted(specific_families)
        
        # If manager filter is applied, filter the families
        if manager_filter and 'commercial_manager' in df.columns:
            family_df = df[df['commercial_manager'] == manager_filter]
            # Only show families that belong to this manager and are in the specific families list
            available_families = sorted(list(set(family_df['family'].unique().tolist()) & set(specific_families)))
            if not available_families:
                available_families = families  # Fallback to all families if none match
        else:
            available_families = families
        
        family_options = ['Todas'] + available_families
        selected_family = st.selectbox(
            "Família",
            options=family_options,
            index=0,
            key="family_selector"
        )
        
        # Update session state
        if selected_family == 'Todas':
            family_filter = None
        else:
            family_filter = selected_family
    
    # Sub Family filter (depends on family)
    with col3:
        filtered_df = df
        
        # Apply manager filter if selected
        if manager_filter and 'commercial_manager' in df.columns:
            filtered_df = filtered_df[filtered_df['commercial_manager'] == manager_filter]
            
        # Apply family filter if selected
        if family_filter:
            # Filter subfamilies based on selected family
            filtered_df = filtered_df[filtered_df['family'] == family_filter]
        else:
            # Apenas subfamílias das 15 famílias específicas ao invés de todas as subfamílias
            filtered_df = filtered_df[filtered_df['family'].isin(specific_families)]
        
        subfamilies = filtered_df['subfamily'].unique().tolist()
        subfamilies.sort()
        subfamily_options = ['Todas'] + subfamilies
        
        selected_subfamily = st.selectbox(
            "Sub Família",
            options=subfamily_options,
            index=0,
            key="subfamily_selector"
        )
        
        # Update filter value
        if selected_subfamily == 'Todas':
            subfamily_filter = None
        else:
            subfamily_filter = selected_subfamily
    
    # Format filter (depends on subfamily)
    with col4:
        filtered_df = df
        
        # Apply manager filter if selected
        if manager_filter and 'commercial_manager' in df.columns:
            filtered_df = filtered_df[filtered_df['commercial_manager'] == manager_filter]
            
        # Apply family filter if selected
        if family_filter:
            filtered_df = filtered_df[filtered_df['family'] == family_filter]
        else:
            # Se nenhuma família selecionada, filtrar apenas as 15 famílias específicas
            filtered_df = filtered_df[filtered_df['family'].isin(specific_families)]
        
        # Apply subfamily filter if selected
        if subfamily_filter:
            filtered_df = filtered_df[filtered_df['subfamily'] == subfamily_filter]
        
        # Get formats from filtered data
        formats = filtered_df['format'].unique().tolist()
        formats.sort()
        
        format_options = ['Todos'] + formats
        selected_format = st.selectbox(
            "Formato",
            options=format_options,
            index=0,
            key="format_selector"
        )
        
        # Update filter value
        if selected_format == 'Todos':
            format_filter = None
        else:
            format_filter = selected_format
    
    # Weight filter (depends on format)
    with col5:
        filtered_df = df
        
        # Apply manager filter if selected
        if manager_filter and 'commercial_manager' in df.columns:
            filtered_df = filtered_df[filtered_df['commercial_manager'] == manager_filter]
            
        # Apply previous filters
        if family_filter:
            filtered_df = filtered_df[filtered_df['family'] == family_filter]
        else:
            # Se nenhuma família selecionada, filtrar apenas as 15 famílias específicas
            filtered_df = filtered_df[filtered_df['family'].isin(specific_families)]
        
        if subfamily_filter:
            filtered_df = filtered_df[filtered_df['subfamily'] == subfamily_filter]
        
        if format_filter:
            filtered_df = filtered_df[filtered_df['format'] == format_filter]
        
        # Get weights from filtered data
        weights = filtered_df['weight'].unique().tolist()
        weights.sort()
        
        weight_options = ['Todas'] + weights
        selected_weight = st.selectbox(
            "Gramagem",
            options=weight_options,
            index=0,
            key="weight_selector"
        )
        
        # Update filter value
        if selected_weight == 'Todas':
            weight_filter = None
        else:
            weight_filter = selected_weight
    
    # SKU filter (depends on all previous filters)
    with col6:
        filtered_df = df
        
        # Apply manager filter if selected
        if manager_filter and 'commercial_manager' in df.columns:
            filtered_df = filtered_df[filtered_df['commercial_manager'] == manager_filter]
            
        # Apply all previous filters
        if family_filter:
            filtered_df = filtered_df[filtered_df['family'] == family_filter]
        else:
            # Se nenhuma família selecionada, filtrar apenas as 15 famílias específicas
            filtered_df = filtered_df[filtered_df['family'].isin(specific_families)]
        
        if subfamily_filter:
            filtered_df = filtered_df[filtered_df['subfamily'] == subfamily_filter]
        
        if format_filter:
            filtered_df = filtered_df[filtered_df['format'] == format_filter]
        
        if weight_filter:
            filtered_df = filtered_df[filtered_df['weight'] == weight_filter]
        
        # Get SKUs from filtered data
        skus = filtered_df['sku'].unique().tolist()
        skus.sort()
        
        sku_options = ['All SKUs'] + skus
        selected_sku = st.selectbox(
            "SKU",
            options=sku_options,
            index=0,
            key="sku_selector"
        )
        
        # Update filter value
        if selected_sku == 'All SKUs':
            sku_filter = None
        else:
            sku_filter = selected_sku
    
    # Atualizar st.session_state.time_series_filters com os novos valores
    st.session_state.time_series_filters = {
        'commercial_manager': manager_filter,
        'family': family_filter,
        'subfamily': subfamily_filter,
        'format': format_filter,
        'weight': weight_filter,
        'sku': sku_filter
    }
    
    # Additional options
    col1, col2, col3 = st.columns(3)

    with col1:
        # Normalize checkbox
        normalize = st.checkbox("Normalizar valores (0-1)", value=False, key="normalize")

    with col2:
        # Time aggregation
        time_agg_options = {
            "Mensal": "M",
            "Trimestral": "Q",
            "Anual": "Y"
        }
        time_agg = st.selectbox(
            "Agregação temporal", 
            options=list(time_agg_options.keys()),
            index=0,
            key="time_agg"
        )
        time_agg_code = time_agg_options[time_agg]

    with col3:
        # Opção para mostrar apenas a série total
        if 'show_only_total' not in st.session_state:
            st.session_state.show_only_total = False
        
        show_only_total = st.checkbox(
            "Mostrar apenas o total", 
            value=st.session_state.show_only_total,
            key="show_only_total"
        )
    
    # Filter data based on selections
    filtered_data = df.copy()
    
    # Filtrar para incluir apenas as 15 famílias específicas
    filtered_data = filtered_data[filtered_data['family'].isin(specific_families)]

    # Aplicar os filtros selecionados pelo usuário
    for filter_key, filter_value in st.session_state.time_series_filters.items():
        if filter_value is not None and filter_key in filtered_data.columns:
            filtered_data = filtered_data[filtered_data[filter_key] == filter_value]
    
    # Create title for the chart based on selections
    if st.session_state.time_series_filters['sku']:
        chart_title = f"Séries Temporais - {st.session_state.time_series_filters['sku']}"
    elif st.session_state.time_series_filters['subfamily']:
        chart_title = f"Séries Temporais - Subfamília: {st.session_state.time_series_filters['subfamily']}"
    elif st.session_state.time_series_filters['family']:
        chart_title = f"Séries Temporais - Família: {st.session_state.time_series_filters['family']}"
    elif st.session_state.time_series_filters['commercial_manager'] and 'commercial_manager' in filtered_data.columns:
        chart_title = f"Séries Temporais - Gestor: {st.session_state.time_series_filters['commercial_manager']}"
    else:
        chart_title = "Séries Temporais por Família"
    
    # Create visualization
    if not filtered_data.empty:
        # Determine what grouping to use
        if st.session_state.time_series_filters['sku']:
            # Single SKU selected
            fig = create_aggregated_time_series(
                filtered_data,
                group_by='sku', 
                group_value=st.session_state.time_series_filters['sku'],
                period="M",
                reference_date=pd.Timestamp('2025-02-01'),  # Set February 2025 as reference
                normalize=normalize,
                show_only_total=show_only_total
            )
        elif st.session_state.time_series_filters['family'] and not st.session_state.time_series_filters['subfamily']:
            # Only family selected, group by subfamily
            fig = create_aggregated_time_series(
                filtered_data,
                group_by='subfamily',
                period="M",
                reference_date=pd.Timestamp('2025-02-01'),  # Set February 2025 as reference
                normalize=normalize,
                show_only_total=show_only_total
            )
        elif st.session_state.time_series_filters['subfamily']:
            # Subfamily selected, group by SKU
            fig = create_aggregated_time_series(
                filtered_data,
                group_by='sku',
                period="M",
                reference_date=pd.Timestamp('2025-02-01'),  # Set February 2025 as reference
                normalize=normalize,
                show_only_total=show_only_total
            )
        else:
            # No specific filters, group by family
            fig = create_aggregated_time_series(
                filtered_data,
                group_by='family',
                period="M",
                reference_date=pd.Timestamp('2025-02-01'),  # Set February 2025 as reference
                normalize=normalize,
                show_only_total=show_only_total
            )
            
        # Update chart with dark theme styling
        fig.update_layout(
            template="plotly_dark",
            plot_bgcolor='rgba(30, 33, 48, 1)',
            paper_bgcolor='rgba(30, 33, 48, 1)',
            font=dict(color="white"),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(70, 70, 70, 0.3)',
            ),
            yaxis=dict(
                showgrid=True,
                gridcolor='rgba(70, 70, 70, 0.3)',
            ),
            height=500,
            margin=dict(l=40, r=40, t=40, b=40),
            title=chart_title
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Nenhum dado encontrado com os filtros selecionados.")
    
    # Statistics section with 4 cards (Media, Maximo, Minimo, Tendencia)
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    if not filtered_data.empty:
        # Ensure invoice_date is datetime for calculations
        filtered_data['invoice_date'] = pd.to_datetime(filtered_data['invoice_date'])
        filtered_data = filtered_data.sort_values('invoice_date')
        
        # Get time range from the plot (if not explicitly set, use the entire date range)
        min_date = filtered_data['invoice_date'].min()
        max_date = filtered_data['invoice_date'].max()
        
        # Calculate monthly data for the range
        filtered_data['month'] = filtered_data['invoice_date'].dt.to_period('M')
        monthly_data = filtered_data.groupby('month')['sales_value'].sum().reset_index()
        
        # Calculate key metrics
        mean_monthly = monthly_data['sales_value'].mean()
        
        # For maximum/minimum, find which family/subfamily/SKU it belongs to
        # Max calculation
        if st.session_state.time_series_filters['sku']:
            # Single SKU selected - find the max month
            max_month = monthly_data[monthly_data['sales_value'] == monthly_data['sales_value'].max()]['month'].iloc[0]
            max_value = monthly_data['sales_value'].max()
            max_entity = st.session_state.time_series_filters['sku']
            max_entity_type = "SKU"
            max_date_str = max_month.strftime('%b-%y')
        elif st.session_state.time_series_filters['subfamily']:
            # Subfamily selected - find max SKU
            monthly_sku_data = filtered_data.groupby(['month', 'sku'])['sales_value'].sum().reset_index()
            monthly_sku_data = monthly_sku_data.sort_values('sales_value', ascending=False)
            max_row = monthly_sku_data.iloc[0]
            max_value = max_row['sales_value']
            max_entity = max_row['sku']
            max_entity_type = "SKU"
            max_date_str = max_row['month'].strftime('%b-%y')
        elif st.session_state.time_series_filters['family']:
            # Family selected - find max subfamily
            monthly_subfamily_data = filtered_data.groupby(['month', 'subfamily'])['sales_value'].sum().reset_index()
            monthly_subfamily_data = monthly_subfamily_data.sort_values('sales_value', ascending=False)
            max_row = monthly_subfamily_data.iloc[0]
            max_value = max_row['sales_value']
            max_entity = max_row['subfamily']
            max_entity_type = "Subfamília"
            max_date_str = max_row['month'].strftime('%b-%y')
        else:
            # Nothing selected - find max family
            monthly_family_data = filtered_data.groupby(['month', 'family'])['sales_value'].sum().reset_index()
            monthly_family_data = monthly_family_data.sort_values('sales_value', ascending=False)
            max_row = monthly_family_data.iloc[0]
            max_value = max_row['sales_value']
            max_entity = max_row['family']
            max_entity_type = "Família"
            max_date_str = max_row['month'].strftime('%b-%y')
        
        # Min calculation
        if st.session_state.time_series_filters['sku']:
            # Single SKU selected - find the min month
            min_month = monthly_data[monthly_data['sales_value'] == monthly_data['sales_value'].min()]['month'].iloc[0]
            min_value = monthly_data['sales_value'].min()
            min_entity = st.session_state.time_series_filters['sku']
            min_entity_type = "SKU"
            min_date_str = min_month.strftime('%b-%y')
        elif st.session_state.time_series_filters['subfamily']:
            # Subfamily selected - find min SKU
            monthly_sku_data = filtered_data.groupby(['month', 'sku'])['sales_value'].sum().reset_index()
            # Filter out zeros or very small values
            monthly_sku_data = monthly_sku_data[monthly_sku_data['sales_value'] > 0.01]
            monthly_sku_data = monthly_sku_data.sort_values('sales_value')
            if not monthly_sku_data.empty:
                min_row = monthly_sku_data.iloc[0]
                min_value = min_row['sales_value']
                min_entity = min_row['sku']
                min_entity_type = "SKU"
                min_date_str = min_row['month'].strftime('%b-%y')
            else:
                min_value = 0
                min_entity = "N/A"
                min_entity_type = "SKU"
                min_date_str = ""
        elif st.session_state.time_series_filters['family']:
            # Family selected - find min subfamily
            monthly_subfamily_data = filtered_data.groupby(['month', 'subfamily'])['sales_value'].sum().reset_index()
            # Filter out zeros or very small values
            monthly_subfamily_data = monthly_subfamily_data[monthly_subfamily_data['sales_value'] > 0.01]
            monthly_subfamily_data = monthly_subfamily_data.sort_values('sales_value')
            if not monthly_subfamily_data.empty:
                min_row = monthly_subfamily_data.iloc[0]
                min_value = min_row['sales_value']
                min_entity = min_row['subfamily']
                min_entity_type = "Subfamília"
                min_date_str = min_row['month'].strftime('%b-%y')
            else:
                min_value = 0
                min_entity = "N/A"
                min_entity_type = "Subfamília"
                min_date_str = ""
        else:
            # Nothing selected - find min family
            monthly_family_data = filtered_data.groupby(['month', 'family'])['sales_value'].sum().reset_index()
            # Filter out zeros or very small values
            monthly_family_data = monthly_family_data[monthly_family_data['sales_value'] > 0.01]
            monthly_family_data = monthly_family_data.sort_values('sales_value')
            if not monthly_family_data.empty:
                min_row = monthly_family_data.iloc[0]
                min_value = min_row['sales_value']
                min_entity = min_row['family']
                min_entity_type = "Família"
                min_date_str = min_row['month'].strftime('%b-%y')
            else:
                min_value = 0
                min_entity = "N/A"
                min_entity_type = "Família"
                min_date_str = ""
        
        # Calcular tendência CAGR comparando 2023 e 2024
        if len(monthly_data) >= 2:
            # Excluir dados de 2025 para a análise de tendência
            filtered_trend_data = filtered_data[filtered_data['invoice_date'].dt.year < 2025]
            
            # Dados de 2023
            data_2023 = filtered_trend_data[filtered_trend_data['invoice_date'].dt.year == 2023]
            # Dados de 2024
            data_2024 = filtered_trend_data[filtered_trend_data['invoice_date'].dt.year == 2024]
            
            # Verificar se temos dados para ambos os anos
            if not data_2023.empty and not data_2024.empty:
                # Calcular valor total para cada ano (já aplicando os filtros corretos)
                value_2023 = data_2023['sales_value'].sum()
                value_2024 = data_2024['sales_value'].sum()
                
                # Calcular CAGR: (Valor Final/Valor Inicial)^(1/n) - 1
                # Utilizando n = 2 anos na fórmula
                if value_2023 > 0:  # Evitar divisão por zero
                    cagr = ((value_2024 / value_2023) ** (1/2) - 1) * 100  # CAGR com n=2
                    
                    # Adicionar informação de debug
                    st.session_state.cagr_debug = {
                        "2023": value_2023,
                        "2024": value_2024,
                        "n": 2,
                        "cagr": cagr
                    }
                else:
                    cagr = 0
                    st.session_state.cagr_debug = {
                        "error": "Valor de 2023 é zero - impossível calcular crescimento"
                    }
            else:
                # Não temos dados suficientes para calcular
                cagr = 0
                st.session_state.cagr_debug = {
                    "error": "Sem dados suficientes para 2023 e/ou 2024",
                    "has_2023": not data_2023.empty,
                    "has_2024": not data_2024.empty
                }
        else:
            cagr = 0
            st.session_state.cagr_debug = {
                "error": "Sem dados mensais suficientes"
            }
                
        # Display statistics
        card_style = """
        <style>
        .metric-card {
            background-color: #1E2130;
            border-radius: 5px;
            padding: 15px;
            text-align: left;
        }
        .metric-title {
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 10px;
            color: #FFFFFF;
        }
        .metric-value {
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 10px;
            color: #FFFFFF;
        }
        .metric-delta {
            font-size: 14px;
            color: #98FB98;
        }
        .metric-date {
            font-size: 14px;
            color: #AAAAAA;
        }
        </style>
        """
        st.markdown(card_style, unsafe_allow_html=True)
        
        # Create the context label based on filters
        if st.session_state.time_series_filters['sku']:
            context_label = f"SKU: {st.session_state.time_series_filters['sku']}"
        elif st.session_state.time_series_filters['subfamily']:
            context_label = f"Subfamília: {st.session_state.time_series_filters['subfamily']}"
        elif st.session_state.time_series_filters['family']:
            context_label = f"Família: {st.session_state.time_series_filters['family']}"
        else:
            context_label = "Total"
            
        # Show range dates
        range_str = f"{min_date.strftime('%b-%y')} a {max_date.strftime('%b-%y')}"
        
        with stat_col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Média Mensal</div>
                    <div class="metric-value">{mean_monthly:,.2f} Kg</div>
                    <div class="metric-delta">{context_label}</div>
                    <div class="metric-date">{range_str}</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        with stat_col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Máximo</div>
                    <div class="metric-value">{max_value:,.2f} Kg</div>
                    <div class="metric-delta">{max_entity_type}: {max_entity}</div>
                    <div class="metric-date">{max_date_str}</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        with stat_col3:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Mínimo</div>
                    <div class="metric-value">{min_value:,.2f} Kg</div>
                    <div class="metric-delta">{min_entity_type}: {min_entity}</div>
                    <div class="metric-date">{min_date_str}</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
        with stat_col4:
            # Adicionando setas visuais e formatação baseada no valor do CAGR
            if cagr >= 0:
                trend_text = "Crescimento"
                arrow_icon = "▲"  # Seta para cima
                color = "#33cc33"  # Verde
            else:
                trend_text = "Decrescimento"
                arrow_icon = "▼"  # Seta para baixo
                color = "#ff3333"  # Vermelho
            
            # Criar uma string informativa para os valores usados
            if hasattr(st.session_state, 'cagr_debug') and not isinstance(st.session_state.cagr_debug.get('error', None), str):
                debug_info = f"2023: {st.session_state.cagr_debug.get('2023', 0):,.2f} Kg | 2024: {st.session_state.cagr_debug.get('2024', 0):,.2f} Kg"
            else:
                debug_info = "Dados insuficientes para comparação"
            
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-title">Tendência</div>
                    <div class="metric-value" style="color: {color};">{trend_text} {arrow_icon}</div>
                    <div class="metric-delta">CAGR {cagr:.2f}%</div>
                    <div class="metric-date" title="{debug_info}">2023-2024</div>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Adicionar botão de detalhes caso queira verificar os valores
            with st.expander("Detalhes do cálculo", expanded=False):
                if hasattr(st.session_state, 'cagr_debug'):
                    if 'error' in st.session_state.cagr_debug and st.session_state.cagr_debug['error']:
                        st.warning(st.session_state.cagr_debug['error'])
                    else:
                        st.write(f"**Valor 2023:** {st.session_state.cagr_debug.get('2023', 0):,.2f} Kg")
                        st.write(f"**Valor 2024:** {st.session_state.cagr_debug.get('2024', 0):,.2f} Kg")
                        st.write(f"**Período (n):** {st.session_state.cagr_debug.get('n', 2)} anos")
                        st.write(f"**CAGR:** {st.session_state.cagr_debug.get('cagr', 0):.2f}%")
                        st.caption("Fórmula: ((Valor 2024 / Valor 2023)^(1/2) - 1) * 100%") 
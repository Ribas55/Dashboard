"""
Dashboard entry point for sales time series analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import plotly.express as px
import plotly.graph_objects as go
from src.data_loader import load_sales_data
from src.visualizations import timeseries_visualization, get_sku_dropdown_options
from src.abc_xyz import abc_xyz_classification, create_abc_xyz_matrix, filter_active_skus, entity_abc_xyz_classification
from src.mercado_especifico import get_mercado_especifico_skus
from utils.filters import filter_dataframe, filter_date_range
from typing import Optional
# Import page modules moved after set_page_config
# from pages import overview, time_series, aggregated_series, abc_xyz_page, intermittency_page, weighted_forecast_page
import os

# Configure a página para esconder os arquivos Python da barra lateral
st.set_page_config(
    page_title="Análise de Vendas",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Page Imports --- (Moved Here)
from pages import overview, time_series, aggregated_series, abc_xyz_page, intermittency_page, weighted_forecast_page
from pages import forecasting_methods_page # <-- Import the new page
from pages import results_comparison_page # Add this import

# CSS adicional para forçar a sidebar ao topo absoluto
st.markdown("""
<style>
    /* Hack para forçar a sidebar ao topo absoluto da página */
    iframe {
        position: fixed;
        top: 0;
        height: 0;
        width: 0;
    }
</style>
""", unsafe_allow_html=True)

# Ocultar elementos não desejados da UI
st.markdown("""
<style>
    /* Ocultar elementos indesejados */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    [data-testid="stSidebarNav"] {display: none;}
    
    /* Reset de margens e paddings do corpo da página */
    body {
        margin: 0;
        padding: 0;
    }
    
    /* Estilo base para o tema escuro */
    .main {
        background-color: #111520;
        color: white;
        padding-top: 0;
        margin-top: 0;
    }
    
    .stApp {
        background-color: #111520;
    }
    
    /* Remover espaços desnecessários e alinhar ao topo */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
        margin-left: 270px;
        padding-left: 1rem;
    }
    
    /* Ajustar sidebar para ficar absolutamente no topo e fixa durante a rolagem */
    section[data-testid="stSidebar"] {
        position: fixed;
        top: 0;
        left: 0;
        bottom: 0;
        width: 270px;
        overflow-y: auto;
        background-color: #1E2130;
        border-right: 1px solid rgba(255, 255, 255, 0.1);
        padding: 0;
        margin: 0;
    }
    
    /* Zerar todos os espaços na div principal da sidebar */
    section[data-testid="stSidebar"] > div:first-child {
        padding: 0;
        margin: 0;
    }
    
    /* Zerar espaços nos containers da sidebar */
    section[data-testid="stSidebar"] .block-container {
        padding: 0 !important;
        margin: 0 !important;
    }
    
    /* Garantir que não haja espaço em nenhum elemento dentro da sidebar */
    section[data-testid="stSidebar"] .element-container {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Ajustar o título da sidebar */
    .sidebar-title {
        color: white;
        font-size: 24px;
        font-weight: 600;
        margin: 0;
        padding: 12px 10px;
        text-align: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        position: sticky;
        top: 0;
        background-color: #1E2130;
        z-index: 100;
    }
    
    /* Container para menu */
    .menu-container {
        padding: 5px 10px;
        margin: 0;
    }
    
    /* Botões mais juntos para otimizar espaço */
    div[data-testid="stButton"] {
        margin-bottom: 3px;
        height: 40px; /* Altura fixa para todos os botões */
    }
    
    /* Estilo para todos os botões */
    div[data-testid="stButton"] > button {
        background-color: transparent !important;
        color: #d1d1d1 !important;
        border: none !important;
        border-radius: 5px !important;
        text-align: left !important;
        width: 100% !important;
        padding: 10px 15px !important;
        font-weight: normal !important;
        font-size: 14px !important;
        box-shadow: none !important;
        transition: all 0.2s ease !important;
        display: flex !important;
        align-items: center !important;
        justify-content: flex-start !important;
        height: 100% !important; /* Altura 100% do container */
        border-left: 2px solid transparent !important; /* Borda transparente para todos */
    }
    
    div[data-testid="stButton"] > button:hover {
        background-color: rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Estilo para botão ativo - sombreado vermelho suave */
    div[data-testid="stButton"] > button.active {
        background-color: rgba(220, 0, 0, 0.25) !important;
        color: white !important;
        border-left: 2px solid rgba(220, 0, 0, 0.9) !important;
    }
    
    /* Carregar Dados botão - estilo especial */
    div[data-testid="stButton"]:last-child > button {
        background-color: #1e2130 !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        justify-content: center !important;
        margin-top: 5px;
        border-left: 1px solid rgba(255, 255, 255, 0.1) !important; /* Substitui a borda à esquerda */
    }
    
    div[data-testid="stButton"]:last-child > button:hover {
        background-color: rgba(255, 255, 255, 0.05) !important;
    }
    
    /* Fonte de dados título */
    .fonte-dados-title {
        color: white;
        font-size: 16px;
        font-weight: normal;
        margin-top: 0;
        margin-bottom: 10px;
        border-top: 1px solid rgba(255, 255, 255, 0.1);
        padding: 15px 10px 8px 10px;
        text-align: left;
        opacity: 0.8;
    }
    
    /* Estilo para ícones nos botões */
    div[data-testid="stButton"] span {
        display: flex !important;
        align-items: center !important;
    }
    
    /* Correções para elementos específicos da interface */
    .stSelectbox, .stMultiselect {
        margin-top: 0;
    }
    
    .stTabs {
        background-color: #1E2130;
        border-radius: 5px;
    }
    
    /* Ajuste para o topo de páginas com títulos */
    h1, h2, h3 {
        margin-top: 0;
        padding-top: 0;
    }
    
    /* Ajustes para marcar corretamente a altura da sidebar */
    .element-container {
        margin-bottom: 0.5rem;
    }

    /* Ajuste para alinhar o conteúdo principal com o título da sidebar */
    .stApp [data-testid="stAppViewBlockContainer"] > div {
        padding-top: 12px;
    }

    /* Info message styling to align with sidebar title */
    .stAlert {
        margin-top: 0;
    }
</style>
""", unsafe_allow_html=True)

# JavaScript para adicionar classe ao botão ativo
st.markdown("""
<script>
document.addEventListener('DOMContentLoaded', function() {
    const currentPage = "%s";
    const buttons = document.querySelectorAll('[data-testid="stButton"] button');
    
    buttons.forEach(button => {
        if (button.innerText.includes(currentPage)) {
            button.classList.add('active-button');
        }
    });
});
</script>
""" % st.session_state.get('current_page', 'Visão Geral'), unsafe_allow_html=True)

# Add JavaScript to force the sidebar to scroll to top on page load
st.markdown("""
<script>
document.addEventListener('DOMContentLoaded', function() {
    // Force sidebar to scroll to top on page load
    setTimeout(function() {
        const sidebar = document.querySelector('section[data-testid="stSidebar"]');
        if (sidebar) {
            sidebar.scrollTop = 0;
        }
    }, 100);
});
</script>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
    
if 'sales_data' not in st.session_state:
    st.session_state.sales_data = None
    
if 'abc_xyz_data' not in st.session_state:
    st.session_state.abc_xyz_data = None
    
if 'active_skus' not in st.session_state:
    st.session_state.active_skus = None

if 'mercado_especifico_skus' not in st.session_state:
    st.session_state.mercado_especifico_skus = []

def create_aggregated_time_series(
    df: pd.DataFrame, 
    group_by: str, 
    group_value: Optional[str] = None,
    date_col: str = "invoice_date", 
    value_col: str = "sales_value", 
    period: str = "M",
    normalize: bool = False,
    show_only_total: bool = False
) -> go.Figure:
    """
    Create a time series chart for aggregated data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe with sales data
    group_by : str
        Column to group by (e.g., 'family', 'subfamily', 'sku')
    group_value : str, optional
        Specific value to filter by (if None, shows top 3)
    date_col : str
        Date column name
    value_col : str
        Value column name
    period : str
        Aggregation period ('D' for day, 'M' for month, 'Q' for quarter)
    normalize : bool
        Whether to normalize values to 0-1 scale
    show_only_total : bool
        If True, only shows the aggregated total line
        
    Returns:
    --------
    go.Figure
        Plotly figure with the time series
    """
    # Make a copy and ensure date column is datetime
    df_copy = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_col]):
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
    
    # Create a period column for aggregation
    df_copy['period'] = df_copy[date_col].dt.to_period(period)
    
    # Filter by group value if provided
    if group_value is not None and group_by in df_copy.columns:
        df_copy = df_copy[df_copy[group_by] == group_value]
    
    # Aggregate by period and group
    if group_by in df_copy.columns:
        # If no specific group selected, get top 3 by sales volume
        if group_value is None:
            top_groups = df_copy.groupby(group_by)[value_col].sum().nlargest(3).index.tolist()
            df_filtered = df_copy[df_copy[group_by].isin(top_groups)]
            
            # Aggregate by period and group
            agg_data = df_filtered.groupby(['period', group_by])[value_col].sum().reset_index()
            
            # Pivot to have one column per group
            pivot_data = agg_data.pivot(index='period', columns=group_by, values=value_col)
            # Convert Period index to datetime index explicitly
            pivot_data.index = pd.DatetimeIndex([p.to_timestamp() for p in pivot_data.index])
            pivot_data = pivot_data.sort_index()
            
            # Normalize if requested
            if normalize and not pivot_data.empty:
                for col in pivot_data.columns:
                    min_val = pivot_data[col].min()
                    max_val = pivot_data[col].max()
                    if max_val > min_val:
                        pivot_data[col] = (pivot_data[col] - min_val) / (max_val - min_val)
            
            # Create the plot
            fig = go.Figure()
            
            # Se show_only_total for True, calcular e mostrar apenas a soma total
            if show_only_total:
                total_data = pivot_data.sum(axis=1)
                fig.add_trace(go.Scatter(
                    x=pivot_data.index, 
                    y=total_data,
                    name="Total",
                    mode='lines',
                    line=dict(color='#ffffff', width=3)
                ))
            else:
                # Add traces for each group
                for col in pivot_data.columns:
                    fig.add_trace(go.Scatter(
                        x=pivot_data.index, 
                        y=pivot_data[col],
                        name=str(col),
                        mode='lines+markers'
                    ))
                
            title = f"Time Series by {group_by.capitalize()}"
            if normalize:
                title += " (Normalized)"
            if show_only_total:
                title += " (Total Only)"
            
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Normalized Value" if normalize else "Sales Value",
                legend_title=group_by.capitalize(),
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
            )
            
        else:
            # For a specific group, aggregate by period
            agg_data = df_copy.groupby('period')[value_col].sum().reset_index()
            agg_data['period'] = agg_data['period'].dt.to_timestamp()
            agg_data = agg_data.sort_values('period')
            
            # Normalize if requested
            if normalize and not agg_data.empty:
                min_val = agg_data[value_col].min()
                max_val = agg_data[value_col].max()
                if max_val > min_val:
                    agg_data['normalized'] = (agg_data[value_col] - min_val) / (max_val - min_val)
                    value_col = 'normalized'
            
            # Create the plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=agg_data['period'], 
                y=agg_data[value_col],
                name=str(group_value) if not show_only_total else "Total",
                mode='lines+markers'
            ))
                
            title = f"Time Series for {group_by.capitalize()}: {group_value}"
            if normalize:
                title += " (Normalized)"
            # Se for apenas total, ajustar o título
            if show_only_total:
                title = f"Total Sales for {group_by.capitalize()}"
            
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Normalized Value" if normalize else "Sales Value",
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
            )
    else:
        # If group_by column not in dataframe, aggregate all data by period
        agg_data = df_copy.groupby('period')[value_col].sum().reset_index()
        agg_data['period'] = agg_data['period'].dt.to_timestamp()
        agg_data = agg_data.sort_values('period')
        
        # Normalize if requested
        if normalize and not agg_data.empty:
            min_val = agg_data[value_col].min()
            max_val = agg_data[value_col].max()
            if max_val > min_val:
                agg_data['normalized'] = (agg_data[value_col] - min_val) / (max_val - min_val)
                value_col = 'normalized'
        
        # Create the plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=agg_data['period'], 
            y=agg_data[value_col],
            name="Total",
            mode='lines+markers'
        ))
            
        title = "Time Series for All Data"
        if normalize:
            title += " (Normalized)"
            
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Normalized Value" if normalize else "Sales Value",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
        )
    
    # Add range slider
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    return fig

def create_percentage_distribution(df: pd.DataFrame, group_by: str, value_col: str = "sales_value") -> go.Figure:
    """
    Create a percentage distribution visualization.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe with sales data
    group_by : str
        Column to group by (e.g., 'family', 'subfamily', 'sku')
    value_col : str
        Value column name
        
    Returns:
    --------
    go.Figure
        Plotly figure with the distribution
    """
    if group_by not in df.columns:
        # Return empty figure if group_by column not in dataframe
        fig = go.Figure()
        fig.update_layout(
            title=f"Distribution by {group_by.capitalize()} (No data)",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        return fig
    
    # Calculate total sales by group
    group_sales = df.groupby(group_by)[value_col].sum().reset_index()
    group_sales = group_sales.sort_values(value_col, ascending=False)
    
    # Calculate percentages
    total_sales = group_sales[value_col].sum()
    group_sales['percentage'] = group_sales[value_col] / total_sales * 100
    
    # Get top 5 groups and combine the rest
    if len(group_sales) > 5:
        top5 = group_sales.head(5)
        others_sum = group_sales.iloc[5:][value_col].sum()
        others_pct = group_sales.iloc[5:]['percentage'].sum()
        others = pd.DataFrame({
            group_by: ['Others'],
            value_col: [others_sum],
            'percentage': [others_pct]
        })
        plot_data = pd.concat([top5, others])
    else:
        plot_data = group_sales
    
    # Create a pie chart
    fig = px.pie(
        plot_data, 
        values='percentage', 
        names=group_by,
        title=f"Sales Distribution by {group_by.capitalize()}"
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    return fig

def main():
    """Main application entry point."""
    
    # Initialize session state for navigation
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Visão Geral"
    
    # Sidebar navigation
    with st.sidebar:
        # Remover qualquer espaço antes do título
        st.markdown("""
        <style>
            /* Remover espaço antes do primeiro elemento no app */
            div.block-container {padding-top: 0; margin-top: 0;}
            /* Forçar sidebar a começar absolutamente no topo */
            [data-testid="stSidebar"] {top: 0 !important; margin-top: 0 !important;}
            /* Título da sidebar sticky */
            .sidebar-title {position: sticky; top: 0; z-index: 100; background-color: #1E2130;}
        </style>
        """, unsafe_allow_html=True)
        
        # Título da sidebar
        st.markdown('<div class="sidebar-title">Análise de Vendas</div>', unsafe_allow_html=True)
        
        # Container para menu - melhor organização
        st.markdown('<div class="menu-container">', unsafe_allow_html=True)
        
        # Definir opções de menu com ícones
        menu_options = [
            {"id": "Visão Geral", "icon": "🏠", "label": "Visão Geral"},
            {"id": "Séries Temporais", "icon": "📈", "label": "Séries Temporais"},
            {"id": "Séries Agregadas", "icon": "📊", "label": "Séries Agregadas"},
            {"id": "Análise ABC/XYZ", "icon": "📋", "label": "Análise ABC/XYZ"},
            {"id": "Análise de Intermitência", "icon": "⚡", "label": "Análise de Intermitência"},
            {"id": "Previsão Ponderada", "icon": "⚖️", "label": "Previsão Ponderada"},
            {"id": "Métodos de Forecasting", "icon": "🔮", "label": "Métodos de Forecasting"},
            {"id": "Resultados e Comparações", "icon": "🆚", "label": "Resultados e Comparações"}
        ]
        
        # CSS global para todos os botões de uma vez, evitando alterações dinâmicas que movem os botões
        active_css = """
        <style>
        """
        
        for option in menu_options:
            is_active = st.session_state.current_page == option["id"]
            if is_active:
                active_css += f"""
                [data-testid="stButton"] button[key="menu_{option['id']}"] {{
                    background-color: rgba(220, 0, 0, 0.25) !important;
                    color: white !important;
                    border-left: 2px solid rgba(220, 0, 0, 0.9) !important;
                }}
                """
        
        active_css += """
        </style>
        """
        
        # Aplicar o CSS de uma vez só
        st.markdown(active_css, unsafe_allow_html=True)
        
        # Criar menu com botões estilizados uniformemente
        for option in menu_options:
            # Criar botão com ícone
            button_label = f"{option['icon']}  {option['label']}"
            if st.button(button_label, key=f"menu_{option['id']}", use_container_width=True):
                st.session_state.current_page = option["id"]
                st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Seção de fonte de dados
        st.markdown('<div class="fonte-dados-title">Fonte de Dados</div>', unsafe_allow_html=True)
        file_path = "data/2022-2025.xlsx"
        
        # Botão de carregar dados
        if st.button("📁  Carregar Dados", key="load_data_button", use_container_width=True):
            with st.spinner("Carregando dados da planilha BD_Vendas..."):
                try:
                    df = load_sales_data(file_path, sheet_name="BD_Vendas")
                    
                    # Garantir que a coluna SKU seja sempre string para consistência nas comparações
                    if 'sku' in df.columns:
                        df['sku'] = df['sku'].astype(str)
                    
                    st.session_state.sales_data = df
                    st.session_state.data_loaded = True
                    active_skus = filter_active_skus(df, year=2024)
                    st.session_state.active_skus = active_skus
                    
                    with st.spinner("Realizando classificação ABC/XYZ..."):
                        # Classificação ABC/XYZ para SKUs
                        abc_xyz_df = abc_xyz_classification(df, year=2024)
                        
                        # Garantir que a coluna SKU seja string
                        if 'sku' in abc_xyz_df.columns:
                            abc_xyz_df['sku'] = abc_xyz_df['sku'].astype(str)
                        
                        st.session_state.abc_xyz_data = abc_xyz_df
                        
                        # Classificação ABC/XYZ para famílias
                        family_abc_xyz_df = entity_abc_xyz_classification(df, "family", year=2024)
                        st.session_state.family_abc_xyz_data = family_abc_xyz_df
                        
                        # Classificação ABC/XYZ para subfamílias
                        subfamily_abc_xyz_df = entity_abc_xyz_classification(df, "subfamily", year=2024)
                        st.session_state.subfamily_abc_xyz_data = subfamily_abc_xyz_df
                    
                    with st.spinner("Carregando SKUs de mercado específico..."):
                        # Carregar SKUs de mercado específico
                        mercado_especifico_skus = get_mercado_especifico_skus()
                        st.session_state.mercado_especifico_skus = mercado_especifico_skus
                    
                    st.success(f"Dados carregados: {len(df)} linhas")
                except Exception as e:
                    st.error(f"Erro ao carregar dados: {str(e)}")
    
    # Main content area
    if not st.session_state.data_loaded:
        # Add padding-top to align with sidebar title
        st.markdown("""
        <style>
            [data-testid="stAlert"] {margin-top: 0;}
        </style>
        """, unsafe_allow_html=True)
        st.info("Por favor, carregue os dados usando o botão no painel lateral para começar.")
        return
    
    # Add padding-top for consistent alignment when data is loaded
    st.markdown("""
    <style>
        .main .block-container {padding-top: 12px !important;}
    </style>
    """, unsafe_allow_html=True)
    
    # Display content based on selected page
    if st.session_state.current_page == "Visão Geral":
        # st.write("Visão Geral (Page content commented out for testing)")
        overview.render()
    elif st.session_state.current_page == "Séries Temporais":
        time_series.render()
    elif st.session_state.current_page == "Séries Agregadas":
        aggregated_series.render()
    elif st.session_state.current_page == "Análise ABC/XYZ":
        abc_xyz_page.render()
    elif st.session_state.current_page == "Análise de Intermitência":
        intermittency_page.render()
    elif st.session_state.current_page == "Previsão Ponderada":
        weighted_forecast_page.render()
    elif st.session_state.current_page == "Métodos de Forecasting":
        forecasting_methods_page.render()
    elif st.session_state.current_page == "Resultados e Comparações":
        results_comparison_page.render()

if __name__ == "__main__":
    main() 
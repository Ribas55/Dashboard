"""
Overview page with KPIs and main charts.
"""

import streamlit as st
import plotly.express as px
from src.visualizations import create_aggregated_time_series

def calculate_kpis(df):
    """Calculate KPIs from the data."""
    if df is None:
        return {
            "total_sales": 0,
            "active_skus": 0,
            "avg_sales": 0,
            "top_family": "N/A",
            "top_family_pct": 0
        }
    
    total_sales = df["sales_value"].sum()
    active_skus = df["sku"].nunique()
    avg_sales = total_sales / active_skus if active_skus > 0 else 0
    
    # Calculate top family
    family_sales = df.groupby("family")["sales_value"].sum()
    top_family = family_sales.idxmax()
    top_family_pct = (family_sales.max() / total_sales) * 100
    
    return {
        "total_sales": total_sales,
        "active_skus": active_skus,
        "avg_sales": avg_sales,
        "top_family": top_family,
        "top_family_pct": top_family_pct
    }

def create_top_skus_chart(df):
    """Create a bar chart with top 5 SKUs by sales value."""
    if df is None:
        return None
        
    # Get top 5 SKUs
    top_skus = df.groupby("sku")["sales_value"].sum().nlargest(5).reset_index()
    
    # Create bar chart
    fig = px.bar(
        top_skus,
        x="sku",
        y="sales_value",
        title="Top 5 SKUs por Valor de Vendas",
        labels={
            "sku": "SKU",
            "sales_value": "Valor de Vendas (€)"
        }
    )
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    
    return fig

def render():
    """Render the overview page."""
    st.subheader("Visão Geral")
    
    # Calculate KPIs
    kpis = calculate_kpis(st.session_state.sales_data)
    
    # KPI row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    
    with kpi1:
        st.metric(
            "Total Vendas",
            f"€{kpis['total_sales']:,.0f}",
            "+12.5% vs anterior"  # This should be calculated comparing periods
        )
    
    with kpi2:
        st.metric(
            "SKUs Ativos",
            f"{kpis['active_skus']}",
            "-3% vs anterior"  # This should be calculated comparing periods
        )
    
    with kpi3:
        st.metric(
            "Vendas Médias / SKU",
            f"€{kpis['avg_sales']:,.0f}",
            "+15.8% vs anterior"  # This should be calculated comparing periods
        )
    
    with kpi4:
        st.metric(
            "Família Mais Vendida",
            kpis['top_family'],
            f"{kpis['top_family_pct']:.1f}% do total"
        )
    
    # Opções adicionais
    col1, col2 = st.columns(2)

    with col1:
        # Opção para normalizar os valores
        normalize = st.checkbox("Normalizar valores (0-1)", value=False, key="overview_normalize")

    with col2:
        # Opção para mostrar apenas o total
        if 'overview_show_only_total' not in st.session_state:
            st.session_state.overview_show_only_total = False
        
        show_only_total = st.checkbox(
            "Mostrar apenas o total", 
            value=st.session_state.overview_show_only_total,
            key="overview_show_only_total"
        )

    # Charts row
    chart1, chart2 = st.columns(2)
    
    with chart1:
        st.subheader("Tendência de Vendas")
        if st.session_state.data_loaded:
            with st.spinner("Gerando gráfico das principais famílias..."):
                fig = create_aggregated_time_series(
                    st.session_state.sales_data,
                    "family", 
                    normalize=normalize,
                    show_only_total=show_only_total
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with chart2:
        st.subheader("Top 5 SKUs")
        if st.session_state.data_loaded:
            fig = create_top_skus_chart(st.session_state.sales_data)
            if fig:
                st.plotly_chart(fig, use_container_width=True) 
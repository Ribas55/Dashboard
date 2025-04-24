"""
Página de Análise ABC/XYZ com matriz interativa e filtros por família, subfamília e tipo de mercado.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.abc_xyz_analysis import (
    get_abc_xyz_data,
    create_abc_xyz_matrix,
    create_quadrant_chart,
    get_sku_details,
    get_subfamily_options,
    get_quadrant_interpretation
)
from src.mercado_especifico import get_mercado_especifico_skus

def render():
    """Render the ABC/XYZ analysis page."""
    # Adicionar estilo CSS personalizado para melhorar a aparência
    st.markdown("""
    <style>
    .stButton button {
        background-color: #1E88E5;
        font-weight: bold;
    }
    .stDataFrame {
        background-color: rgba(30, 30, 45, 0.2);
    }
    .stExpander {
        background-color: rgba(50, 50, 70, 0.2);
    }
    h1, h2, h3 {
        color: #FFFFFF;
    }
    .interpretation-box {
        background-color: rgba(50, 50, 70, 0.3);
        padding: 15px;
        border-radius: 5px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("Análise ABC/XYZ")
    
    if not st.session_state.data_loaded:
        st.info("Por favor, carregue os dados primeiro.")
        return
    
    # Get data
    df = st.session_state.sales_data
    
    # Ensure necessary columns are available
    required_columns = ["sku", "family", "subfamily", "sales_value", "invoice_date"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Colunas necessárias não encontradas: {', '.join(missing_columns)}")
        return
    
    # Load list of specific market SKUs if not already loaded
    if 'mercado_especifico_skus' not in st.session_state or not st.session_state.mercado_especifico_skus:
        with st.spinner("Carregando SKUs do mercado específico..."):
            mercado_especifico_skus = get_mercado_especifico_skus()
            st.session_state.mercado_especifico_skus = mercado_especifico_skus
    
    # Get the 15 specific families
    default_families = [
        "Cream Cracker", "Maria", "Wafer", "Sortido", 
        "Cobertas de Chocolate", "Água e Sal", "Digestiva",
        "Recheada", "Circus", "Tartelete", "Torrada",
        "Flocos de Neve", "Integral", "Mentol", "Aliança"
    ]
    
    # Filter only the specified families
    df_filtered = df[df["family"].isin(default_families)]
    
    # Check if we have filtered data
    if df_filtered.empty:
        st.warning("Não há dados para as famílias específicas.")
        return
    
    # Initialize session state for filters
    if 'abc_xyz_family' not in st.session_state:
        st.session_state.abc_xyz_family = "Todas"
    
    if 'abc_xyz_subfamily' not in st.session_state:
        st.session_state.abc_xyz_subfamily = "Todas"
    
    if 'abc_xyz_market' not in st.session_state:
        st.session_state.abc_xyz_market = "Todos"
    
    if 'abc_xyz_data' not in st.session_state:
        st.session_state.abc_xyz_data = None
    
    if 'abc_xyz_stats' not in st.session_state:
        st.session_state.abc_xyz_stats = {}
    
    # Create filter layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Família")
        # All families + Todas option
        family_options = ["Todas"] + sorted(default_families)
        selected_family = st.selectbox(
            "Selecione a família:",
            options=family_options,
            index=family_options.index(st.session_state.abc_xyz_family),
            key="family_selector"
        )
        
        # Store the selected value
        st.session_state.abc_xyz_family = selected_family
    
    with col2:
        st.subheader("Subfamília")
        # Get subfamily options based on family selection
        subfamily_options = get_subfamily_options(df_filtered, selected_family)
        
        # Select default index based on current selection or default to "Todas"
        current_subfamily = st.session_state.abc_xyz_subfamily
        if current_subfamily not in subfamily_options:
            current_subfamily = "Todas"
            st.session_state.abc_xyz_subfamily = current_subfamily
        
        selected_subfamily = st.selectbox(
            "Selecione a subfamília:",
            options=subfamily_options,
            index=subfamily_options.index(current_subfamily),
            key="subfamily_selector"
        )
        
        # Store the selected value
        st.session_state.abc_xyz_subfamily = selected_subfamily
    
    with col3:
        st.subheader("Período")
        # For now, we'll always use 2024 as the analysis period
        st.selectbox(
            "Período de análise:",
            options=["Último Ano"],
            index=0,
            disabled=True
        )
    
    # ABC/XYZ thresholds row
    st.subheader("Limites ABC/XYZ")
    col1, col2 = st.columns(2)
    
    with col1:
        # ABC thresholds
        abc_col1, abc_col2, abc_col3 = st.columns([1, 0.2, 1])
        with abc_col1:
            a_threshold = st.number_input("", min_value=50, max_value=90, value=80, step=5, label_visibility="collapsed")
        with abc_col2:
            st.write("-")
        with abc_col3:
            b_threshold = st.number_input("", min_value=90, max_value=99, value=95, step=1, label_visibility="collapsed")
        
        st.caption("Limites ABC (%)")
    
    with col2:
        # XYZ thresholds
        xyz_col1, xyz_col2, xyz_col3 = st.columns([1, 0.2, 1])
        with xyz_col1:
            x_threshold = st.number_input("", min_value=10, max_value=30, value=20, step=5, label_visibility="collapsed")
        with xyz_col2:
            st.write("-")
        with xyz_col3:
            y_threshold = st.number_input("", min_value=30, max_value=70, value=50, step=5, label_visibility="collapsed")
        
        st.caption("Limites XYZ (%)")
    
    # Market type selector
    market_options = ["Todos", "Normal", "Mercado Específico"]
    market_cols = st.columns([2, 1])
    
    with market_cols[0]:
        st.subheader("SKU:")
        selected_market = st.selectbox(
            "Tipo de mercado:",
            options=market_options,
            index=market_options.index(st.session_state.abc_xyz_market),
            key="market_selector"
        )
        
        # Store the selected value
        st.session_state.abc_xyz_market = selected_market
    
    with market_cols[1]:
        st.write("")  # Spacer
        st.write("")  # Spacer
        view_matrix_button = st.button(
            "📊 Ver Matriz",
            type="primary",
            use_container_width=True
        )
    
    # Create a divider
    st.divider()
    
    # Inicialize uma variável de estado para controlar se o botão foi clicado
    if 'abc_xyz_button_clicked' not in st.session_state:
        st.session_state.abc_xyz_button_clicked = False
    
    # Process data and show matrix only when the button is clicked
    if view_matrix_button:
        st.session_state.abc_xyz_button_clicked = True
        with st.spinner("Calculando classificação ABC/XYZ..."):
            # Calculate ABC/XYZ classification with filters
            abc_xyz_data = get_abc_xyz_data(
                df=df_filtered,
                family_filter=selected_family if selected_family != "Todas" else None,
                subfamily_filter=selected_subfamily if selected_subfamily != "Todas" else None,
                market_type=selected_market,
                mercado_especifico_skus=st.session_state.mercado_especifico_skus,
                a_threshold=a_threshold/100,  # Convert percentage to decimal
                b_threshold=b_threshold/100,  # Convert percentage to decimal
                x_threshold=x_threshold/100,  # Convert percentage to decimal
                y_threshold=y_threshold/100   # Convert percentage to decimal
            )
            
            # Store the data for reuse
            st.session_state.abc_xyz_data = abc_xyz_data
            
            # If no data found with the current filters
            if abc_xyz_data.empty:
                st.warning("Não foram encontrados SKUs com os filtros selecionados.")
                st.session_state.abc_xyz_button_clicked = False  # Reset para false quando não há dados
                return
    
    # Only proceed if the button was clicked AND we have ABC/XYZ data calculated
    if st.session_state.abc_xyz_button_clicked and st.session_state.abc_xyz_data is not None and not st.session_state.abc_xyz_data.empty:
        # Create a title that reflects the current filters
        title_suffix = []
        if selected_family != "Todas":
            title_suffix.append(f"Família: {selected_family}")
        if selected_subfamily != "Todas":
            title_suffix.append(f"Subfamília: {selected_subfamily}")
        if selected_market != "Todos":
            title_suffix.append(f"Mercado: {selected_market}")
        
        matrix_title = "Matriz ABC/XYZ"
        if title_suffix:
            matrix_title += f" ({', '.join(title_suffix)})"
        
        # Create layout for matrix and quadrant chart
        matrix_col, quadrant_col = st.columns([3, 2])
        
        # Show the ABC/XYZ matrix
        with matrix_col:
            st.subheader("Matriz ABC/XYZ")
            matrix_fig, matrix_stats = create_abc_xyz_matrix(
                st.session_state.abc_xyz_data,
                custom_title=matrix_title
            )
            st.plotly_chart(matrix_fig, use_container_width=True)
            
            # Store matrix stats
            st.session_state.abc_xyz_stats = matrix_stats
        
        # Show the quadrant chart
        with quadrant_col:
            st.subheader("SKUs por Quadrante")
            quadrant_fig = create_quadrant_chart(st.session_state.abc_xyz_data)
            st.plotly_chart(quadrant_fig, use_container_width=True)
        
        # Interpretação dos Quadrantes
        st.subheader("Interpretação dos Quadrantes:")
        
        # Show key quadrant interpretations
        interp_cols = st.columns(3)
        with interp_cols[0]:
            st.markdown(
                f"""
                <div style="background-color:rgba(0,200,0,0.2); padding:10px; border-radius:5px;">
                <span style="color:#33cc33;">■</span> <b>AX:</b> {get_quadrant_interpretation('AX')}
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with interp_cols[1]:
            st.markdown(
                f"""
                <div style="background-color:rgba(200,200,0,0.2); padding:10px; border-radius:5px;">
                <span style="color:#ffcc00;">■</span> <b>BY:</b> {get_quadrant_interpretation('BY')}
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with interp_cols[2]:
            st.markdown(
                f"""
                <div style="background-color:rgba(200,0,0,0.2); padding:10px; border-radius:5px;">
                <span style="color:#ff3300;">■</span> <b>CZ:</b> {get_quadrant_interpretation('CZ')}
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # SKUs Analisados
        st.subheader("SKUs Analisados")
        
        # Get top 5 SKUs by sales volume
        top_skus = get_sku_details(st.session_state.abc_xyz_data, top_n=5)
        
        if not top_skus.empty:
            # Display top SKUs in a table
            # Format columns for better presentation
            display_df = top_skus.copy()
            
            # Select and rename columns for display
            if all(col in display_df.columns for col in ['sku', 'family', 'subfamily', 'total_sales', 'percent_formatted', 'cv', 'abc_xyz_class']):
                display_df = display_df[['sku', 'family', 'subfamily', 'total_sales', 'percent_formatted', 'cv', 'abc_xyz_class']]
                display_df.columns = ['SKU', 'FAMÍLIA', 'SUB-FAMÍLIA', 'VENDAS (KG)', '% TOTAL', 'CV²', 'QUADRANTE']
            
            # Show the table
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )
            
            # Show 'Ver Todos' button if there are more than 5 SKUs
            total_skus = len(st.session_state.abc_xyz_data)
            if total_skus > 5:
                if st.button(f"Ver Todos ({total_skus} SKUs)", key="view_all_button"):
                    # Create an expander to show all SKUs
                    with st.expander("Todos os SKUs Analisados", expanded=True):
                        # Format the full dataframe for display
                        full_df = st.session_state.abc_xyz_data.copy()
                        
                        # Select and rename columns for display
                        if all(col in full_df.columns for col in ['sku', 'family', 'subfamily', 'total_sales', 'percent_of_total', 'cv', 'abc_xyz_class']):
                            full_df['percent_formatted'] = full_df['percent_of_total'].map(lambda x: f"{x*100:.1f}%")
                            full_df['cv_formatted'] = full_df['cv'].map(lambda x: f"{x:.2f}" if x != float('inf') else "∞")
                            
                            display_cols = ['sku', 'family', 'subfamily', 'total_sales', 'percent_formatted', 'cv_formatted', 'abc_xyz_class']
                            full_display_df = full_df[display_cols]
                            full_display_df.columns = ['SKU', 'FAMÍLIA', 'SUB-FAMÍLIA', 'VENDAS (KG)', '% TOTAL', 'CV²', 'QUADRANTE']
                            
                            # Show the full table with pagination
                            st.dataframe(
                                full_display_df,
                                use_container_width=True,
                                hide_index=True,
                            )
        else:
            st.info("Nenhum SKU encontrado com os filtros atuais.")
    
    # Reset button
    if st.session_state.abc_xyz_button_clicked and st.session_state.abc_xyz_data is not None and not st.session_state.abc_xyz_data.empty:
        # Add space before reset button
        st.write("")
        
        # Smaller column to center the button
        _, center_col, _ = st.columns([3, 2, 3])
        with center_col:
            if st.button("🔄 Redefinir Filtros", key="reset_filters"):
                st.session_state.abc_xyz_family = "Todas"
                st.session_state.abc_xyz_subfamily = "Todas"
                st.session_state.abc_xyz_market = "Todos"
                st.session_state.abc_xyz_data = None
                st.session_state.abc_xyz_stats = {}
                st.session_state.abc_xyz_button_clicked = False  # Reset o status do botão
                st.rerun() 
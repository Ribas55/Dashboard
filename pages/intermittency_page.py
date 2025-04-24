"""
Página de Análise de Intermitência com matriz interativa e filtros por família, subfamília e tipo de mercado.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.intermittency_analysis import (
    get_intermittency_data,
    create_intermittency_matrix,
    create_quadrant_chart,
    get_sku_details,
    get_subfamily_options,
    get_quadrant_interpretation
)
from src.mercado_especifico import get_mercado_especifico_skus

def render():
    """Render the Intermittency analysis page."""
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
    
    st.title("Análise de Intermitência")
    
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
    if 'intermittency_family' not in st.session_state:
        st.session_state.intermittency_family = "Todas"
    
    if 'intermittency_subfamily' not in st.session_state:
        st.session_state.intermittency_subfamily = "Todas"
    
    if 'intermittency_market' not in st.session_state:
        st.session_state.intermittency_market = "Todos"
    
    if 'intermittency_data' not in st.session_state:
        st.session_state.intermittency_data = None
    
    if 'intermittency_stats' not in st.session_state:
        st.session_state.intermittency_stats = {}
    
    # Create filter layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Família")
        # All families + Todas option
        family_options = ["Todas"] + sorted(default_families)
        selected_family = st.selectbox(
            "Selecione a família:",
            options=family_options,
            index=family_options.index(st.session_state.intermittency_family),
            key="family_selector"
        )
        
        # Store the selected value
        st.session_state.intermittency_family = selected_family
    
    with col2:
        st.subheader("Subfamília")
        # Get subfamily options based on family selection
        subfamily_options = get_subfamily_options(df_filtered, selected_family)
        
        # Select default index based on current selection or default to "Todas"
        current_subfamily = st.session_state.intermittency_subfamily
        if current_subfamily not in subfamily_options:
            current_subfamily = "Todas"
            st.session_state.intermittency_subfamily = current_subfamily
        
        selected_subfamily = st.selectbox(
            "Selecione a subfamília:",
            options=subfamily_options,
            index=subfamily_options.index(current_subfamily),
            key="subfamily_selector"
        )
        
        # Store the selected value
        st.session_state.intermittency_subfamily = selected_subfamily
    
    with col3:
        st.subheader("Período")
        # For now, we'll always use the last year as the analysis period
        st.selectbox(
            "Período de análise:",
            options=["Último Ano"],
            index=0,
            disabled=True
        )
    
    # Intermittency thresholds row
    st.subheader("Limites CV² / Limite ADI")
    col1, col2 = st.columns(2)
    
    with col1:
        # CV² threshold (vertical line)
        cv2_threshold = st.number_input(
            "Limite CV²:", 
            min_value=0.1, 
            max_value=1.0, 
            value=0.49, 
            step=0.01,
            format="%.2f"
        )
    
    with col2:
        # ADI threshold (horizontal line)
        adi_threshold = st.number_input(
            "Limite ADI:", 
            min_value=1.0, 
            max_value=2.0, 
            value=1.32, 
            step=0.01,
            format="%.2f"
        )
    
    # Market type selector
    market_options = ["Todos", "Normal", "Mercado Específico"]
    market_cols = st.columns([2, 1])
    
    with market_cols[0]:
        st.subheader("SKU:")
        selected_market = st.selectbox(
            "Tipo de mercado:",
            options=market_options,
            index=market_options.index(st.session_state.intermittency_market),
            key="market_selector"
        )
        
        # Store the selected value
        st.session_state.intermittency_market = selected_market
    
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
    if 'intermittency_button_clicked' not in st.session_state:
        st.session_state.intermittency_button_clicked = False
    
    # Process data and show matrix only when the button is clicked
    if view_matrix_button:
        st.session_state.intermittency_button_clicked = True
        with st.spinner("Calculando classificação de intermitência..."):
            # Calculate intermittency classification with filters
            intermittency_data = get_intermittency_data(
                df=df_filtered,
                family_filter=selected_family if selected_family != "Todas" else None,
                subfamily_filter=selected_subfamily if selected_subfamily != "Todas" else None,
                market_type=selected_market,
                mercado_especifico_skus=st.session_state.mercado_especifico_skus,
                cv2_threshold=cv2_threshold,
                adi_threshold=adi_threshold
            )
            
            # Store the data for reuse
            st.session_state.intermittency_data = intermittency_data
            
            # If no data found with the current filters
            if intermittency_data.empty:
                st.warning("Não foram encontrados SKUs com os filtros selecionados.")
                st.session_state.intermittency_button_clicked = False  # Reset para false quando não há dados
                return
    
    # Only proceed if the button was clicked AND we have intermittency data calculated
    if st.session_state.intermittency_button_clicked and st.session_state.intermittency_data is not None and not st.session_state.intermittency_data.empty:
        # Create a title that reflects the current filters
        title_suffix = []
        if selected_family != "Todas":
            title_suffix.append(f"Família: {selected_family}")
        if selected_subfamily != "Todas":
            title_suffix.append(f"Subfamília: {selected_subfamily}")
        if selected_market != "Todos":
            title_suffix.append(f"Mercado: {selected_market}")
        
        matrix_title = "Matriz de Classificação de Intermitência"
        if title_suffix:
            matrix_title += f" ({', '.join(title_suffix)})"
        
        # Create layout for matrix and quadrant chart
        matrix_col, quadrant_col = st.columns([3, 2])
        
        # Show the intermittency matrix
        with matrix_col:
            st.subheader("Matriz de Classificação de Intermitência")
            matrix_fig, matrix_stats = create_intermittency_matrix(
                st.session_state.intermittency_data,
                custom_title=matrix_title
            )
            st.plotly_chart(matrix_fig, use_container_width=True)
            
            # Store matrix stats
            st.session_state.intermittency_stats = matrix_stats
        
        # Show the quadrant chart
        with quadrant_col:
            st.subheader("Detalhes por Quadrante")
            quadrant_fig = create_quadrant_chart(st.session_state.intermittency_data)
            st.plotly_chart(quadrant_fig, use_container_width=True)
        
        # Interpretação dos Quadrantes
        st.subheader("Interpretação")
        
        # Create columns for each category with appropriate styling
        interp_cols = st.columns(2)
        
        with interp_cols[0]:
            # Smooth
            st.markdown(
                f"""
                <div style="background-color:rgba(0,200,0,0.2); padding:10px; border-radius:5px; margin-bottom:10px;">
                <span style="color:#33cc33; font-weight:bold;">■ Smooth:</span> {get_quadrant_interpretation('Smooth')}
                </div>
                
                # Intermittent
                <div style="background-color:rgba(0,0,200,0.2); padding:10px; border-radius:5px;">
                <span style="color:#3333cc; font-weight:bold;">■ Intermittent:</span> {get_quadrant_interpretation('Intermittent')}
                </div>
                """, 
                unsafe_allow_html=True
            )
        
        with interp_cols[1]:
            # Erratic
            st.markdown(
                f"""
                <div style="background-color:rgba(200,200,0,0.2); padding:10px; border-radius:5px; margin-bottom:10px;">
                <span style="color:#cccc00; font-weight:bold;">■ Erratic:</span> {get_quadrant_interpretation('Erratic')}
                </div>
                
                # Lumpy
                <div style="background-color:rgba(200,0,0,0.2); padding:10px; border-radius:5px;">
                <span style="color:#cc3300; font-weight:bold;">■ Lumpy:</span> {get_quadrant_interpretation('Lumpy')}
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # SKUs Analisados
        st.subheader("SKUs Analisados")
        
        # Get top 5 SKUs by sales volume
        top_skus = get_sku_details(st.session_state.intermittency_data, top_n=5)
        
        if not top_skus.empty:
            # Display top SKUs in a table
            # Format columns for better presentation
            display_df = top_skus.copy()
            
            # Select and rename columns for display
            if all(col in display_df.columns for col in ['sku', 'family', 'subfamily', 'total_sales', 'percent_formatted', 'cv2_formatted', 'adi_formatted', 'category']):
                display_df = display_df[['sku', 'family', 'subfamily', 'total_sales', 'percent_formatted', 'cv2_formatted', 'adi_formatted', 'category']]
                display_df.columns = ['SKU', 'FAMÍLIA', 'SUB-FAMÍLIA', 'VENDAS (KG)', '% TOTAL', 'CV²', 'ADI', 'CATEGORIA']
            
            # Show the table
            st.dataframe(
                display_df,
                use_container_width=True,
                hide_index=True,
            )
            
            # Show 'Ver Todos' button if there are more than 5 SKUs
            total_skus = len(st.session_state.intermittency_data)
            if total_skus > 5:
                if st.button(f"Ver Todos ({total_skus} SKUs)", key="view_all_button"):
                    # Create an expander to show all SKUs
                    with st.expander("Todos os SKUs Analisados", expanded=True):
                        # Format the full dataframe for display
                        full_df = st.session_state.intermittency_data.copy()
                        
                        # Ensure we have the formatted columns
                        if 'cv2_formatted' not in full_df.columns:
                            full_df['cv2_formatted'] = full_df['cv2'].apply(lambda x: f"{x:.2f}" if x != float('inf') else "∞")
                        if 'adi_formatted' not in full_df.columns:
                            full_df['adi_formatted'] = full_df['adi'].apply(lambda x: f"{x:.2f}" if x != float('inf') else "∞")
                        if 'percent_formatted' not in full_df.columns:
                            full_df['percent_formatted'] = full_df['percent_of_total'].apply(lambda x: f"{x*100:.1f}%")
                        
                        # Select and rename columns for display
                        display_cols = ['sku', 'family', 'subfamily', 'total_sales', 'percent_formatted', 'cv2_formatted', 'adi_formatted', 'category']
                        full_display_df = full_df[display_cols]
                        full_display_df.columns = ['SKU', 'FAMÍLIA', 'SUB-FAMÍLIA', 'VENDAS (KG)', '% TOTAL', 'CV²', 'ADI', 'CATEGORIA']
                        
                        # Show the full table with pagination
                        st.dataframe(
                            full_display_df,
                            use_container_width=True,
                            hide_index=True,
                        )
        else:
            st.info("Nenhum SKU encontrado com os filtros atuais.")
    
    # Reset button
    if st.session_state.intermittency_button_clicked and st.session_state.intermittency_data is not None and not st.session_state.intermittency_data.empty:
        # Add space before reset button
        st.write("")
        
        # Smaller column to center the button
        _, center_col, _ = st.columns([3, 2, 3])
        with center_col:
            if st.button("🔄 Redefinir Filtros", key="reset_filters"):
                st.session_state.intermittency_family = "Todas"
                st.session_state.intermittency_subfamily = "Todas"
                st.session_state.intermittency_market = "Todos"
                st.session_state.intermittency_data = None
                st.session_state.intermittency_stats = {}
                st.session_state.intermittency_button_clicked = False  # Reset o status do botão
                st.rerun() 
"""
Aggregated series analysis page with flexible aggregation options.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from src.visualizations import create_aggregated_time_series
from src.abc_xyz import filter_active_skus

def render():
    """Render the aggregated series page."""
    st.subheader("Séries Agregadas - Visualização")
    
    if not st.session_state.data_loaded:
        st.info("Por favor, carregue os dados primeiro.")
        return
    
    # Get active SKUs in 2024
    df = st.session_state.sales_data
    
    # Lista das 15 famílias específicas
    specific_families = [
        "Cream Cracker", "Maria", "Wafer", "Sortido", 
        "Cobertas de Chocolate", "Água e Sal", "Digestiva",
        "Recheada", "Circus", "Tartelete", "Torrada",
        "Flocos de Neve", "Integral", "Mentol", "Aliança"
    ]
    
    # Filtrar dados para incluir apenas as 15 famílias específicas
    df = df[df['family'].isin(specific_families)]
    
    # Ensure date column is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['invoice_date']):
        df['invoice_date'] = pd.to_datetime(df['invoice_date'], errors='coerce')
    
    # Filter for active SKUs in 2024
    if 'active_skus_2024' not in st.session_state:
        active_skus = filter_active_skus(df, year=2024)
        st.session_state.active_skus_2024 = active_skus
    
    active_df = df[df['sku'].isin(st.session_state.active_skus_2024)]
    
    # Initialize session state variables if needed
    if 'agg_aggregation_level' not in st.session_state:
        st.session_state.agg_aggregation_level = "Família"
    
    if 'agg_combination_type' not in st.session_state:
        st.session_state.agg_combination_type = "Diferentes Famílias"
    
    if 'agg_selected_entities' not in st.session_state:
        st.session_state.agg_selected_entities = []
    
    if 'agg_abc_selection' not in st.session_state:
        st.session_state.agg_abc_selection = "Todas"
    
    if 'agg_xyz_selection' not in st.session_state:
        st.session_state.agg_xyz_selection = "Todas"
    
    # Inicializar estrutura para armazenar conjuntos de entidades
    if 'agg_entity_sets' not in st.session_state:
        st.session_state.agg_entity_sets = []  # Lista de dicionários com 'name' e 'entities'
    
    # Create first row for primary selections
    col1, col2 = st.columns(2)
    
    with col1:
        # Aggregation level selector
        aggregation_options = ["Família", "Subfamília", "Formato", "Gramagem", "SKU"]
        aggregation_level = st.selectbox(
            "1. Agregar por:",
            options=aggregation_options,
            index=aggregation_options.index(st.session_state.agg_aggregation_level),
            key="agg_aggregation_level"
        )
    
    with col2:
        # Combination type selector
        if st.session_state.agg_aggregation_level == "SKU":
            combination_options = ["Categorias ABC/XYZ"]
        else:
            combination_options = ["Diferentes " + st.session_state.agg_aggregation_level + "s", "Categorias ABC/XYZ"]
        
        # Handle case when previous selection is no longer valid
        current_combination = st.session_state.agg_combination_type
        if current_combination not in combination_options:
            current_combination = combination_options[0]
            st.session_state.agg_combination_type = current_combination
        
        combination_type = st.selectbox(
            "2. Combinações:",
            options=combination_options,
            index=combination_options.index(current_combination),
            key="agg_combination_type"
        )
    
    # Adicionar opção "Tipo de Mercado" apenas quando SKU está selecionado
    if st.session_state.agg_aggregation_level == "SKU":
        # Inicializar a variável de estado se não existir
        if 'agg_market_type' not in st.session_state:
            st.session_state.agg_market_type = "Todos"
        
        # Criar uma nova linha para essa opção
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            # Seletor de tipo de mercado
            market_options = ["Normal", "Mercado Específico", "Todos"]
            market_type = st.selectbox(
                "Tipo de mercado",
                options=market_options,
                index=market_options.index(st.session_state.agg_market_type),
                key="agg_market_type"
            )
    
    # Create divider
    st.divider()
    
    # Show different UI based on combination type selection
    if "Diferentes" in combination_type:
        entity_type = combination_type.split(" ")[1].lower()[:-1]  # Extract entity type (família, subfamília, etc.)
        
        # Get unique values for the selected entity type
        entity_column = entity_type
        if entity_type == "família":
            entity_column = "family"
        elif entity_type == "subfamília":
            entity_column = "subfamily"
        elif entity_type == "formato":
            entity_column = "format"
        elif entity_type == "gramagem":
            entity_column = "weight"
        
        unique_entities = active_df[entity_column].unique().tolist()
        unique_entities.sort()
        
        # Create a multiselect for entities
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Filter to show only entities not already selected
            available_entities = [e for e in unique_entities if e not in st.session_state.agg_selected_entities]
            
            # Entity selector
            new_entity = st.selectbox(
                f"Selecionar {entity_type}",
                options=[""] + available_entities,
                index=0,
                key="new_entity"
            )
        
        with col2:
            st.write("")
            st.write("")
            add_button = st.button("Adicionar", key="add_entity")
            if add_button and new_entity:
                if new_entity not in st.session_state.agg_selected_entities:
                    st.session_state.agg_selected_entities.append(new_entity)
                    st.rerun()
        
        # Show currently selected entities with remove option
        if st.session_state.agg_selected_entities:
            st.write(f"{entity_type.capitalize()}s selecionadas:")
            
            # Create a container for the selected entities
            selected_container = st.container()
            
            # Usar um layout mais compacto para as entidades selecionadas
            cols = selected_container.columns(6)  # Aumentar número de colunas para ficarem mais juntas
            for i, entity in enumerate(st.session_state.agg_selected_entities):
                col_idx = i % 6  # Usar 6 colunas em vez de 3
                with cols[col_idx]:
                    if st.button(f"❌ {entity}", key=f"remove_{i}", use_container_width=True):
                        st.session_state.agg_selected_entities.remove(entity)
                        st.rerun()
            
            # Adicionar opção para definir conjuntos de entidades
            st.write("")
            
            # Apenas mostrar a opção de definir conjunto se houver pelo menos 2 entidades selecionadas
            if len(st.session_state.agg_selected_entities) >= 2:
                col1, col2, col3 = st.columns([1, 2, 1])
                with col2:
                    with st.expander("Definir Conjunto", expanded=False):
                        set_name = st.text_input("Nome do conjunto:", key="set_name_input", 
                                               value=f"Conjunto {len(st.session_state.agg_entity_sets) + 1}")
                        
                        if st.button("Definir Conjunto", key="create_set_button", use_container_width=True):
                            # Adicionar novo conjunto
                            new_set = {
                                'name': set_name,
                                'entities': st.session_state.agg_selected_entities.copy()
                            }
                            st.session_state.agg_entity_sets.append(new_set)
                            
                            # Limpar seleções atuais
                            st.session_state.agg_selected_entities = []
                            st.rerun()
            
            # Adicionar botão de refresh para limpar todas as seleções
            st.write("")
            col1, col2, col3 = st.columns([4, 2, 4])
            with col2:
                if st.button("🔄 Limpar Seleções", key="refresh_selections", use_container_width=True):
                    st.session_state.agg_selected_entities = []
                    st.rerun()
            
            # Mostrar conjuntos definidos (se houver)
            if st.session_state.agg_entity_sets:
                st.divider()
                st.write("Conjuntos definidos:")
                
                for i, entity_set in enumerate(st.session_state.agg_entity_sets):
                    col1, col2 = st.columns([4, 1])
                    with col1:
                        st.write(f"**{entity_set['name']}**: {', '.join(entity_set['entities'])}")
                    with col2:
                        if st.button("❌", key=f"remove_set_{i}", help=f"Remover conjunto {entity_set['name']}"):
                            st.session_state.agg_entity_sets.pop(i)
                            st.rerun()
                
                # Adicionar botão para limpar todos os conjuntos
                st.write("")
                col1, col2, col3 = st.columns([4, 2, 4])
                with col2:
                    if st.button("🔄 Limpar Conjuntos", key="clear_sets", use_container_width=True):
                        st.session_state.agg_entity_sets = []
                        st.rerun()
    
    else:  # Categorias ABC/XYZ selected
        st.write("Selecione as categorias ABC e XYZ para visualizar a agregação de " + 
                st.session_state.agg_aggregation_level.lower() + "s:")
        st.write("*Será mostrada uma linha para cada " + 
                st.session_state.agg_aggregation_level.lower() + 
                " e uma linha para a agregação (total).*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            abc_options = ["A", "B", "C", "AB", "AC", "BC", "ABC", "Todas"]
            abc_selection = st.selectbox(
                "Categoria ABC:",
                options=abc_options,
                index=abc_options.index(st.session_state.agg_abc_selection),
                key="agg_abc_selection",
                help="A = Top 80% vendas, B = Próximos 15%, C = Últimos 5%"
            )
        
        with col2:
            xyz_options = ["X", "Y", "Z", "XY", "XZ", "YZ", "XYZ", "Todas"]
            xyz_selection = st.selectbox(
                "Categoria XYZ:",
                options=xyz_options,
                index=xyz_options.index(st.session_state.agg_xyz_selection),
                key="agg_xyz_selection",
                help="X = Demanda estável (CV ≤ 20%), Y = Variável (CV 20-50%), Z = Alta variabilidade (CV > 50%)"
            )
        
        # Adicionar botão "Realizar Agregação" se todas as opções estiverem selecionadas
        if (st.session_state.agg_abc_selection != "" and 
            st.session_state.agg_xyz_selection != ""):
            
            # Espaçador para separar o botão das opções
            st.write("")
            
            # Centralizar o botão
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                perform_aggregation = st.button(
                    "Realizar Agregação", 
                    key="perform_aggregation_button",
                    use_container_width=True,
                    help="Clique para gerar o gráfico de séries temporais com a linha de agregação e as linhas individuais"
                )
                
                # Armazenar no estado da sessão se a agregação foi solicitada
                if perform_aggregation:
                    st.session_state.agg_perform_analysis = True
                    # Salvar as seleções atuais quando o botão é clicado
                    st.session_state.agg_current_selections = {
                        'abc': st.session_state.agg_abc_selection,
                        'xyz': st.session_state.agg_xyz_selection,
                        'aggregation_level': st.session_state.agg_aggregation_level
                    }
                    
                    # Se for SKU, salvar também o tipo de mercado selecionado
                    if st.session_state.agg_aggregation_level == "SKU" and 'agg_market_type' in st.session_state:
                        st.session_state.agg_current_selections['market_type'] = st.session_state.agg_market_type
                    
                    st.rerun()  # Rerun para aplicar as alterações
        
        # Criar um espaço entre os controles e o possível gráfico
        st.divider()
    
    # Inicializar flag de análise se não existir
    if 'agg_perform_analysis' not in st.session_state:
        st.session_state.agg_perform_analysis = False
    
    # Armazenar as seleções atuais apenas quando o botão é clicado
    if 'agg_current_selections' not in st.session_state:
        st.session_state.agg_current_selections = {
            'abc': 'Todas',
            'xyz': 'Todas',
            'aggregation_level': 'Família'
        }
    
    # Criar visualização apenas se a agregação foi solicitada
    # Ou se temos entidades selecionadas no caso de "Diferentes"
    show_visualization = False
    
    if "Diferentes" in st.session_state.agg_combination_type:
        # Mostrar visualização se tiver entidades selecionadas OU conjuntos definidos
        if st.session_state.agg_selected_entities or st.session_state.agg_entity_sets:
            show_visualization = True
    elif "Categorias" in st.session_state.agg_combination_type and st.session_state.agg_perform_analysis:
        show_visualization = True
    
    if show_visualization:
        st.divider()
        
        # Filter data based on selections
        filtered_data = active_df.copy()
        
        if "Diferentes" in st.session_state.agg_combination_type:
            # Map the entity_type to the actual column name in the dataframe
            column_mapping = {
                "família": "family",
                "subfamília": "subfamily",
                "formato": "format",
                "gramagem": "weight",
                "sku": "sku"
            }
            
            entity_column = column_mapping.get(entity_type, entity_type)
            
            # Obter todas as entidades a serem consideradas (entidades individuais + entidades em conjuntos)
            all_entities = set(st.session_state.agg_selected_entities)
            for entity_set in st.session_state.agg_entity_sets:
                all_entities.update(entity_set['entities'])
            
            # Filtrar dados para incluir todas as entidades necessárias
            filtered_data = filtered_data[filtered_data[entity_column].isin(all_entities)]
            
            # Create the chart title
            chart_title = f"Séries Agregadas por {entity_type.capitalize()}"
            
            # Default aggregation (monthly sum)
            period = "M"  # Monthly
            method = "sum"  # Sum
            
            # Create period column for aggregation
            filtered_data['period'] = filtered_data['invoice_date'].dt.to_period(period)
            
            # Preparar dados para visualização
            # Primeiro, agregar por período e entidade
            agg_data = filtered_data.groupby(['period', entity_column])['sales_value'].agg(method).reset_index()
            agg_data['period'] = agg_data['period'].dt.to_timestamp()
            
            # Criar um DataFrame para armazenar todas as séries (entidades individuais + conjuntos)
            all_series_data = pd.DataFrame(index=agg_data['period'].unique())
            all_series_data.index = pd.DatetimeIndex(all_series_data.index)
            all_series_data = all_series_data.sort_index()
            
            # Adicionar séries para entidades individuais (não incluídas em conjuntos)
            for entity in st.session_state.agg_selected_entities:
                entity_data = agg_data[agg_data[entity_column] == entity]
                if not entity_data.empty:
                    entity_series = entity_data.set_index('period')['sales_value']
                    all_series_data[entity] = entity_series
            
            # Adicionar séries para conjuntos
            for entity_set in st.session_state.agg_entity_sets:
                set_name = entity_set['name']
                set_entities = entity_set['entities']
                
                # Filtrar dados apenas para entidades do conjunto
                set_data = agg_data[agg_data[entity_column].isin(set_entities)]
                
                # Agregar por período (somar todas as entidades do conjunto)
                set_aggregated = set_data.groupby('period')['sales_value'].sum()
                
                # Adicionar ao DataFrame de séries
                all_series_data[set_name] = set_aggregated
            
            # Create the plot
            fig = go.Figure()
            
            # Usar uma checkbox para controlar a visibilidade da linha de agregação total
            st.write("")
            col1, col2, col3 = st.columns([4, 2, 4])
            with col2:
                show_aggregation = st.checkbox("Mostrar Soma Agregada Total", value=False, key="show_aggregation")
                
                # Adicionar opção para normalizar os dados (escala 0-1)
                normalize_data = st.checkbox("Normalizar Dados (0-1)", value=False, key="normalize_data", 
                                           help="Normaliza os valores de cada série para uma escala de 0 a 1")
            
            # Normalizar os dados se a opção estiver marcada
            if normalize_data:
                # Criar uma cópia para não modificar os dados originais
                normalized_data = all_series_data.copy()
                
                # Normalizar cada série individualmente (0-1)
                for col in normalized_data.columns:
                    if normalized_data[col].notna().any():  # Verificar se há dados não-nulos
                        min_val = normalized_data[col].min()
                        max_val = normalized_data[col].max()
                        if max_val > min_val:  # Evitar divisão por zero
                            normalized_data[col] = (normalized_data[col] - min_val) / (max_val - min_val)
                
                # Usar os dados normalizados
                plot_data = normalized_data
                y_axis_title = "Valor Normalizado (0-1)"
                chart_title += " - Normalizado"
            else:
                # Usar os dados originais
                plot_data = all_series_data
                y_axis_title = "Valor de Vendas"
            
            # Adicionar opção para esconder/mostrar a linha de agregação de todas as séries
            total_data = plot_data.sum(axis=1)
            
            # Adicionar linha de soma agregada total apenas se show_aggregation for True
            if show_aggregation:
                fig.add_trace(go.Scatter(
                    x=plot_data.index,
                    y=total_data,
                    name=f"SOMA AGREGADA TOTAL",
                    mode='lines',
                    line=dict(color='#ffffff', width=3, dash='solid'),
                ))
            
            # Adicionar linhas para cada entidade individual e conjunto
            for col in plot_data.columns:
                # Verificar se esta coluna é um conjunto
                is_set = any(entity_set['name'] == col for entity_set in st.session_state.agg_entity_sets)
                
                line_width = 2.5 if is_set else 1.5  # Conjuntos têm linhas mais grossas
                dash_type = 'solid' if is_set else None  # Conjuntos têm linhas sólidas
                marker_size = 8 if is_set else 6  # Conjuntos têm marcadores maiores
                
                # Criar série temporal
                fig.add_trace(go.Scatter(
                    x=plot_data.index,
                    y=plot_data[col],
                    name=str(col),
                    mode='lines+markers',
                    line=dict(width=line_width, dash=dash_type),
                    marker=dict(size=marker_size)
                ))
            
            fig.update_layout(
                title=chart_title,
                xaxis_title="Período",
                yaxis_title=y_axis_title,
                height=500,
                template="plotly_dark",
                plot_bgcolor='rgba(25, 25, 35, 1)',
                paper_bgcolor='rgba(25, 25, 35, 1)',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Adicionar range slider para melhor navegação temporal
            fig.update_xaxes(
                rangeslider=dict(
                    visible=True,
                    bgcolor='rgba(50, 50, 70, 0.7)',
                    bordercolor='rgba(70, 70, 90, 1)',
                    thickness=0.08
                ),
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all", label="Todos")
                    ]),
                    bgcolor='rgba(35, 35, 50, 1)',
                    activecolor='rgba(70, 70, 120, 1)'
                )
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Add statistics section
            st.subheader("Estatísticas")
            
            # Calculate statistics for all data (incluindo conjuntos e entidades individuais)
            total_by_series = {}
            for col in all_series_data.columns:
                total_by_series[col] = all_series_data[col].sum()
            
            total_value = sum(total_by_series.values())
            avg_value = total_value / len(all_series_data.index) if not all_series_data.empty else 0
            
            # Find entity/set with highest total sales
            if total_by_series:
                max_entity = max(total_by_series.items(), key=lambda x: x[1])[0]
                max_value = total_by_series[max_entity]
            else:
                max_entity = "N/A"
                max_value = 0
                
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total de Vendas", f"{total_value:.2f}")
            
            with col2:
                st.metric("Média por Período", f"{avg_value:.2f}")
            
            with col3:
                st.metric(f"Maior Venda", f"{max_entity} ({max_value:.2f})")
            
            # Add the new tabs for ABC/XYZ and Intermittency analysis
            st.divider()
            st.subheader("Análises Adicionais")
            
            # Create tabs for additional analyses
            abc_xyz_tab, intermittency_tab, percent_dist_tab = st.tabs(["Matriz ABC/XYZ", "Intermitência", "Distribuição Percentual"])
            
            # Initialize session state variables for the analyses if not exist
            if 'agg_analysis_entity_type' not in st.session_state:
                st.session_state.agg_analysis_entity_type = "Família"
            
            if 'agg_analysis_entity_selection' not in st.session_state:
                st.session_state.agg_analysis_entity_selection = "Selecionadas"
            
            if 'agg_analysis_abc_xyz_clicked' not in st.session_state:
                st.session_state.agg_analysis_abc_xyz_clicked = False
                
            if 'agg_analysis_intermittency_clicked' not in st.session_state:
                st.session_state.agg_analysis_intermittency_clicked = False

            if 'agg_analysis_percent_dist_clicked' not in st.session_state:
                st.session_state.agg_analysis_percent_dist_clicked = False
            
            # ABC/XYZ Matrix Tab
            with abc_xyz_tab:
                st.write("Selecione os parâmetros para visualizar a matriz ABC/XYZ")
                
                # Simplified interface - only show entity selection options
                col1, col2 = st.columns(2)
                
                with col1:
                    # Entity selection method - only show this dropdown
                    entity_selection_options = ["Total das Seleções Atuais", "Conjunto", "Individual"]
                    analysis_entity_selection = st.selectbox(
                        "Analisar com base em:",
                        options=entity_selection_options,
                        index=0,
                        key="abc_xyz_entity_selection"
                    )
                
                # Add second column for selecting specific entities or sets
                with col2:
                    # Based on the entity selection, show the appropriate options
                    if analysis_entity_selection == "Conjunto":
                        if st.session_state.agg_entity_sets:
                            set_names = [s['name'] for s in st.session_state.agg_entity_sets]
                            selected_set = st.selectbox(
                                "Selecione o conjunto:",
                                options=set_names,
                                key="abc_xyz_selected_set"
                            )
                            # Get the entities in the selected set
                            selected_set_entities = next((s['entities'] for s in st.session_state.agg_entity_sets if s['name'] == selected_set), [])
                        else:
                            st.warning("Nenhum conjunto definido. Por favor, defina conjuntos na seção acima.")
                            selected_set_entities = []
                    
                    elif analysis_entity_selection == "Individual":
                        if st.session_state.agg_selected_entities:
                            selected_entity = st.selectbox(
                                "Selecione a entidade:",
                                options=st.session_state.agg_selected_entities,
                                key="abc_xyz_selected_entity"
                            )
                            # Create a list with just the selected entity
                            selected_entities = [selected_entity]
                        else:
                            st.warning("Nenhuma entidade selecionada individualmente.")
                            selected_entities = []
                
                # ABC/XYZ thresholds
                st.write("Limites ABC/XYZ:")
                thresholds_col1, thresholds_col2 = st.columns(2)
                
                with thresholds_col1:
                    # ABC thresholds
                    abc_col1, abc_col2, abc_col3 = st.columns([1, 0.2, 1])
                    with abc_col1:
                        a_threshold = st.number_input("", min_value=50, max_value=90, value=80, step=5, label_visibility="collapsed", key="abc_threshold_a")
                    with abc_col2:
                        st.write("-")
                    with abc_col3:
                        b_threshold = st.number_input("", min_value=90, max_value=99, value=95, step=1, label_visibility="collapsed", key="abc_threshold_b")
                    
                    st.caption("Limites ABC (%)")
                
                with thresholds_col2:
                    # XYZ thresholds
                    xyz_col1, xyz_col2, xyz_col3 = st.columns([1, 0.2, 1])
                    with xyz_col1:
                        x_threshold = st.number_input("", min_value=10, max_value=30, value=20, step=5, label_visibility="collapsed", key="xyz_threshold_x")
                    with xyz_col2:
                        st.write("-")
                    with xyz_col3:
                        y_threshold = st.number_input("", min_value=30, max_value=70, value=50, step=5, label_visibility="collapsed", key="xyz_threshold_y")
                    
                    st.caption("Limites XYZ (%)")
                
                # Button to view the matrix
                view_button_col1, view_button_col2, view_button_col3 = st.columns([1, 2, 1])
                with view_button_col2:
                    view_abc_xyz_matrix = st.button(
                        "📊 Ver Matriz ABC/XYZ",
                        type="primary",
                        use_container_width=True,
                        key="view_abc_xyz_matrix"
                    )
                    
                    if view_abc_xyz_matrix:
                        st.session_state.agg_analysis_abc_xyz_clicked = True
                
                # Display the ABC/XYZ matrix if button was clicked
                if st.session_state.agg_analysis_abc_xyz_clicked:
                    with st.spinner("Gerando matriz ABC/XYZ..."):
                        # Get the entity column based on the entity type from the first part of the page
                        entity_column = column_mapping.get(entity_type, entity_type)
                        
                        # Determine which entities to filter by based on selection
                        if analysis_entity_selection == "Total das Seleções Atuais":
                            # Use all currently selected entities (individual and in sets)
                            all_selected_entities = set(st.session_state.agg_selected_entities)
                            for entity_set in st.session_state.agg_entity_sets:
                                all_selected_entities.update(entity_set['entities'])
                            target_entities = list(all_selected_entities)
                        elif analysis_entity_selection == "Conjunto" and 'selected_set_entities' in locals():
                            # Use only entities from the selected set
                            target_entities = selected_set_entities
                        elif analysis_entity_selection == "Individual" and 'selected_entities' in locals():
                            # Use only the single selected entity
                            target_entities = selected_entities
                        
                        # Filter data for SKUs associated with the selected entities
                        if target_entities:
                            # First, filter by the entity type (família, subfamília, etc.)
                            entity_filtered_data = filtered_data[filtered_data[entity_column].isin(target_entities)]
                            
                            # Get all SKUs associated with these entities
                            relevant_skus = entity_filtered_data['sku'].unique().tolist()
                            
                            # Filter the full dataset to include only these SKUs
                            abc_xyz_filtered_data = active_df[active_df['sku'].isin(relevant_skus)]
                            
                            # Get required modules 
                            from src.abc_xyz_analysis import create_abc_xyz_matrix, get_abc_xyz_data
                            
                            # Always perform ABC/XYZ classification at SKU level
                            abc_xyz_data = get_abc_xyz_data(
                                abc_xyz_filtered_data,
                                family_filter=None,
                                subfamily_filter=None,
                                market_type="Todos",
                                mercado_especifico_skus=st.session_state.mercado_especifico_skus if 'mercado_especifico_skus' in st.session_state else None,
                                a_threshold=a_threshold/100,
                                b_threshold=b_threshold/100,
                                x_threshold=x_threshold/100,
                                y_threshold=y_threshold/100,
                                year=2024
                            )
                            
                            # Create the matrix
                            if not abc_xyz_data.empty:
                                # Create title based on selection
                                if analysis_entity_selection == "Total das Seleções Atuais":
                                    matrix_title = "Matriz ABC/XYZ - Total das Seleções Atuais"
                                elif analysis_entity_selection == "Conjunto":
                                    matrix_title = f"Matriz ABC/XYZ - Conjunto: {selected_set}"
                                elif analysis_entity_selection == "Individual":
                                    matrix_title = f"Matriz ABC/XYZ - {entity_type.capitalize()}: {selected_entities[0]}"
                                
                                # Create and display the matrix
                                matrix_fig, matrix_stats = create_abc_xyz_matrix(
                                    abc_xyz_data,
                                    custom_title=matrix_title
                                )
                                st.plotly_chart(matrix_fig, use_container_width=True)
                                
                                # Show statistics
                                st.subheader("Estatísticas por Quadrante")
                                
                                # Create a clean table with the statistics
                                # Fix for proper handling of matrix_stats which is a dict of dicts
                                quadrants = []
                                counts = []
                                
                                for quadrant, stats in matrix_stats.items():
                                    quadrants.append(quadrant)
                                    # Check if 'count' is available in stats (which is a dict)
                                    if isinstance(stats, dict) and 'count' in stats:
                                        counts.append(stats['count'])
                                    else:
                                        # Fallback if the structure is different
                                        counts.append(stats if isinstance(stats, (int, float)) else 0)
                                
                                stats_df = pd.DataFrame({
                                    "Quadrante": quadrants,
                                    "Quantidade": counts
                                })
                                
                                # Add percentage column
                                total_entities = sum(counts)
                                stats_df["Percentual"] = stats_df["Quantidade"].apply(
                                    lambda x: f"{x/total_entities*100:.1f}%" if total_entities > 0 else "0%"
                                )
                                
                                # Display the statistics
                                st.dataframe(
                                    stats_df,
                                    use_container_width=True,
                                    hide_index=True
                                )
                                
                                # Display top SKUs
                                st.subheader("SKUs por Quadrante")
                                
                                # Get top SKUs for a quick view
                                if not abc_xyz_data.empty:
                                    # Make sure we have all needed columns
                                    if all(col in abc_xyz_data.columns for col in ['sku', 'family', 'subfamily']):
                                        # Top 5 SKUs by sales value
                                        top_skus = abc_xyz_data.sort_values('total_sales', ascending=False).head(5)
                                        
                                        # Create display dataframe with essential columns
                                        display_df = top_skus[['sku', 'family', 'subfamily', 'total_sales', 'abc_xyz_class']]
                                        
                                        # Format percentage
                                        if 'percent_of_total' in top_skus.columns:
                                            display_df['percent'] = top_skus['percent_of_total'].apply(lambda x: f"{x*100:.2f}%")
                                        else:
                                            # Calculate percentage if not available
                                            total_sales = abc_xyz_data['total_sales'].sum()
                                            display_df['percent'] = display_df['total_sales'].apply(lambda x: f"{x/total_sales*100:.2f}%" if total_sales > 0 else "0%")
                                        
                                        # Rename columns for display
                                        display_df.columns = ['SKU', 'Família', 'Subfamília', 'Vendas', 'Quadrante', '% Total']
                                        
                                        # Show the dataframe
                                        st.dataframe(
                                            display_df,
                                            use_container_width=True,
                                            hide_index=True
                                        )
                                        
                                        # Add "View All SKUs" button
                                        if len(abc_xyz_data) > 5:
                                            st.text(f"Exibindo 5 de {len(abc_xyz_data)} SKUs")
                                            if st.button("Ver Todos os SKUs", key="view_all_abc_xyz_skus"):
                                                st.subheader(f"Todos os SKUs ({len(abc_xyz_data)})")
                                                
                                                # Create complete dataframe
                                                full_df = abc_xyz_data[['sku', 'family', 'subfamily', 'total_sales', 'abc_xyz_class']]
                                                
                                                # Format percentage
                                                if 'percent_of_total' in abc_xyz_data.columns:
                                                    full_df['percent'] = abc_xyz_data['percent_of_total'].apply(lambda x: f"{x*100:.2f}%")
                                                else:
                                                    # Calculate percentage if not available
                                                    total_sales = abc_xyz_data['total_sales'].sum()
                                                    full_df['percent'] = full_df['total_sales'].apply(lambda x: f"{x/total_sales*100:.2f}%" if total_sales > 0 else "0%")
                                                
                                                # Rename columns for display
                                                full_df.columns = ['SKU', 'Família', 'Subfamília', 'Vendas', 'Quadrante', '% Total']
                                                
                                                # Show the full dataframe
                                                st.dataframe(
                                                    full_df,
                                                    use_container_width=True,
                                                    hide_index=True
                                                )
                                    else:
                                        st.warning("Informações de SKU, família ou subfamília não estão disponíveis nos dados.")
                                else:
                                    st.info("Não há dados de SKU para exibir.")
                            else:
                                st.warning("Não foi possível gerar a matriz ABC/XYZ. Nenhum dado encontrado com os filtros atuais.")
                        else:
                            st.warning("Nenhuma entidade selecionada para análise.")
            
            # Intermittency Analysis Tab
            with intermittency_tab:
                st.write("Selecione os parâmetros para visualizar a análise de intermitência")
                
                # Simplified interface - only show entity selection options
                col1, col2 = st.columns(2)
                
                with col1:
                    # Entity selection method - only show this dropdown
                    entity_selection_options = ["Total das Seleções Atuais", "Conjunto", "Individual"]
                    intermittency_entity_selection = st.selectbox(
                        "Analisar com base em:",
                        options=entity_selection_options,
                        index=0,
                        key="intermittency_entity_selection"
                    )
                
                # Add second column for selecting specific entities or sets
                with col2:
                    # Based on the entity selection, show the appropriate options
                    if intermittency_entity_selection == "Conjunto":
                        if st.session_state.agg_entity_sets:
                            set_names = [s['name'] for s in st.session_state.agg_entity_sets]
                            intermittency_selected_set = st.selectbox(
                                "Selecione o conjunto:",
                                options=set_names,
                                key="intermittency_selected_set"
                            )
                            # Get the entities in the selected set
                            intermittency_set_entities = next((s['entities'] for s in st.session_state.agg_entity_sets if s['name'] == intermittency_selected_set), [])
                        else:
                            st.warning("Nenhum conjunto definido. Por favor, defina conjuntos na seção acima.")
                            intermittency_set_entities = []
                    
                    elif intermittency_entity_selection == "Individual":
                        if st.session_state.agg_selected_entities:
                            intermittency_selected_entity = st.selectbox(
                                "Selecione a entidade:",
                                options=st.session_state.agg_selected_entities,
                                key="intermittency_selected_entity"
                            )
                            # Create a list with just the selected entity
                            intermittency_selected_entities = [intermittency_selected_entity]
                        else:
                            st.warning("Nenhuma entidade selecionada individualmente.")
                            intermittency_selected_entities = []
                
                # Intermittency thresholds
                st.write("Limites de Intermitência:")
                thresholds_col1, thresholds_col2 = st.columns(2)
                
                with thresholds_col1:
                    # CV² threshold (vertical line)
                    cv2_threshold = st.number_input(
                        "Limite CV²:", 
                        min_value=0.1, 
                        max_value=1.0, 
                        value=0.49, 
                        step=0.01,
                        format="%.2f",
                        key="intermittency_cv2_threshold"
                    )
                
                with thresholds_col2:
                    # ADI threshold (horizontal line)
                    adi_threshold = st.number_input(
                        "Limite ADI:", 
                        min_value=1.0, 
                        max_value=2.0, 
                        value=1.32, 
                        step=0.01,
                        format="%.2f",
                        key="intermittency_adi_threshold"
                    )
                
                # Button to view the intermittency analysis
                view_button_col1, view_button_col2, view_button_col3 = st.columns([1, 2, 1])
                with view_button_col2:
                    view_intermittency = st.button(
                        "📊 Ver Análise de Intermitência",
                        type="primary",
                        use_container_width=True,
                        key="view_intermittency"
                    )
                    
                    if view_intermittency:
                        st.session_state.agg_analysis_intermittency_clicked = True
                
                # Display the intermittency analysis if button was clicked
                if st.session_state.agg_analysis_intermittency_clicked:
                    with st.spinner("Gerando análise de intermitência..."):
                        # Get the entity column based on the entity type from the first part of the page
                        entity_column = column_mapping.get(entity_type, entity_type)
                        
                        # Determine which entities to filter by based on selection
                        if intermittency_entity_selection == "Total das Seleções Atuais":
                            # Use all currently selected entities (individual and in sets)
                            all_selected_entities = set(st.session_state.agg_selected_entities)
                            for entity_set in st.session_state.agg_entity_sets:
                                all_selected_entities.update(entity_set['entities'])
                            target_entities = list(all_selected_entities)
                        elif intermittency_entity_selection == "Conjunto" and 'intermittency_set_entities' in locals():
                            # Use only entities from the selected set
                            target_entities = intermittency_set_entities
                        elif intermittency_entity_selection == "Individual" and 'intermittency_selected_entities' in locals():
                            # Use only the single selected entity
                            target_entities = intermittency_selected_entities
                        
                        # Filter data for SKUs associated with the selected entities
                        if target_entities:
                            # First, filter by the entity type (família, subfamília, etc.)
                            entity_filtered_data = filtered_data[filtered_data[entity_column].isin(target_entities)]
                            
                            # Get all SKUs associated with these entities
                            relevant_skus = entity_filtered_data['sku'].unique().tolist()
                            
                            # Filter the full dataset to include only these SKUs
                            intermittency_filtered_data = active_df[active_df['sku'].isin(relevant_skus)]
                            
                            # Get required modules
                            from src.intermittency_analysis import get_intermittency_data, create_intermittency_matrix, create_quadrant_chart
                            
                            # Check if we need to handle mercado_especifico_skus
                            if 'mercado_especifico_skus' not in st.session_state:
                                mercado_especifico_skus = []
                            else:
                                mercado_especifico_skus = st.session_state.mercado_especifico_skus
                            
                            # Perform intermittency analysis
                            intermittency_data = get_intermittency_data(
                                df=intermittency_filtered_data,
                                family_filter=None,  # No additional filtering needed as data is already filtered
                                subfamily_filter=None,
                                market_type="Todos",  # No market filtering in this context
                                mercado_especifico_skus=mercado_especifico_skus,
                                cv2_threshold=cv2_threshold,
                                adi_threshold=adi_threshold
                            )
                            
                            # Create the matrix and charts
                            if not intermittency_data.empty:
                                # Create title based on selection
                                if intermittency_entity_selection == "Total das Seleções Atuais":
                                    matrix_title = "Análise de Intermitência - Total das Seleções Atuais"
                                elif intermittency_entity_selection == "Conjunto":
                                    matrix_title = f"Análise de Intermitência - Conjunto: {intermittency_selected_set}"
                                elif intermittency_entity_selection == "Individual":
                                    matrix_title = f"Análise de Intermitência - {entity_type.capitalize()}: {intermittency_selected_entities[0]}"
                                
                                # Create layout for matrix and quadrant chart
                                matrix_col, quadrant_col = st.columns([3, 2])
                                
                                # Create and display the matrix
                                with matrix_col:
                                    matrix_fig, matrix_stats = create_intermittency_matrix(
                                        intermittency_data,
                                        custom_title=matrix_title
                                    )
                                    st.plotly_chart(matrix_fig, use_container_width=True)
                                    
                                    # Add category counts after the matrix
                                    st.subheader("Distribuição por Categoria")
                                    
                                    # Create a clean table with the statistics
                                    categories = []
                                    counts = []
                                    percentages = []
                                    
                                    for category, stats in matrix_stats.items():
                                        categories.append(category)
                                        if isinstance(stats, dict) and 'count' in stats:
                                            counts.append(stats['count'])
                                            percentages.append(stats.get('percent', 0) * 100)
                                        else:
                                            counts.append(stats if isinstance(stats, (int, float)) else 0)
                                            percentages.append(0)
                                    
                                    stats_df = pd.DataFrame({
                                        "Categoria": categories,
                                        "Quantidade de SKUs": counts,
                                        "% de Vendas": [f"{p:.1f}%" for p in percentages]
                                    })
                                    
                                    # Display the statistics
                                    st.dataframe(
                                        stats_df,
                                        use_container_width=True,
                                        hide_index=True
                                    )
                                
                                # Create and display the quadrant chart
                                with quadrant_col:
                                    quadrant_fig = create_quadrant_chart(intermittency_data)
                                    st.plotly_chart(quadrant_fig, use_container_width=True)
                                    
                                    # Interpretation of quadrants - Moved to the right column
                                    st.subheader("Interpretação dos Quadrantes")
                                    
                                    # Get the interpretation function
                                    from src.intermittency_analysis import get_quadrant_interpretation
                                    
                                    # Display interpretation directly in the right column
                                    st.markdown(
                                        f"""
                                        <div style="background-color:rgba(0,200,0,0.2); padding:10px; border-radius:5px; margin-bottom:10px;">
                                        <span style="color:#33cc33; font-weight:bold;">■ Smooth:</span> {get_quadrant_interpretation('Smooth')}
                                        </div>
                                        
                                        <div style="background-color:rgba(0,0,200,0.2); padding:10px; border-radius:5px; margin-bottom:10px;">
                                        <span style="color:#3333cc; font-weight:bold;">■ Intermittent:</span> {get_quadrant_interpretation('Intermittent')}
                                        </div>
                                        
                                        <div style="background-color:rgba(200,200,0,0.2); padding:10px; border-radius:5px; margin-bottom:10px;">
                                        <span style="color:#cccc00; font-weight:bold;">■ Erratic:</span> {get_quadrant_interpretation('Erratic')}
                                        </div>
                                        
                                        <div style="background-color:rgba(200,0,0,0.2); padding:10px; border-radius:5px;">
                                        <span style="color:#cc3300; font-weight:bold;">■ Lumpy:</span> {get_quadrant_interpretation('Lumpy')}
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )
                                
                                # Display top SKUs by category
                                st.subheader("SKUs por Categoria de Intermitência")
                                
                                # Get top SKUs for a quick view
                                if not intermittency_data.empty:
                                    # Make sure we have all needed columns
                                    if all(col in intermittency_data.columns for col in ['sku', 'family', 'subfamily']):
                                        # Top 5 SKUs by sales value
                                        top_skus = intermittency_data.sort_values('total_sales', ascending=False).head(5)
                                        
                                        # Create display dataframe with essential columns
                                        display_df = top_skus[['sku', 'family', 'subfamily', 'total_sales', 'category', 'cv2', 'adi']]
                                        
                                        # Format values
                                        display_df['cv2'] = display_df['cv2'].apply(lambda x: f"{x:.2f}")
                                        display_df['adi'] = display_df['adi'].apply(lambda x: f"{x:.2f}")
                                        
                                        # Format percentage
                                        total_sales = intermittency_data['total_sales'].sum()
                                        display_df['percent'] = display_df['total_sales'].apply(lambda x: f"{x/total_sales*100:.2f}%" if total_sales > 0 else "0%")
                                        
                                        # Rename columns for display
                                        display_df.columns = ['SKU', 'Família', 'Subfamília', 'Vendas', 'Categoria', 'CV²', 'ADI', '% Total']
                                        
                                        # Show the dataframe
                                        st.dataframe(
                                            display_df,
                                            use_container_width=True,
                                            hide_index=True
                                        )
                                        
                                        # Add "View All SKUs" button
                                        if len(intermittency_data) > 5:
                                            st.text(f"Exibindo 5 de {len(intermittency_data)} SKUs")
                                            if st.button("Ver Todos os SKUs", key="view_all_intermittency_skus"):
                                                st.subheader(f"Todos os SKUs ({len(intermittency_data)})")
                                                
                                                # Create complete dataframe
                                                full_df = intermittency_data[['sku', 'family', 'subfamily', 'total_sales', 'category', 'cv2', 'adi']]
                                                
                                                # Format values
                                                full_df['cv2'] = full_df['cv2'].apply(lambda x: f"{x:.2f}")
                                                full_df['adi'] = full_df['adi'].apply(lambda x: f"{x:.2f}")
                                                
                                                # Format percentage
                                                full_df['percent'] = full_df['total_sales'].apply(lambda x: f"{x/total_sales*100:.2f}%" if total_sales > 0 else "0%")
                                                
                                                # Rename columns for display
                                                full_df.columns = ['SKU', 'Família', 'Subfamília', 'Vendas', 'Categoria', 'CV²', 'ADI', '% Total']
                                                
                                                # Show the full dataframe
                                                st.dataframe(
                                                    full_df,
                                                    use_container_width=True,
                                                    hide_index=True
                                                )
                                    else:
                                        st.warning("Informações de SKU, família ou subfamília não estão disponíveis nos dados.")
                                else:
                                    st.info("Não há dados de SKU para exibir.")
                            else:
                                st.warning("Não foi possível gerar a análise de intermitência. Nenhum dado encontrado com os filtros atuais.")
                        else:
                            st.warning("Nenhuma entidade selecionada para análise.")
            
            # Percentage Distribution Tab
            with percent_dist_tab:
                st.write("Selecione os parâmetros para visualizar a evolução da distribuição percentual")
                
                # Simplified interface - only show entity selection options
                col1, col2 = st.columns(2)
                
                with col1:
                    # Entity selection method
                    entity_selection_options = ["Total das Seleções Atuais", "Conjunto", "Individual"]
                    percent_dist_entity_selection = st.selectbox(
                        "Analisar com base em:",
                        options=entity_selection_options,
                        index=0,
                        key="percent_dist_entity_selection"
                    )
                
                # Based on the entity selection, show appropriate options in second column
                with col2:
                    if percent_dist_entity_selection == "Conjunto":
                        if st.session_state.agg_entity_sets:
                            set_names = [s['name'] for s in st.session_state.agg_entity_sets]
                            selected_percent_dist_set = st.selectbox(
                                "Selecione o conjunto:",
                                options=set_names,
                                key="percent_dist_selected_set"
                            )
                            # Get the entities in the selected set
                            selected_percent_dist_entities = next((s['entities'] for s in st.session_state.agg_entity_sets if s['name'] == selected_percent_dist_set), [])
                        else:
                            st.warning("Nenhum conjunto definido. Por favor, defina conjuntos na seção acima.")
                            selected_percent_dist_entities = []
                    
                    elif percent_dist_entity_selection == "Individual":
                        if st.session_state.agg_selected_entities:
                            selected_percent_dist_entity = st.selectbox(
                                "Selecione a entidade:",
                                options=st.session_state.agg_selected_entities,
                                key="percent_dist_selected_entity"
                            )
                            # Create a list with just the selected entity
                            selected_percent_dist_entities = [selected_percent_dist_entity]
                        else:
                            st.warning("Nenhuma entidade selecionada individualmente.")
                            selected_percent_dist_entities = []
                
                # Distribution options based on current aggregation level
                st.write("Opções de distribuição:")
                dist_col1, dist_col2 = st.columns(2)
                
                with dist_col1:
                    # Distribution by options based on entity type
                    if entity_type == "família":
                        dist_by_options = ["Sub Família", "SKU"]
                    elif entity_type == "subfamília":
                        dist_by_options = ["SKU"]
                    else:
                        dist_by_options = ["SKU"]  # Default to SKU for other entity types
                    
                    dist_by = st.selectbox(
                        "Distribuição por:",
                        options=dist_by_options,
                        key="percent_dist_by"
                    )
                
                # Add SKU filter options if distribution by SKU is selected
                if dist_by == "SKU":
                    with dist_col2:
                        sku_filter_options = ["Todos SKUs", "SKUs-Normal", "SKUs-Específico", "Normal/Específico"]
                        sku_filter = st.selectbox(
                            "Filtrar SKUs:",
                            options=sku_filter_options,
                            key="percent_dist_sku_filter"
                        )
                
                # Button to view the distribution
                view_button_col1, view_button_col2, view_button_col3 = st.columns([1, 2, 1])
                with view_button_col2:
                    view_percent_dist = st.button(
                        "📊 Ver Distribuição Percentual",
                        type="primary",
                        use_container_width=True,
                        key="view_percent_dist_button"
                    )
                    
                    if view_percent_dist:
                        st.session_state.agg_analysis_percent_dist_clicked = True
                
                # Display the percentage distribution visualization if button was clicked
                if st.session_state.agg_analysis_percent_dist_clicked:
                    with st.spinner("Gerando visualização de distribuição percentual..."):
                        # Get the entity column based on the entity type
                        entity_column = column_mapping.get(entity_type, entity_type)
                        
                        # Determine which entities to filter by based on selection
                        if percent_dist_entity_selection == "Total das Seleções Atuais":
                            # Use all currently selected entities (individual and in sets)
                            all_selected_entities = set(st.session_state.agg_selected_entities)
                            for entity_set in st.session_state.agg_entity_sets:
                                all_selected_entities.update(entity_set['entities'])
                            target_percent_dist_entities = list(all_selected_entities)
                        elif percent_dist_entity_selection == "Conjunto" and 'selected_percent_dist_entities' in locals():
                            # Use only entities from the selected set
                            target_percent_dist_entities = selected_percent_dist_entities
                        elif percent_dist_entity_selection == "Individual" and 'selected_percent_dist_entities' in locals():
                            # Use only the single selected entity
                            target_percent_dist_entities = selected_percent_dist_entities
                        else:
                            target_percent_dist_entities = []
                        
                        # Filter data for entities associated with the selected entities
                        if target_percent_dist_entities:
                            # Filter by the entity type (família, subfamília, etc.)
                            percent_dist_filtered_data = filtered_data[filtered_data[entity_column].isin(target_percent_dist_entities)]
                            
                            # Determine the distribution column based on selection
                            if dist_by == "Sub Família":
                                distribution_column = "subfamily"
                            else:  # Default to SKU
                                distribution_column = "sku"
                            
                            # Apply additional filtering for SKUs if needed
                            if dist_by == "SKU" and 'sku_filter' in locals():
                                # Check if we have the mercado_especifico_skus loaded
                                if 'mercado_especifico_skus' in st.session_state:
                                    mercado_especifico_skus = set(str(sku) for sku in st.session_state.mercado_especifico_skus)
                                    
                                    if sku_filter == "SKUs-Normal":
                                        # Keep only normal market SKUs
                                        percent_dist_filtered_data = percent_dist_filtered_data[
                                            ~percent_dist_filtered_data["sku"].astype(str).isin(mercado_especifico_skus)
                                        ]
                                    elif sku_filter == "SKUs-Específico":
                                        # Keep only special market SKUs
                                        percent_dist_filtered_data = percent_dist_filtered_data[
                                            percent_dist_filtered_data["sku"].astype(str).isin(mercado_especifico_skus)
                                        ]
                                    elif sku_filter == "Normal/Específico":
                                        # Will handle this special case later - need to create a new column for this grouping
                                        percent_dist_filtered_data["market_type"] = "Normal"
                                        percent_dist_filtered_data.loc[
                                            percent_dist_filtered_data["sku"].astype(str).isin(mercado_especifico_skus), 
                                            "market_type"
                                        ] = "Específico"
                                        distribution_column = "market_type"
                                else:
                                    st.warning("Dados de mercado específico não disponíveis. Por favor, carregue os dados novamente.")
                            
                            # Create a period column for aggregation (monthly)
                            percent_dist_filtered_data['period'] = percent_dist_filtered_data['invoice_date'].dt.to_period('M')
                            
                            # Aggregate data by period and distribution entity
                            period_entity_sales = percent_dist_filtered_data.groupby(['period', distribution_column])['sales_value'].sum().reset_index()
                            
                            # Calculate total sales by period
                            period_total_sales = period_entity_sales.groupby('period')['sales_value'].sum().reset_index()
                            period_total_sales.rename(columns={'sales_value': 'total_sales'}, inplace=True)
                            
                            # Merge totals back to calculate percentages
                            period_entity_sales = pd.merge(
                                period_entity_sales, 
                                period_total_sales, 
                                on='period', 
                                how='left'
                            )
                            
                            # Calculate percentage
                            period_entity_sales['percentage'] = period_entity_sales['sales_value'] / period_entity_sales['total_sales'] * 100
                            
                            # Convert period to datetime for plotting
                            period_entity_sales['period_date'] = period_entity_sales['period'].dt.to_timestamp()
                            
                            # For SKU filtering, determine how many entities to show
                            if dist_by == "SKU" and sku_filter != "Normal/Específico":
                                # Remove the top 5 entities and "Others" logic
                                # Instead, just use all entities in the data
                                plot_data = period_entity_sales
                            else:
                                plot_data = period_entity_sales
                            
                            # Create the visualization
                            fig = go.Figure()
                            
                            # For Normal/Específico option, create special visualization with just two categories
                            if dist_by == "SKU" and sku_filter == "Normal/Específico":
                                # Group by period and market type
                                market_dist = plot_data.groupby(['period_date', 'market_type'])['percentage'].sum().reset_index()
                                
                                # Add trace for each market type
                                for market_type in market_dist['market_type'].unique():
                                    market_data = market_dist[market_dist['market_type'] == market_type]
                                    fig.add_trace(go.Scatter(
                                        x=market_data['period_date'],
                                        y=market_data['percentage'],
                                        mode='lines+markers',
                                        name=market_type,
                                        line=dict(width=2.5)
                                    ))
                                
                                chart_title = "Distribuição Percentual: Normal vs. Específico"
                            else:
                                # For regular visualizations, get unique entities
                                entities = plot_data[distribution_column].unique()
                                
                                # Sort entities by total percentage (descending)
                                entity_totals = plot_data.groupby(distribution_column)['percentage'].sum().sort_values(ascending=False)
                                
                                # Get only top 5 entities for the graph
                                top5_entities = entity_totals.head(5).index.tolist()
                                
                                # Add a trace for each of the top 5 entities only
                                for entity in top5_entities:
                                    entity_data = plot_data[plot_data[distribution_column] == entity]
                                    fig.add_trace(go.Scatter(
                                        x=entity_data['period_date'],
                                        y=entity_data['percentage'],
                                        mode='lines+markers',
                                        name=str(entity),
                                        line=dict(width=2.5)
                                    ))
                                
                                # Create chart title based on selections
                                if percent_dist_entity_selection == "Total das Seleções Atuais":
                                    chart_title = f"Distribuição Percentual por {dist_by} (Top 5)"
                                elif percent_dist_entity_selection == "Conjunto":
                                    chart_title = f"Distribuição Percentual por {dist_by} - Conjunto: {selected_percent_dist_set} (Top 5)"
                                elif percent_dist_entity_selection == "Individual":
                                    chart_title = f"Distribuição Percentual por {dist_by} - {entity_type.capitalize()}: {selected_percent_dist_entities[0]} (Top 5)"
                                
                                # Add filter info to title if applicable
                                if dist_by == "SKU" and sku_filter != "Todos SKUs" and sku_filter != "Normal/Específico":
                                    chart_title += f" ({sku_filter})"
                            
                            # Update layout
                            fig.update_layout(
                                title=chart_title,
                                xaxis_title="Período",
                                yaxis_title="Percentual (%)",
                                height=500,
                                template="plotly_dark",
                                plot_bgcolor='rgba(25, 25, 35, 1)',
                                paper_bgcolor='rgba(25, 25, 35, 1)',
                                legend=dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1
                                ),
                                hovermode="x unified"
                            )
                            
                            # Add range slider for better temporal navigation
                            fig.update_xaxes(
                                rangeslider=dict(
                                    visible=True,
                                    bgcolor='rgba(50, 50, 70, 0.7)',
                                    bordercolor='rgba(70, 70, 90, 1)',
                                    thickness=0.08
                                ),
                                rangeselector=dict(
                                    buttons=list([
                                        dict(count=1, label="1m", step="month", stepmode="backward"),
                                        dict(count=6, label="6m", step="month", stepmode="backward"),
                                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                                        dict(count=1, label="1y", step="year", stepmode="backward"),
                                        dict(step="all", label="Todos")
                                    ]),
                                    bgcolor='rgba(35, 35, 50, 1)',
                                    activecolor='rgba(70, 70, 120, 1)'
                                )
                            )
                            
                            # Set y-axis to always show 0-100%
                            fig.update_yaxes(range=[0, 100])
                            
                            # Display the chart
                            st.plotly_chart(fig, use_container_width=True)
                            
                            # Add statistics section
                            st.subheader("Estatísticas de Distribuição")
                            
                            # Calculate average distribution percentages for each entity
                            avg_percentages = plot_data.groupby(distribution_column)['percentage'].mean().reset_index()
                            avg_percentages = avg_percentages.sort_values('percentage', ascending=False)
                            avg_percentages.columns = ['Entidade', 'Percentual Médio']
                            avg_percentages['Percentual Médio'] = avg_percentages['Percentual Médio'].round(2).astype(str) + '%'
                            
                            # Display statistics for all entities, not just top 10
                            st.dataframe(
                                avg_percentages,  # Show all entities instead of just top 10
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            # Remove the note about showing only top 10
                            # If we have more than 10 entities, show a note
                            if len(avg_percentages) > 10:
                                st.caption(f"Mostrando todas as {len(avg_percentages)} entidades")
                        else:
                            st.warning("Nenhuma entidade selecionada para análise.")
        
        elif "Categorias" in st.session_state.agg_combination_type:
            # For ABC/XYZ categories, we need to load the categorization data
            # Usar as seleções salvas quando o botão foi clicado, não as atuais do UI
            aggregation_level = st.session_state.agg_current_selections['aggregation_level']
            abc_selection = st.session_state.agg_current_selections['abc']
            xyz_selection = st.session_state.agg_current_selections['xyz']
            
            # Escolher a classificação ABC/XYZ correta com base no nível de agregação
            if aggregation_level == "Família":
                if 'family_abc_xyz_data' not in st.session_state or st.session_state.family_abc_xyz_data is None:
                    st.warning("Dados de classificação ABC/XYZ para famílias não disponíveis. Por favor, carregue os dados novamente.")
                    return
                abc_xyz_df = st.session_state.family_abc_xyz_data
                entity_column = "family"
            elif aggregation_level == "Subfamília":
                if 'subfamily_abc_xyz_data' not in st.session_state or st.session_state.subfamily_abc_xyz_data is None:
                    st.warning("Dados de classificação ABC/XYZ para subfamílias não disponíveis. Por favor, carregue os dados novamente.")
                    return
                abc_xyz_df = st.session_state.subfamily_abc_xyz_data
                entity_column = "subfamily"
            else:  # Default para SKU
                if 'abc_xyz_data' not in st.session_state or st.session_state.abc_xyz_data is None:
                    st.warning("Dados de classificação ABC/XYZ para SKUs não disponíveis. Por favor, carregue os dados novamente.")
                    return
                abc_xyz_df = st.session_state.abc_xyz_data
                entity_column = "sku"
                
                # Se for SKU, aplicar filtro adicional de tipo de mercado
                if aggregation_level == "SKU" and 'agg_market_type' in st.session_state:
                    market_type = st.session_state.agg_market_type
                    
                    # Verificar se temos os SKUs de mercado específico carregados
                    if 'mercado_especifico_skus' not in st.session_state:
                        st.warning("Dados de mercado específico não disponíveis. Por favor, carregue os dados novamente.")
                        return
                    
                    mercado_especifico_skus = set(st.session_state.mercado_especifico_skus)
                    
                    # Filtrar com base no tipo de mercado
                    if market_type == "Mercado Específico":
                        # Manter apenas SKUs do mercado específico
                        abc_xyz_df = abc_xyz_df[abc_xyz_df['sku'].astype(str).isin(mercado_especifico_skus)]
                        
                    elif market_type == "Normal":
                        # Manter apenas SKUs que NÃO são do mercado específico
                        abc_xyz_df = abc_xyz_df[~abc_xyz_df['sku'].astype(str).isin(mercado_especifico_skus)]
                    
                    # Para "Todos", não aplicamos filtro adicional
            
            # Filter by ABC selection
            if abc_selection != "Todas":
                # Handle combined selections like "AB" or "ABC"
                abc_filters = list(abc_selection)
                abc_xyz_df = abc_xyz_df[abc_xyz_df['abc_class'].isin(abc_filters)]
            
            # Filter by XYZ selection
            if xyz_selection != "Todas":
                # Handle combined selections like "XY" or "XYZ"
                xyz_filters = list(xyz_selection)
                abc_xyz_df = abc_xyz_df[abc_xyz_df['xyz_class'].isin(xyz_filters)]
            
            # Get the entities that match our ABC/XYZ criteria (usando classificação baseada em 2024)
            selected_entities = abc_xyz_df[entity_column].unique().tolist()
            
            if not selected_entities:
                st.info(f"Nenhuma {aggregation_level} corresponde às categorias ABC/XYZ selecionadas.")
                return
            
            # Filtrar dados COM BASE NA CLASSIFICAÇÃO DE 2024, mas mostrar TODA A SÉRIE TEMPORAL
            # Filtrar apenas para as entidades selecionadas, mas manter todos os dados temporais
            filtered_entities_data = df[df[entity_column].isin(selected_entities)]
            
            if filtered_entities_data.empty:
                st.info(f"Não há dados para as {aggregation_level}s selecionadas.")
                return
            
            # Map the aggregation level to the column name
            aggregation_mapping = {
                "Família": "family",
                "Subfamília": "subfamily",
                "Formato": "format",
                "Gramagem": "weight",
                "SKU": "sku"
            }
            
            aggregation_column = aggregation_mapping[aggregation_level]
            
            # Create the chart title
            chart_title = f"Séries Agregadas - {aggregation_level} "
            chart_title += f"(ABC: {abc_selection}, XYZ: {xyz_selection})"
            
            # Adicionar informação do tipo de mercado no título quando for SKU
            if aggregation_level == "SKU" and 'market_type' in st.session_state.agg_current_selections:
                market_type = st.session_state.agg_current_selections['market_type']
                if market_type != "Todos":
                    chart_title += f" - {market_type}"
            
            # Group by the selected aggregation level
            entities = filtered_entities_data[aggregation_column].unique().tolist()
            
            # If too many entities, limit to top 10 by sales volume
            if len(entities) > 10:
                top_entities = filtered_entities_data.groupby(aggregation_column)['sales_value'].sum().nlargest(10).index.tolist()
                entities = top_entities
                chart_title += " (Top 10)"
            
            # Create period column for aggregation (mensal)
            filtered_entities_data['period'] = filtered_entities_data['invoice_date'].dt.to_period('M')
            
            # Usar gráfico de linhas em vez de barras para mostrar a série temporal completa
            fig = go.Figure()
            
            # Preparar dados agregados por período para cada entidade
            all_periods = filtered_entities_data['period'].unique()
            # Converter para lista e ordenar (em vez de usar sort() direto no PeriodArray)
            all_periods = sorted(all_periods)
            
            # Dicionário para armazenar os dados agregados
            aggregate_data = {}
            total_by_period = {}
            
            # Inicializar com zero para todos os períodos
            for period in all_periods:
                total_by_period[period] = 0
            
            # Calcular valores para cada entidade
            for entity in entities:
                entity_data = filtered_entities_data[filtered_entities_data[aggregation_column] == entity]
                entity_by_period = entity_data.groupby('period')['sales_value'].sum()
                
                # Armazenar os dados da entidade
                aggregate_data[entity] = entity_by_period
                
                # Adicionar ao total
                for period, value in entity_by_period.items():
                    total_by_period[period] += value
            
            # Converter período para timestamp para plotagem
            period_timestamps = [pd.Period(p).to_timestamp() for p in all_periods]
            
            # Adicionar opção para normalizar os dados
            col1, col2, col3 = st.columns([4, 2, 4])
            with col2:
                normalize_data = st.checkbox("Normalizar Dados (0-1)", value=False, key="abc_xyz_normalize_data", 
                                           help="Normaliza os valores de cada série para uma escala de 0 a 1")
            
            # Normalizar os dados se solicitado
            if normalize_data:
                # Normalizar cada série individualmente
                normalized_aggregate_data = {}
                normalized_total = {}
                
                # Normalizar os dados das entidades
                for entity in entities:
                    entity_data = aggregate_data[entity]
                    if entity_data.sum() > 0:  # Apenas normalizar se houver valores positivos
                        min_val = entity_data.min()
                        max_val = entity_data.max()
                        if max_val > min_val:  # Evitar divisão por zero
                            normalized_series = (entity_data - min_val) / (max_val - min_val)
                            normalized_aggregate_data[entity] = normalized_series
                        else:
                            normalized_aggregate_data[entity] = entity_data
                    else:
                        normalized_aggregate_data[entity] = entity_data
                
                # Normalizar o total
                min_total = min(total_by_period.values())
                max_total = max(total_by_period.values())
                if max_total > min_total:
                    for period, value in total_by_period.items():
                        normalized_total[period] = (value - min_total) / (max_total - min_total)
                else:
                    normalized_total = total_by_period
                
                # Usar os dados normalizados
                y_axis_title = "Valor Normalizado (0-1)"
                chart_title += " - Normalizado"
                
                # Preparar as séries normalizadas para o gráfico
                total_values = [normalized_total.get(p, 0) for p in all_periods]
                
                # Adicionar linha para o total normalizado
                fig.add_trace(go.Scatter(
                    x=period_timestamps,
                    y=total_values,
                    name=f"Total ({abc_selection}/{xyz_selection})",
                    mode='lines',
                    line=dict(color='#ffffff', width=3, dash='solid'),
                ))
                
                # Adicionar linhas para cada entidade individual normalizada
                for entity in entities:
                    entity_values = [normalized_aggregate_data[entity].get(p, 0) for p in all_periods]
                    fig.add_trace(go.Scatter(
                        x=period_timestamps,
                        y=entity_values,
                        name=str(entity),
                        mode='lines+markers',
                    ))
            else:
                # Usar os dados originais sem normalização
                y_axis_title = "Valor de Vendas"
                
                # Adicionar linha para o total (agregação)
                total_values = [total_by_period.get(p, 0) for p in all_periods]
                fig.add_trace(go.Scatter(
                    x=period_timestamps,
                    y=total_values,
                    name=f"Total ({abc_selection}/{xyz_selection})",
                    mode='lines',
                    line=dict(color='#ffffff', width=3, dash='solid'),
                ))
                
                # Adicionar linhas para cada entidade individual
                for entity in entities:
                    entity_values = [aggregate_data[entity].get(p, 0) for p in all_periods]
                    fig.add_trace(go.Scatter(
                        x=period_timestamps,
                        y=entity_values,
                        name=str(entity),
                        mode='lines+markers',
                    ))
            
            # Estilizar o gráfico
            fig.update_layout(
                title=chart_title,
                xaxis_title="Período",
                yaxis_title=y_axis_title,
                height=500,
                template="plotly_dark",
                plot_bgcolor='rgba(25, 25, 35, 1)',
                paper_bgcolor='rgba(25, 25, 35, 1)',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Adicionar range slider semelhante ao da página de séries temporais
            fig.update_xaxes(
                rangeslider=dict(
                    visible=True,
                    bgcolor='rgba(50, 50, 70, 0.7)',
                    bordercolor='rgba(70, 70, 90, 1)',
                    thickness=0.08
                ),
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all", label="Todos")
                    ]),
                    bgcolor='rgba(35, 35, 50, 1)',
                    activecolor='rgba(70, 70, 120, 1)'
                )
            )
            
            # Display the chart
            st.plotly_chart(fig, use_container_width=True)
            
            # Add statistics section
            st.subheader("Estatísticas")
            
            # Calculate total value across all periods
            total_value = sum(total_values)
            avg_value = total_value / len(all_periods) if all_periods else 0
            
            # Find entity with highest total sales
            entity_totals = {entity: sum(aggregate_data[entity].values) for entity in entities}
            if entity_totals:
                # Usar uma abordagem alternativa para encontrar a entidade com maior valor total
                max_entity = sorted(entity_totals.items(), key=lambda x: x[1], reverse=True)[0][0]
                max_value = entity_totals[max_entity]
            else:
                max_entity = "N/A"
                max_value = 0
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total de Vendas", f"{total_value:.2f}")
            
            with col2:
                st.metric("Média Mensal", f"{avg_value:.2f}")
            
            with col3:
                st.metric(
                    f"{aggregation_level} com Maior Venda", 
                    f"{max_entity} ({max_value:.2f})"
                )
    
    elif "Diferentes" in st.session_state.agg_combination_type and not st.session_state.agg_selected_entities and not st.session_state.agg_entity_sets:
        st.info(f"Por favor, adicione pelo menos um(a) {entity_type} ou defina um conjunto para visualizar o gráfico.") 
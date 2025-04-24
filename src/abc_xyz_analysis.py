"""
Module for ABC/XYZ analysis visualization and data processing.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Any

# Reaproveitamos algumas funções do módulo abc_xyz.py, mas com adaptações para a página de análise
def filter_active_skus(
    df: pd.DataFrame,
    year: int = 2024,
    sku_column: str = "sku",
    date_column: str = "invoice_date",
    family_filter: Optional[str] = None,
    subfamily_filter: Optional[str] = None,
    market_type: str = "Todos",
    mercado_especifico_skus: Optional[List[str]] = None
) -> List[str]:
    """
    Filter SKUs that were active (had sales) in the specified year with additional filters.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sales data
    year : int, default=2024
        The year to filter by
    sku_column : str, default="sku"
        Column name containing SKU identifiers
    date_column : str, default="invoice_date"
        Column name containing dates
    family_filter : Optional[str], default=None
        Filter by specific family or None for all families
    subfamily_filter : Optional[str], default=None
        Filter by specific subfamily or None for all subfamilies
    market_type : str, default="Todos"
        Filter by market type: "Normal", "Mercado Específico", or "Todos"
    mercado_especifico_skus : Optional[List[str]], default=None
        List of SKUs in the specific market, needed when market_type is not "Todos"
        
    Returns:
    --------
    List[str]
        List of active SKUs in the specified year with applied filters
    """
    if df.empty:
        return []
    
    # Ensure date column is datetime
    df_copy = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
        df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
    
    # Filter data for the specified year
    year_data = df_copy[df_copy[date_column].dt.year == year]
    
    # If no data found for the year
    if year_data.empty:
        return []
    
    # Apply family filter if specified
    if family_filter and family_filter != "Todas" and "family" in year_data.columns:
        year_data = year_data[year_data["family"] == family_filter]
    
    # Apply subfamily filter if specified
    if subfamily_filter and subfamily_filter != "Todas" and "subfamily" in year_data.columns:
        year_data = year_data[year_data["subfamily"] == subfamily_filter]
    
    # Apply market type filter if specified
    if market_type != "Todos" and mercado_especifico_skus:
        mercado_especifico_set = set(str(sku) for sku in mercado_especifico_skus)
        
        if market_type == "Mercado Específico":
            # Keep only SKUs from the specific market
            year_data = year_data[year_data[sku_column].astype(str).isin(mercado_especifico_set)]
        elif market_type == "Normal":
            # Keep only SKUs NOT from the specific market
            year_data = year_data[~year_data[sku_column].astype(str).isin(mercado_especifico_set)]
    
    # Get unique SKUs with sales in that year and convert to list
    # Ensure each element is a string
    skus_list: List[str] = [str(sku) for sku in year_data[sku_column].unique()]
    
    return skus_list

def get_abc_xyz_data(
    df: pd.DataFrame,
    family_filter: Optional[str] = None,
    subfamily_filter: Optional[str] = None,
    market_type: str = "Todos",
    mercado_especifico_skus: Optional[List[str]] = None,
    a_threshold: float = 0.8,
    b_threshold: float = 0.95,
    x_threshold: float = 0.2,
    y_threshold: float = 0.5,
    year: int = 2024,
    sku_column: str = "sku",
    value_column: str = "sales_value",
    date_column: str = "invoice_date"
) -> pd.DataFrame:
    """
    Get ABC/XYZ classification data with filters.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sales data
    family_filter : Optional[str], default=None
        Filter by specific family or None for all families
    subfamily_filter : Optional[str], default=None
        Filter by specific subfamily or None for all subfamilies
    market_type : str, default="Todos"
        Filter by market type: "Normal", "Mercado Específico", or "Todos"
    mercado_especifico_skus : Optional[List[str]], default=None
        List of SKUs in the specific market, needed when market_type is not "Todos"
    a_threshold : float, default=0.8
        Threshold for class A (0-80% of total sales)
    b_threshold : float, default=0.95
        Threshold for class B (80-95% of total sales)
    x_threshold : float, default=0.2
        Threshold for class X (CV 0-20%)
    y_threshold : float, default=0.5
        Threshold for class Y (CV 20-50%)
    year : int, default=2024
        The year to filter data by for classification
    sku_column : str, default="sku"
        Column name containing SKU identifiers
    value_column : str, default="sales_value"
        Column name containing sales values
    date_column : str, default="invoice_date"
        Column name containing dates
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with ABC/XYZ classification and metadata
    """
    # Ensure date column is datetime
    df_copy = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
        df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
    
    # Filter by family if specified
    if family_filter and family_filter != "Todas" and "family" in df_copy.columns:
        df_copy = df_copy[df_copy["family"] == family_filter]
    
    # Filter by subfamily if specified
    if subfamily_filter and subfamily_filter != "Todas" and "subfamily" in df_copy.columns:
        df_copy = df_copy[df_copy["subfamily"] == subfamily_filter]
    
    # Filter data for the specified year
    year_data = df_copy[df_copy[date_column].dt.year == year]
    
    # If no data found for the year
    if year_data.empty:
        return pd.DataFrame()
    
    # Apply market type filter if specified
    if market_type != "Todos" and mercado_especifico_skus:
        mercado_especifico_set = set(str(sku) for sku in mercado_especifico_skus)
        
        if market_type == "Mercado Específico":
            # Keep only SKUs from the specific market
            year_data = year_data[year_data[sku_column].astype(str).isin(mercado_especifico_set)]
        elif market_type == "Normal":
            # Keep only SKUs NOT from the specific market
            year_data = year_data[~year_data[sku_column].astype(str).isin(mercado_especifico_set)]
    
    # Get active SKUs with all filters applied
    active_skus = [str(sku) for sku in year_data[sku_column].unique()]
    
    if not active_skus:
        return pd.DataFrame()
    
    # Calculate total sales by SKU
    sku_sales = year_data.groupby(sku_column)[value_column].sum().reset_index()
    sku_sales.columns = [sku_column, 'total_sales']
    
    # Sort by total sales in descending order
    sku_sales = sku_sales.sort_values('total_sales', ascending=False)
    
    # Calculate percentages
    total = sku_sales['total_sales'].sum()
    sku_sales['percent_of_total'] = sku_sales['total_sales'] / total
    
    # Calculate cumulative percentage
    sku_sales['cumulative_percent'] = sku_sales['percent_of_total'].cumsum()
    
    # Classify into A, B, C
    conditions = [
        (sku_sales['cumulative_percent'] <= a_threshold),
        (sku_sales['cumulative_percent'] <= b_threshold),
        (sku_sales['cumulative_percent'] > b_threshold)
    ]
    choices = ['A', 'B', 'C']
    sku_sales['abc_class'] = np.select(conditions, choices, default='C')
    
    # Calculate XYZ classification
    # Create a date period (month) for aggregation
    year_data['period'] = year_data[date_column].dt.to_period('M')
    
    # Calculate sales by SKU and period
    sku_period_sales = year_data.groupby([sku_column, 'period'])[value_column].sum().reset_index()
    
    # Create all 12 months of the specified year
    all_months = [pd.Period(f"{year}-{month}", freq='M') for month in range(1, 13)]
    
    # Calculate CV for each SKU
    sku_stats = []
    for sku in active_skus:
        # Filter data for this SKU
        sku_data = sku_period_sales[sku_period_sales[sku_column].astype(str) == sku]
        
        # Create DataFrame with all months and fill with sales (or zero for months without sales)
        monthly_sales = pd.DataFrame({'period': all_months})
        monthly_sales = monthly_sales.merge(
            sku_data[['period', value_column]], 
            on='period', 
            how='left'
        ).fillna(0)
        
        if len(monthly_sales) > 0:  # Ensure we have data
            # Calculate statistics with all 12 months, including zeros
            sales_values = monthly_sales[value_column].to_numpy()
            mean_sales = np.mean(sales_values)
            std_sales = np.std(sales_values, ddof=1)  # using ddof=1 for STDEV.S (sample)
            
            # Coefficient of variation
            cv = std_sales / mean_sales if mean_sales > 0 else float('inf')
            
            # Classify
            if cv <= x_threshold:
                xyz_class = 'X'
            elif cv <= y_threshold:
                xyz_class = 'Y'
            else:
                xyz_class = 'Z'
                
            sku_stats.append({
                sku_column: sku,
                'mean_sales': mean_sales,
                'std_sales': std_sales,
                'cv': cv,
                'xyz_class': xyz_class
            })
        else:
            # Not enough data for coefficient of variation
            sku_stats.append({
                sku_column: sku,
                'mean_sales': 0,
                'std_sales': 0,
                'cv': float('inf'),
                'xyz_class': 'Z'  # Classify as Z if not enough data
            })
    
    # Convert to DataFrame
    xyz_df = pd.DataFrame(sku_stats)
    
    # Merge ABC and XYZ classifications
    if not xyz_df.empty and not sku_sales.empty:
        # Ensure both dataframes have the same type for the SKU column
        sku_sales[sku_column] = sku_sales[sku_column].astype(str)
        xyz_df[sku_column] = xyz_df[sku_column].astype(str)
        
        # Merge the results
        combined = sku_sales.merge(xyz_df, on=sku_column, how='outer')
        
        # Create combined classification
        combined['abc_xyz_class'] = combined['abc_class'] + combined['xyz_class']
        
        # Sort by total_sales (descending) to show most important SKUs first
        if 'total_sales' in combined.columns:
            combined = combined.sort_values('total_sales', ascending=False)
        
        # Add metadata if available
        if 'family' in year_data.columns and 'subfamily' in year_data.columns:
            # Get metadata for each SKU
            agg_dict = {'family': 'first', 'subfamily': 'first'}
            
            # Add optional columns only if they exist
            if 'weight' in year_data.columns:
                agg_dict['weight'] = 'first'
            if 'format' in year_data.columns:
                agg_dict['format'] = 'first'
            
            metadata = year_data.groupby(sku_column).agg(agg_dict).reset_index()
            
            # Remove None columns
            metadata = metadata.loc[:, ~metadata.columns.isnull()]
            
            # Merge with combined data
            combined = combined.merge(metadata, on=sku_column, how='left')
        
        return combined
    else:
        return pd.DataFrame()

def create_abc_xyz_matrix(
    df: pd.DataFrame,
    custom_title: Optional[str] = None
) -> Tuple[go.Figure, Dict[str, Dict[str, Any]]]:
    """
    Create a visual matrix of ABC/XYZ classification with detailed statistics.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with ABC/XYZ classification
    custom_title : Optional[str], default=None
        Custom title for the matrix
        
    Returns:
    --------
    Tuple[go.Figure, Dict[str, Dict[str, Any]]]
        A tuple containing the Plotly figure and statistics for each category
    """
    # Ordem correta: A na primeira linha, B na segunda, C na terceira
    # Invertemos a ordem dos elementos porque o eixo y no plotly cresce para cima
    classes = ['C', 'B', 'A']
    variations = ['X', 'Y', 'Z']
    
    # Initialize the matrix with zeros
    count_matrix = np.zeros((len(classes), len(variations)), dtype=int)
    percentage_matrix = np.zeros((len(classes), len(variations)), dtype=float)
    
    # Dictionary to store detailed statistics
    stats = {}
    
    # Total sales from the dataset
    total_sales = df['total_sales'].sum() if 'total_sales' in df.columns else 0
    
    # Fill the matrix with counts and calculate percentages
    if 'abc_class' in df.columns and 'xyz_class' in df.columns and 'total_sales' in df.columns:
        for i, cls in enumerate(classes):
            for j, var in enumerate(variations):
                category = f"{cls}{var}"
                category_df = df[(df['abc_class'] == cls) & (df['xyz_class'] == var)]
                
                # Count
                count = len(category_df)
                count_matrix[i, j] = count
                
                # Sales percentage
                sales_sum = category_df['total_sales'].sum()
                sales_percentage = (sales_sum / total_sales * 100) if total_sales > 0 else 0
                percentage_matrix[i, j] = sales_percentage
                
                # Store statistics
                stats[category] = {
                    'count': count,
                    'sales_value': sales_sum,
                    'sales_percentage': sales_percentage
                }
    
    # Create text for each cell
    text_matrix = []
    for i in range(len(classes)):
        text_row = []
        for j in range(len(variations)):
            count = count_matrix[i, j]
            percentage = percentage_matrix[i, j]
            text = f"{count} SKUs<br>{percentage:.1f}% vendas"
            text_row.append(text)
        text_matrix.append(text_row)
    
    # Definir cores mais contrastantes para cada linha (categoria)
    colors = {
        'A': '#82E0AA',  # Verde mais vibrante
        'B': '#F7DC6F',  # Amarelo mais vibrante
        'C': '#EC7063',  # Vermelho mais vibrante
    }
    
    # Hover text com informações adicionais
    hover_matrix = []
    for i, cls in enumerate(classes):
        hover_row = []
        for j, var in enumerate(variations):
            count = count_matrix[i, j]
            percentage = percentage_matrix[i, j]
            hover_text = f"<b>Quadrante {cls}{var}:</b><br>{count} SKUs<br>{percentage:.1f}% das vendas"
            hover_row.append(hover_text)
        hover_matrix.append(hover_row)
    
    # Criar figura base
    fig = go.Figure()
    
    # Configuração de layout para a grade - reduzir tamanho e margens
    grid_width = 3  # número de colunas
    grid_height = 3  # número de linhas
    cell_width = 0.28  # largura de cada célula - aumentada ligeiramente
    cell_height = 0.28  # altura de cada célula - aumentada ligeiramente
    x_margin = 0.03  # margem entre células horizontalmente - reduzida
    y_margin = 0.03  # margem entre células verticalmente - reduzida
    
    # Adicionar cada célula como um retângulo separado
    for i, cls in enumerate(classes):
        for j, var in enumerate(variations):
            # Calcular posição de cada célula com margens
            x0 = j * (cell_width + x_margin)
            x1 = x0 + cell_width
            y0 = i * (cell_height + y_margin)
            y1 = y0 + cell_height
            
            # Texto para a célula e hover
            count = count_matrix[i, j]
            percentage = percentage_matrix[i, j]
            cell_text = f"{count} SKUs<br>{percentage:.1f}% vendas"
            hover_text = hover_matrix[i][j]
            
            # Borda mais escura para destacar a célula
            border_color = 'rgba(0,0,0,0.3)'
            
            # Adicionar retângulo para cada célula
            fig.add_trace(go.Scatter(
                x=[x0, x1, x1, x0, x0],
                y=[y0, y0, y1, y1, y0],
                fill="toself",
                fillcolor=colors[cls],
                line=dict(width=1, color=border_color),
                mode="lines",
                name=f"{cls}{var}",
                text=hover_text,
                hoverinfo="text",
                showlegend=False
            ))
            
            # Adicionar texto no centro da célula
            fig.add_annotation(
                x=(x0 + x1) / 2,
                y=(y0 + y1) / 2,
                text=cell_text,
                showarrow=False,
                font=dict(size=11, color="black", family="Arial, sans-serif"),
                align="center",
                xanchor="center",
                yanchor="middle"
            )
    
    # Adicionar cabeçalhos das colunas (X, Y, Z)
    for j, var in enumerate(variations):
        x_pos = j * (cell_width + x_margin) + cell_width / 2
        fig.add_annotation(
            x=x_pos,
            y=grid_height * (cell_height + y_margin) + 0.01,
            text=f"{var} " + ({"X": "(≤20%)", "Y": "(20-50%)", "Z": "(>50%)"}[var]),
            showarrow=False,
            yshift=5,
            font=dict(size=14, color="white", family="Arial, sans-serif"),
            align="center",
            xanchor="center",
            yanchor="bottom"
        )
    
    # Adicionar título para o eixo X (acima das colunas)
    fig.add_annotation(
        x=(grid_width * (cell_width + x_margin)) / 2 - 0.02,
        y=grid_height * (cell_height + y_margin) + 0.05,
        text="Coeficiente de Variação (XYZ)",
        showarrow=False,
        yshift=15,
        font=dict(size=14, color="white", family="Arial, sans-serif"),
        align="center",
        xanchor="center",
        yanchor="bottom",
        bgcolor="rgba(30, 30, 45, 0.7)",
        borderpad=4
    )
    
    # Adicionar título para o eixo Y (ao lado das linhas)
    fig.add_annotation(
        x=-0.12,
        y=(grid_height * (cell_height + y_margin)) / 2,
        text="Volume de Vendas (ABC)",
        showarrow=False,
        xshift=-22,
        font=dict(size=14, color="white", family="Arial, sans-serif"),
        align="center",
        xanchor="center",
        yanchor="middle",
        textangle=-90,
        bgcolor="rgba(30, 30, 45, 0.7)",
        borderpad=4
    )
    
    # Adicionar cabeçalhos das linhas (A, B, C) - com rótulos corretos
    # Importante: usamos os índices invertidos (2=A, 1=B, 0=C) porque a ordem foi invertida
    row_labels = [
        {'pos': 2, 'label': 'A', 'desc': '(80%)'},
        {'pos': 1, 'label': 'B', 'desc': '(15%)'},
        {'pos': 0, 'label': 'C', 'desc': '(5%)'}
    ]
    
    for item in row_labels:
        y_pos = item['pos'] * (cell_height + y_margin) + cell_height / 2
        fig.add_annotation(
            x=-0.05,
            y=y_pos,
            text=f"{item['label']} {item['desc']}",
            showarrow=False,
            xshift=-5,
            font=dict(size=14, color="white", family="Arial, sans-serif"),
            align="center",
            xanchor="right",
            yanchor="middle",
            bgcolor="rgba(30, 30, 45, 0.7)",
            borderpad=3
        )
    
    # Ajustar o layout da figura para ser mais responsivo e compacto
    fig.update_layout(
        showlegend=False,
        xaxis=dict(
            range=[-0.15, grid_width * (cell_width + x_margin)],
            showticklabels=False,
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            range=[-0.06, grid_height * (cell_height + y_margin) + 0.07],
            showticklabels=False,
            showgrid=False,
            zeroline=False
        ),
        autosize=True,
        width=700,
        height=480,
        paper_bgcolor='rgba(30, 30, 45, 1)',
        plot_bgcolor='rgba(30, 30, 45, 1)',
        font=dict(color='white', family="Arial, sans-serif"),
        margin=dict(l=60, r=15, t=50, b=30),
        hovermode="closest"
    )
    
    return fig, stats

def create_quadrant_chart(df: pd.DataFrame) -> go.Figure:
    """
    Create a horizontal bar chart showing SKUs per quadrant (ABC/XYZ class).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with ABC/XYZ classification
        
    Returns:
    --------
    go.Figure
        A Plotly bar chart figure
    """
    if 'abc_xyz_class' not in df.columns:
        return go.Figure()
    
    # Count SKUs per category
    quadrant_counts = df['abc_xyz_class'].value_counts().reset_index()
    quadrant_counts.columns = ['quadrant', 'count']
    
    # Sort in logical order
    quadrant_order = ['AX', 'AY', 'AZ', 'BX', 'BY', 'BZ', 'CX', 'CY', 'CZ']
    quadrant_counts['order'] = quadrant_counts['quadrant'].map({q: i for i, q in enumerate(quadrant_order)})
    quadrant_counts = quadrant_counts.sort_values('order')
    
    # Usar exatamente as mesmas cores da matriz ABC/XYZ para consistência
    # Cores base para cada categoria A, B, C
    base_colors = {
        'A': '#82E0AA',  # Verde (mesmo da matriz)
        'B': '#F7DC6F',  # Amarelo (mesmo da matriz)
        'C': '#EC7063',  # Vermelho (mesmo da matriz)
    }
    
    # Atribuir a cor base baseada apenas na primeira letra (A, B ou C) do quadrante
    color_list = [base_colors[q[0]] for q in quadrant_counts['quadrant']]
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=quadrant_counts['quadrant'],
        x=quadrant_counts['count'],
        orientation='h',
        marker_color=color_list,
        text=quadrant_counts['count'],
        textposition='outside',
        hoverinfo='text',
        hovertext=[f"{q}: {c} SKUs" for q, c in zip(quadrant_counts['quadrant'], quadrant_counts['count'])]
    ))
    
    # Update layout - aumentar altura para alinhar com a matriz
    fig.update_layout(
        xaxis=dict(
            title='Número de SKUs',
            showgrid=True,
            gridcolor='rgba(70, 70, 90, 0.3)'
        ),
        yaxis=dict(
            title='Quadrante ABC/XYZ',
            categoryorder='array',
            categoryarray=quadrant_counts['quadrant'],
            showgrid=True,
            gridcolor='rgba(70, 70, 90, 0.3)'
        ),
        height=480,  # Aumentada para corresponder à altura da matriz ABC/XYZ
        paper_bgcolor='rgba(30, 30, 45, 1)',
        plot_bgcolor='rgba(30, 30, 45, 1)',
        font=dict(color='white', family="Arial, sans-serif"),
        margin=dict(l=20, r=30, t=30, b=30),  # Ajustada margem superior (t) para remover espaço do título
    )
    
    return fig

def get_sku_details(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """
    Get detailed information for the top N SKUs by sales volume.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with ABC/XYZ classification and metadata
    top_n : int, default=5
        Number of top SKUs to return
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with detailed information for the top N SKUs
    """
    if df.empty or 'total_sales' not in df.columns:
        return pd.DataFrame()
    
    # Sort by total sales (descending)
    sorted_df = df.sort_values('total_sales', ascending=False)
    
    # Get top N SKUs
    top_skus = sorted_df.head(top_n).copy()
    
    # Format the coefficient of variation
    if 'cv' in top_skus.columns:
        top_skus['cv'] = top_skus['cv'].map(lambda x: f"{x:.2f}" if not pd.isna(x) and x != float('inf') else "∞")
    
    # Add a formatted percent column
    if 'percent_of_total' in top_skus.columns:
        top_skus['percent_formatted'] = top_skus['percent_of_total'].map(lambda x: f"{x*100:.1f}%")
    
    return top_skus

def get_subfamily_options(df: pd.DataFrame, selected_family: str) -> List[str]:
    """
    Get subfamily options based on the selected family.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sales data
    selected_family : str
        Selected family to filter subfamilies
        
    Returns:
    --------
    List[str]
        List of subfamilies associated with the selected family
    """
    if df.empty or 'family' not in df.columns or 'subfamily' not in df.columns:
        return []
    
    if selected_family and selected_family != "Todas":
        # Filter subfamilies for the selected family
        subfamilies = df[df['family'] == selected_family]['subfamily'].unique().tolist()
    else:
        # Get all subfamilies
        subfamilies = df['subfamily'].unique().tolist()
    
    # Ensure 'Todas' is the first option
    subfamilies = sorted(subfamilies)
    return ["Todas"] + subfamilies

def get_quadrant_interpretation(quadrant: str) -> str:
    """
    Get interpretation for a specific ABC/XYZ quadrant.
    
    Parameters:
    -----------
    quadrant : str
        The ABC/XYZ quadrant (e.g., 'AX', 'BY', 'CZ')
        
    Returns:
    --------
    str
        Description of the quadrant characteristics
    """
    interpretations = {
        'AX': "Alto volume com consumo regular e previsível. Produtos estratégicos que requerem garantia de disponibilidade constante e gestão eficiente de stock. Foco em acordos de fornecimento de longo prazo e monitoramento proativo da cadeia de suprimentos.",
        
        'AY': "Alto volume com variabilidade média. Produtos importantes que necessitam monitoramento contínuo das tendências de demanda. Recomenda-se revisões frequentes de previsão e manutenção de stock de segurança adequado para absorver flutuações.",
        
        'AZ': "Alto volume com alta variabilidade. Produtos críticos que exigem atenção especial na gestão. Requer combinação de análise aprofundada das causas da variabilidade, possível segmentação de mercados e estratégias adaptativas de stock.",
        
        'BX': "Volume médio com consumo regular. Produtos de importância intermediária que beneficiam-se de pedidos regulares com quantidades bem dimensionadas. Gestão equilibrada com foco em eficiência operacional.",
        
        'BY': "Volume médio com variabilidade média. Produtos que exigem revisões periódicas de demanda e ajustes moderados nos níveis de stock. Considerar políticas flexíveis de reabastecimento e monitorar sazonalidades.",
        
        'BZ': "Volume médio com alta variabilidade. Produtos que requerem avaliação cautelosa para determinar causas de volatilidade. Considerar possibilidades de consolidação de entregas ou ajuste de portfólio para melhor previsibilidade.",
        
        'CX': "Baixo volume com consumo regular. Produtos de nicho com demanda estável mas limitada. Adequados para pedidos ocasionais em pequenas quantidades, potencialmente com estratégia de cross-docking ou entregas diretas.",
        
        'CY': "Baixo volume com variabilidade média. Produtos secundários que são candidatos a pedidos sob demanda ou consolidados. Avaliar o valor estratégico destes SKUs para o portfólio completo e possível racionalização.",
        
        'CZ': "Baixo volume com alta variabilidade. Produtos com baixa relevância financeira e alta complexidade de gestão. Potenciais candidatos à descontinuação ou redefinição de estratégia comercial. Recomenda-se análise de rentabilidade detalhada."
    }
    
    return interpretations.get(quadrant, "") 
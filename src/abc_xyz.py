"""
Module for ABC/XYZ classification of SKUs.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from typing import Dict, List, Tuple

def filter_active_skus(
    df: pd.DataFrame,
    year: int = 2024,
    sku_column: str = "sku",
    date_column: str = "invoice_date"
) -> List[str]:
    """
    Filter SKUs that were active (had sales) in the specified year.
    
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
        
    Returns:
    --------
    List[str]
        List of active SKUs in the specified year
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
    
    # Get unique SKUs with sales in that year and convert to list
    # Garantir que cada elemento seja string
    skus_list: List[str] = [str(sku) for sku in year_data[sku_column].unique()]
    
    return skus_list

def classify_abc(
    df: pd.DataFrame,
    sku_column: str = "sku",
    value_column: str = "sales_value",
    a_threshold: float = 0.8,
    b_threshold: float = 0.95,
    year: int = 2024,
    date_column: str = "invoice_date"
) -> pd.DataFrame:
    """
    Classify SKUs into A, B, C categories based on sales volume.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sales data
    sku_column : str, default="sku"
        Column name containing SKU identifiers
    value_column : str, default="sales_value"
        Column name containing sales values
    a_threshold : float, default=0.8
        Threshold for class A (0-80% of total sales)
    b_threshold : float, default=0.95
        Threshold for class B (80-95% of total sales)
    year : int, default=2024
        The year to filter data by for classification
    date_column : str, default="invoice_date"
        Column name containing dates
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with SKUs and their ABC classification
    """
    if df.empty:
        return pd.DataFrame(columns=[sku_column, 'total_sales', 'percent_of_total', 'cumulative_percent', 'abc_class'])
    
    # Ensure date column is datetime
    df_copy = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
        df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
    
    # Filter data for the specified year
    year_data = df_copy[df_copy[date_column].dt.year == year]
    
    # Get active SKUs in the specified year
    active_skus = filter_active_skus(df, year, sku_column, date_column)
    
    # If no SKUs are active in the specified year, return empty DataFrame
    if not active_skus:
        return pd.DataFrame(columns=[sku_column, 'total_sales', 'percent_of_total', 'cumulative_percent', 'abc_class'])
    
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
    
    return sku_sales

def classify_xyz(
    df: pd.DataFrame,
    sku_column: str = "sku",
    value_column: str = "sales_value", 
    date_column: str = "invoice_date",
    x_threshold: float = 0.2,
    y_threshold: float = 0.5,
    year: int = 2024
) -> pd.DataFrame:
    """
    Classify SKUs into X, Y, Z categories based on coefficient of variation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sales data
    sku_column : str, default="sku"
        Column name containing SKU identifiers
    value_column : str, default="sales_value"
        Column name containing sales values
    date_column : str, default="invoice_date"
        Column name containing dates
    x_threshold : float, default=0.2
        Threshold for class X (CV 0-20%)
    y_threshold : float, default=0.5
        Threshold for class Y (CV 20-50%)
    year : int, default=2024
        The year to filter data by for classification
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with SKUs and their XYZ classification
    """
    if df.empty:
        return pd.DataFrame(columns=[sku_column, 'mean_sales', 'std_sales', 'cv', 'xyz_class'])
    
    # Ensure date column is datetime
    df_copy = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
        df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
    
    # Filter data for the specified year
    year_data = df_copy[df_copy[date_column].dt.year == year]
    
    # Get active SKUs in the specified year
    active_skus = filter_active_skus(df, year, sku_column, date_column)
    
    # If no SKUs are active in the specified year, return empty DataFrame
    if not active_skus:
        return pd.DataFrame(columns=[sku_column, 'mean_sales', 'std_sales', 'cv', 'xyz_class'])
    
    # Create a date period (month) for aggregation
    year_data['period'] = year_data[date_column].dt.to_period('M')
    
    # Calculate sales by SKU and period
    sku_period_sales = year_data.groupby([sku_column, 'period'])[value_column].sum().reset_index()
    
    # Criar todos os 12 meses do ano especificado
    all_months = [pd.Period(f"{year}-{month}", freq='M') for month in range(1, 13)]
    
    # Calculate statistics for each SKU
    sku_stats = []
    for sku in active_skus:
        # Convert the SKU string value back to the original type if needed
        if sku_period_sales[sku_column].dtype != object:
            try:
                sku_value = type(sku_period_sales[sku_column].iloc[0])(sku) if not sku_period_sales.empty else sku
            except (ValueError, TypeError):
                sku_value = sku
        else:
            sku_value = sku
            
        # Filtrar dados apenas para este SKU
        sku_data = sku_period_sales[sku_period_sales[sku_column] == sku_value]
        
        # Criar um DataFrame com todos os meses e preencher com vendas (ou zero para meses sem vendas)
        monthly_sales = pd.DataFrame({'period': all_months})
        monthly_sales = monthly_sales.merge(
            sku_data[['period', value_column]], 
            on='period', 
            how='left'
        ).fillna(0)
        
        if len(monthly_sales) > 0:  # Garantir que temos dados
            # Calcular estatísticas com todos os 12 meses, incluindo zeros
            sales_values = monthly_sales[value_column].to_numpy()  # Converter para NumPy array
            mean_sales = np.mean(sales_values)
            std_sales = np.std(sales_values, ddof=1)  # usando ddof=1 para STDEV.S (amostra)
            
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
                sku_column: sku,  # Keep as string for consistency
                'mean_sales': mean_sales,
                'std_sales': std_sales,
                'cv': cv,
                'xyz_class': xyz_class
            })
        else:
            # Not enough data for coefficient of variation
            sku_stats.append({
                sku_column: sku,  # Keep as string for consistency
                'mean_sales': 0,
                'std_sales': 0,
                'cv': float('inf'),
                'xyz_class': 'Z'  # Classify as Z if not enough data
            })
    
    return pd.DataFrame(sku_stats)

def abc_xyz_classification(
    df: pd.DataFrame,
    sku_column: str = "sku",
    value_column: str = "sales_value",
    date_column: str = "invoice_date",
    year: int = 2024
) -> pd.DataFrame:
    """
    Perform both ABC and XYZ classification on SKUs.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sales data
    sku_column : str, default="sku"
        Column name containing SKU identifiers
    value_column : str, default="sales_value"
        Column name containing sales values
    date_column : str, default="invoice_date"
        Column name containing dates
    year : int, default=2024
        The year to filter data by for classification
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with SKUs and their combined ABC-XYZ classification, sorted by total_sales descending
    """
    # Get active SKUs in the specified year
    active_skus = filter_active_skus(df, year, sku_column, date_column)
    
    # If no SKUs are active in the specified year, return empty DataFrame
    if not active_skus:
        return pd.DataFrame(columns=[sku_column, 'abc_class', 'xyz_class', 'abc_xyz_class'])
    
    # Perform ABC classification
    abc_df = classify_abc(df, sku_column, value_column, year=year, date_column=date_column)
    
    # Perform XYZ classification
    xyz_df = classify_xyz(df, sku_column, value_column, date_column, year=year)
    
    # Ensure both dataframes have the same type for the SKU column (string)
    abc_df[sku_column] = abc_df[sku_column].astype(str)
    xyz_df[sku_column] = xyz_df[sku_column].astype(str)
    
    # Merge the results
    combined = abc_df.merge(xyz_df, on=sku_column, how='outer')
    
    # Create combined classification
    combined['abc_xyz_class'] = combined['abc_class'] + combined['xyz_class']
    
    # Sort by total_sales (descending) to show most important SKUs first
    if 'total_sales' in combined.columns:
        combined = combined.sort_values('total_sales', ascending=False)
    
    return combined

def create_abc_xyz_matrix(df: pd.DataFrame) -> Tuple[go.Figure, Dict[str, int]]:
    """
    Create a visual matrix of ABC-XYZ classification.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with ABC-XYZ classification
        
    Returns:
    --------
    Tuple[go.Figure, Dict[str, int]]
        A tuple containing the Plotly figure and a dictionary with counts for each category
    """
    # Create a classification matrix
    classes = ['A', 'B', 'C']
    variations = ['X', 'Y', 'Z']
    
    # Initialize the matrix with zeros
    matrix = np.zeros((len(classes), len(variations)), dtype=int)
    counts = {}
    
    # Fill the matrix with counts
    if 'abc_class' in df.columns and 'xyz_class' in df.columns:
        for i, cls in enumerate(classes):
            for j, var in enumerate(variations):
                category = f"{cls}{var}"
                count = len(df[(df['abc_class'] == cls) & (df['xyz_class'] == var)])
                matrix[i, j] = count
                counts[category] = count
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=variations,
        y=classes,
        text=matrix.astype(str),
        texttemplate="%{text}",
        textfont={"size": 20},
        colorscale='Viridis',
        showscale=False
    ))
    
    # Update layout
    fig.update_layout(
        title='ABC-XYZ Classification Matrix (2024 Data)',
        xaxis_title='Coefficient of Variation (XYZ)',
        yaxis_title='Sales Volume (ABC)',
        xaxis=dict(side='top'),
        width=600,
        height=500,
        margin=dict(l=50, r=50, t=100, b=50),
    )
    
    # Add annotations
    annotations = []
    
    # X-axis annotations
    annotations.append(dict(
        x=0, y=1.15, 
        text="X: CV ≤ 20%<br>Stable demand",
        showarrow=False,
        xref="x", yref="paper",
        font=dict(size=10)
    ))
    annotations.append(dict(
        x=1, y=1.15, 
        text="Y: CV 20-50%<br>Variable demand",
        showarrow=False,
        xref="x", yref="paper",
        font=dict(size=10)
    ))
    annotations.append(dict(
        x=2, y=1.15, 
        text="Z: CV > 50%<br>Highly variable",
        showarrow=False,
        xref="x", yref="paper",
        font=dict(size=10)
    ))
    
    # Y-axis annotations
    annotations.append(dict(
        x=-0.15, y=0, 
        text="A: Top 80% sales",
        showarrow=False,
        xref="paper", yref="y",
        font=dict(size=10)
    ))
    annotations.append(dict(
        x=-0.15, y=1, 
        text="B: Next 15% sales",
        showarrow=False,
        xref="paper", yref="y",
        font=dict(size=10)
    ))
    annotations.append(dict(
        x=-0.15, y=2, 
        text="C: Last 5% sales",
        showarrow=False,
        xref="paper", yref="y",
        font=dict(size=10)
    ))
    
    fig.update_layout(annotations=annotations)
    
    return fig, counts

def filter_active_entities(
    df: pd.DataFrame,
    entity_column: str,
    year: int = 2024,
    date_column: str = "invoice_date"
) -> List[str]:
    """
    Filter entities (families, subfamilies, or SKUs) that were active (had sales) in the specified year.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sales data
    entity_column : str
        Column name containing the entity to filter by (family, subfamily, or sku)
    year : int, default=2024
        The year to filter by
    date_column : str, default="invoice_date"
        Column name containing dates
        
    Returns:
    --------
    List[str]
        List of active entities in the specified year
    """
    if df.empty or entity_column not in df.columns:
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
    
    # Get unique entities with sales in that year and convert to list
    entities_list: List[str] = [str(entity) for entity in year_data[entity_column].unique()]
    
    return entities_list

def classify_entity_abc(
    df: pd.DataFrame,
    entity_column: str,
    value_column: str = "sales_value",
    a_threshold: float = 0.8,
    b_threshold: float = 0.95,
    year: int = 2024,
    date_column: str = "invoice_date"
) -> pd.DataFrame:
    """
    Classify entities (families, subfamilies, or SKUs) into A, B, C categories based on sales volume.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sales data
    entity_column : str
        Column name containing the entity to classify (family, subfamily, or sku)
    value_column : str, default="sales_value"
        Column name containing sales values
    a_threshold : float, default=0.8
        Threshold for class A (0-80% of total sales)
    b_threshold : float, default=0.95
        Threshold for class B (80-95% of total sales)
    year : int, default=2024
        The year to filter data by for classification
    date_column : str, default="invoice_date"
        Column name containing dates
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with entities and their ABC classification
    """
    if df.empty or entity_column not in df.columns:
        return pd.DataFrame(columns=[entity_column, 'total_sales', 'percent_of_total', 'cumulative_percent', 'abc_class'])
    
    # Ensure date column is datetime
    df_copy = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
        df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
    
    # Filter data for the specified year
    year_data = df_copy[df_copy[date_column].dt.year == year]
    
    # Get active entities in the specified year
    active_entities = filter_active_entities(df, entity_column, year, date_column)
    
    # If no entities are active in the specified year, return empty DataFrame
    if not active_entities:
        return pd.DataFrame(columns=[entity_column, 'total_sales', 'percent_of_total', 'cumulative_percent', 'abc_class'])
    
    # Calculate total sales by entity
    entity_sales = year_data.groupby(entity_column)[value_column].sum().reset_index()
    entity_sales.columns = [entity_column, 'total_sales']
    
    # Sort by total sales in descending order
    entity_sales = entity_sales.sort_values('total_sales', ascending=False)
    
    # Calculate percentages
    total = entity_sales['total_sales'].sum()
    entity_sales['percent_of_total'] = entity_sales['total_sales'] / total
    
    # Calculate cumulative percentage
    entity_sales['cumulative_percent'] = entity_sales['percent_of_total'].cumsum()
    
    # Classify into A, B, C
    conditions = [
        (entity_sales['cumulative_percent'] <= a_threshold),
        (entity_sales['cumulative_percent'] <= b_threshold),
        (entity_sales['cumulative_percent'] > b_threshold)
    ]
    choices = ['A', 'B', 'C']
    entity_sales['abc_class'] = np.select(conditions, choices, default='C')
    
    return entity_sales

def classify_entity_xyz(
    df: pd.DataFrame,
    entity_column: str,
    value_column: str = "sales_value", 
    date_column: str = "invoice_date",
    x_threshold: float = 0.2,
    y_threshold: float = 0.5,
    year: int = 2024
) -> pd.DataFrame:
    """
    Classify entities (families, subfamilies, or SKUs) into X, Y, Z categories based on coefficient of variation.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sales data
    entity_column : str
        Column name containing the entity to classify (family, subfamily, or sku)
    value_column : str, default="sales_value"
        Column name containing sales values
    date_column : str, default="invoice_date"
        Column name containing dates
    x_threshold : float, default=0.2
        Threshold for class X (CV 0-20%)
    y_threshold : float, default=0.5
        Threshold for class Y (CV 20-50%)
    year : int, default=2024
        The year to filter data by for classification
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with entities and their XYZ classification
    """
    if df.empty or entity_column not in df.columns:
        return pd.DataFrame(columns=[entity_column, 'mean_sales', 'std_sales', 'cv', 'xyz_class'])
    
    # Ensure date column is datetime
    df_copy = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_column]):
        df_copy[date_column] = pd.to_datetime(df_copy[date_column], errors='coerce')
    
    # Filter data for the specified year
    year_data = df_copy[df_copy[date_column].dt.year == year]
    
    # Get active entities in the specified year
    active_entities = filter_active_entities(df, entity_column, year, date_column)
    
    # If no entities are active in the specified year, return empty DataFrame
    if not active_entities:
        return pd.DataFrame(columns=[entity_column, 'mean_sales', 'std_sales', 'cv', 'xyz_class'])
    
    # Create a date period (month) for aggregation
    year_data['period'] = year_data[date_column].dt.to_period('M')
    
    # Calculate sales by entity and period
    entity_period_sales = year_data.groupby([entity_column, 'period'])[value_column].sum().reset_index()
    
    # Criar todos os 12 meses do ano especificado
    all_months = [pd.Period(f"{year}-{month}", freq='M') for month in range(1, 13)]
    
    # Calculate statistics for each entity
    entity_stats = []
    for entity in active_entities:
        # Converter valor para o mesmo tipo da coluna no DataFrame
        if entity_period_sales[entity_column].dtype != object:
            try:
                entity_value = type(entity_period_sales[entity_column].iloc[0])(entity) if not entity_period_sales.empty else entity
            except (ValueError, TypeError):
                entity_value = entity
        else:
            entity_value = entity
            
        # Filtrar dados apenas para esta entidade
        entity_data = entity_period_sales[entity_period_sales[entity_column] == entity_value]
        
        # Criar um DataFrame com todos os meses e preencher com vendas (ou zero para meses sem vendas)
        monthly_sales = pd.DataFrame({'period': all_months})
        monthly_sales = monthly_sales.merge(
            entity_data[['period', value_column]], 
            on='period', 
            how='left'
        ).fillna(0)
        
        if len(monthly_sales) > 0:  # Garantir que temos dados
            # Calcular estatísticas com todos os 12 meses, incluindo zeros
            sales_values = monthly_sales[value_column].to_numpy()  # Converter para NumPy array
            mean_sales = np.mean(sales_values)
            std_sales = np.std(sales_values, ddof=1)  # usando ddof=1 para STDEV.S (amostra)
            
            # Coefficient of variation
            cv = std_sales / mean_sales if mean_sales > 0 else float('inf')
            
            # Classify
            if cv <= x_threshold:
                xyz_class = 'X'
            elif cv <= y_threshold:
                xyz_class = 'Y'
            else:
                xyz_class = 'Z'
                
            entity_stats.append({
                entity_column: entity,  # Keep as string for consistency
                'mean_sales': mean_sales,
                'std_sales': std_sales,
                'cv': cv,
                'xyz_class': xyz_class
            })
        else:
            # Not enough data for coefficient of variation
            entity_stats.append({
                entity_column: entity,  # Keep as string for consistency
                'mean_sales': 0,
                'std_sales': 0,
                'cv': float('inf'),
                'xyz_class': 'Z'  # Classify as Z if not enough data
            })
    
    return pd.DataFrame(entity_stats)

def entity_abc_xyz_classification(
    df: pd.DataFrame,
    entity_column: str,
    value_column: str = "sales_value",
    date_column: str = "invoice_date",
    year: int = 2024
) -> pd.DataFrame:
    """
    Perform both ABC and XYZ classification on entities (families, subfamilies, or SKUs).
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sales data
    entity_column : str
        Column name containing the entity to classify (family, subfamily, or sku)
    value_column : str, default="sales_value"
        Column name containing sales values
    date_column : str, default="invoice_date"
        Column name containing dates
    year : int, default=2024
        The year to filter data by for classification
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with entities and their combined ABC-XYZ classification, sorted by total_sales descending
    """
    if entity_column not in df.columns:
        raise ValueError(f"Column '{entity_column}' not found in the dataframe. Available columns: {df.columns.tolist()}")
        
    # Get active entities in the specified year
    active_entities = filter_active_entities(df, entity_column, year, date_column)
    
    # If no entities are active in the specified year, return empty DataFrame
    if not active_entities:
        return pd.DataFrame(columns=[entity_column, 'abc_class', 'xyz_class', 'abc_xyz_class'])
    
    # Perform ABC classification
    abc_df = classify_entity_abc(df, entity_column, value_column, year=year, date_column=date_column)
    
    # Perform XYZ classification
    xyz_df = classify_entity_xyz(df, entity_column, value_column, date_column, year=year)
    
    # Ensure both dataframes have the same type for the entity column
    abc_df[entity_column] = abc_df[entity_column].astype(str)
    xyz_df[entity_column] = xyz_df[entity_column].astype(str)
    
    # Merge the results
    combined = abc_df.merge(xyz_df, on=entity_column, how='outer')
    
    # Create combined classification
    combined['abc_xyz_class'] = combined['abc_class'] + combined['xyz_class']
    
    # Sort by total_sales (descending) to show most important entities first
    if 'total_sales' in combined.columns:
        combined = combined.sort_values('total_sales', ascending=False)
    
    return combined

def create_entity_abc_xyz_matrix(df: pd.DataFrame, entity_type: str) -> Tuple[go.Figure, Dict[str, int]]:
    """
    Create a visual matrix of ABC-XYZ classification for the specified entity type.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with ABC-XYZ classification
    entity_type : str
        The type of entity being analyzed (e.g., "Família", "Subfamília", "SKU")
        
    Returns:
    --------
    Tuple[go.Figure, Dict[str, int]]
        A tuple containing the Plotly figure and a dictionary with counts for each category
    """
    # Create a classification matrix
    classes = ['A', 'B', 'C']
    variations = ['X', 'Y', 'Z']
    
    # Initialize the matrix with zeros
    matrix = np.zeros((len(classes), len(variations)), dtype=int)
    counts = {}
    
    # Fill the matrix with counts
    if 'abc_class' in df.columns and 'xyz_class' in df.columns:
        for i, cls in enumerate(classes):
            for j, var in enumerate(variations):
                category = f"{cls}{var}"
                count = len(df[(df['abc_class'] == cls) & (df['xyz_class'] == var)])
                matrix[i, j] = count
                counts[category] = count
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=variations,
        y=classes,
        text=matrix.astype(str),
        texttemplate="%{text}",
        textfont={"size": 20},
        colorscale='Viridis',
        showscale=False
    ))
    
    # Update layout
    fig.update_layout(
        title=f'ABC-XYZ Classification Matrix - {entity_type} (2024 Data)',
        xaxis_title='Coefficient of Variation (XYZ)',
        yaxis_title='Sales Volume (ABC)',
        xaxis=dict(side='top'),
        width=600,
        height=500,
        margin=dict(l=50, r=50, t=100, b=50),
    )
    
    # Add annotations
    annotations = []
    
    # X-axis annotations
    annotations.append(dict(
        x=0, y=1.15, 
        text="X: CV ≤ 20%<br>Stable demand",
        showarrow=False,
        xref="x", yref="paper",
        font=dict(size=10)
    ))
    annotations.append(dict(
        x=1, y=1.15, 
        text="Y: CV 20-50%<br>Variable demand",
        showarrow=False,
        xref="x", yref="paper",
        font=dict(size=10)
    ))
    annotations.append(dict(
        x=2, y=1.15, 
        text="Z: CV > 50%<br>Highly variable",
        showarrow=False,
        xref="x", yref="paper",
        font=dict(size=10)
    ))
    
    # Y-axis annotations
    annotations.append(dict(
        x=-0.15, y=0, 
        text="A: Top 80% sales",
        showarrow=False,
        xref="paper", yref="y",
        font=dict(size=10)
    ))
    annotations.append(dict(
        x=-0.15, y=1, 
        text="B: Next 15% sales",
        showarrow=False,
        xref="paper", yref="y",
        font=dict(size=10)
    ))
    annotations.append(dict(
        x=-0.15, y=2, 
        text="C: Last 5% sales",
        showarrow=False,
        xref="paper", yref="y",
        font=dict(size=10)
    ))
    
    fig.update_layout(annotations=annotations)
    
    return fig, counts 
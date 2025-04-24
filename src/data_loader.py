"""
Module for loading and cleaning data from Excel files.
"""

import pandas as pd
from typing import List, Optional, Union, Dict

def load_sales_data(
    file_path: str,
    sheet_name: Optional[str] = "BD_Vendas",
    filtered_families: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Reads an Excel file containing SKU metadata and sales data, and returns a cleaned DataFrame.
    
    Parameters:
    -----------
    file_path : str
        Path to the Excel file
    sheet_name : str or int, optional
        Name or index of the sheet to read (default: "BD_Vendas")
    filtered_families : List[str], optional
        List of 'Familia' values to filter by. If None, all families are included.
        
    Returns:
    --------
    pd.DataFrame
        Cleaned DataFrame with sales data
    """
    # Read Excel file
    print(f"Reading Excel file: {file_path}")
    excel_data = pd.read_excel(file_path, sheet_name=sheet_name)
    
    # Handle the case where a dictionary of DataFrames is returned
    if isinstance(excel_data, dict):
        if sheet_name in excel_data:
            df = excel_data[sheet_name]
        else:
            # If the requested sheet isn't found, use the first sheet
            sheet_key = list(excel_data.keys())[0]
            df = excel_data[sheet_key]
            print(f"Warning: Sheet '{sheet_name}' not found. Using '{sheet_key}' instead.")
    else:
        # In this case, excel_data is already a DataFrame
        df = excel_data
    
    print(f"Columns found in the file: {df.columns.tolist()}")
    
    # Converter a coluna de data para datetime
    df['Data do faturamento'] = pd.to_datetime(df['Data do faturamento'], dayfirst=True)
    
    # Renomear colunas para manter compatibilidade com o resto do código
    column_mapping = {
        'Data do faturamento': 'invoice_date',
        'Familia': 'family',
        'Sub Familia': 'subfamily',
        'Material Mapeado': 'sku',
        'Valor Kg': 'sales_value',
        'Gestor Comercial': 'commercial_manager',
        'Gramagem': 'weight',
        'Formato': 'format'
    }
    
    # Renomear apenas as colunas que existem
    existing_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
    df = df.rename(columns=existing_columns)
    
    # Filtrar por famílias específicas se fornecido
    if filtered_families is not None:
        df = df[df['family'].isin(filtered_families)]
    
    print(f"Processed data: {len(df)} rows remaining")
    
    return df

def get_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract metadata information from the sales DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sales data
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with unique SKU metadata
    """
    # Select only metadata columns and drop duplicates based on SKU
    metadata_columns = [col for col in ["commercial_manager", "family", "subfamily", 
                                       "weight", "format", "sku"] 
                        if col in df.columns]
    
    metadata_df = df[metadata_columns].drop_duplicates(subset=["sku"])
    
    return metadata_df 

def load_budget_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Loads and processes budget data from an Excel file.

    Parameters:
    ----------
    file_path : str
        Path to the budget Excel file (e.g., 'data/Orçamento2022-2025.xlsx').

    Returns:
    --------
    pd.DataFrame or None
        A DataFrame containing the aggregated monthly budget data by 
        Gestor Comercial, Familia, year, and month. Returns None if the
        file is not found or required columns are missing.
    """
    try:
        print(f"Reading budget file: {file_path}")
        df_budget = pd.read_excel(file_path)
        print(f"Budget columns found: {df_budget.columns.tolist()}")

        # Required columns check
        required_cols = ['Cod. Mapeados', 'Segmento', 'Gestor Comercial', 'KG', 'Ano_Mês']
        if not all(col in df_budget.columns for col in required_cols):
            print(f"Error: Budget file missing one or more required columns: {required_cols}")
            missing = [col for col in required_cols if col not in df_budget.columns]
            print(f"Missing columns: {missing}")
            return None

        # Rename columns for consistency and processing
        df_budget.rename(columns={'Segmento': 'Familia', 'Cod. Mapeados': 'sku'}, inplace=True)

        # Map 'MP' to 'MDD' in 'Gestor Comercial'
        df_budget.loc[df_budget['Gestor Comercial'] == 'MP', 'Gestor Comercial'] = 'MDD'

        # Process 'Ano_Mês'
        df_budget['Ano_Mês'] = pd.to_datetime(df_budget['Ano_Mês'], format='%m-%Y')
        df_budget['year'] = df_budget['Ano_Mês'].dt.year
        df_budget['month'] = df_budget['Ano_Mês'].dt.month
        
        # Aggregate budget data
        agg_cols = ['Gestor Comercial', 'Familia', 'year', 'month']
        budget_aggregated = df_budget.groupby(agg_cols)['KG'].sum().reset_index()
        budget_aggregated.rename(columns={'Gestor Comercial': 'commercial_manager', 'Familia': 'family'}, inplace=True)


        print(f"Processed budget data: {len(budget_aggregated)} rows")
        return budget_aggregated

    except FileNotFoundError:
        print(f"Error: Budget file not found at {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred while processing the budget file: {e}")
        return None 
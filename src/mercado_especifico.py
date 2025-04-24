import pandas as pd
import os
from typing import List, Optional

def load_sales_data(
    file_path: str,
    sheet_name: Optional[str] = "BD_Vendas"
) -> pd.DataFrame:
    """
    Versão simplificada da função original em data_loader.py
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
    
    # Determine which columns we have
    # We'll look for columns containing key terms like "Gestor", "Familia", etc.
    column_mapping = {}
    
    # Check for common patterns in column names
    for col in df.columns:
        col_lower = str(col).lower()
        
        if any(term in col_lower for term in ["gestor", "comercial", "manager"]):
            column_mapping["commercial_manager"] = col
        elif any(term in col_lower for term in ["familia", "family"]) and "sub" not in col_lower:
            column_mapping["family"] = col
        elif any(term in col_lower for term in ["sub familia", "sub family", "subfamily"]):
            column_mapping["subfamily"] = col
        elif any(term in col_lower for term in ["gramagem", "weight"]):
            column_mapping["weight"] = col
        elif any(term in col_lower for term in ["formato", "format"]):
            column_mapping["format"] = col
        elif any(term in col_lower for term in ["material", "mapeado", "sku"]):
            column_mapping["sku"] = col
        elif any(term in col_lower for term in ["valor", "kg", "value", "sales"]):
            column_mapping["sales_value"] = col
        elif any(term in col_lower for term in ["data", "faturamento", "date", "invoice"]):
            column_mapping["invoice_date"] = col
    
    # Print found mappings
    print("Found column mappings:")
    for target, source in column_mapping.items():
        print(f"  {target} <- {source}")
    
    # Check if we have the minimum required columns
    required_columns = ["family", "sku", "sales_value", "invoice_date"]
    missing_columns = [col for col in required_columns if col not in column_mapping]
    
    if missing_columns:
        # If we're missing required columns, try to make some educated guesses
        print(f"Warning: Missing required columns: {missing_columns}")
        print("Trying to make educated guesses about column names...")
        
        # If we're missing family, try to use the first text column
        if "family" in missing_columns:
            text_cols = [col for col in df.columns if df[col].dtype == 'object']
            if text_cols:
                column_mapping["family"] = text_cols[0]
                print(f"  Guessing 'family' might be: {text_cols[0]}")
        
        # If we're missing sku, try to use a column with unique values
        if "sku" in missing_columns:
            for col in df.columns:
                if df[col].nunique() > df.shape[0] * 0.5:  # More than 50% unique values
                    column_mapping["sku"] = col
                    print(f"  Guessing 'sku' might be: {col}")
                    break
        
        # If we're missing sales_value, try to use a numeric column
        if "sales_value" in missing_columns:
            numeric_cols = [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]
            if numeric_cols:
                column_mapping["sales_value"] = numeric_cols[0]
                print(f"  Guessing 'sales_value' might be: {numeric_cols[0]}")
        
        # If we're missing invoice_date, try to use a date column
        if "invoice_date" in missing_columns:
            date_cols = []
            for col in df.columns:
                try:
                    pd.to_datetime(df[col], errors='raise')
                    date_cols.append(col)
                except:
                    pass
            
            if date_cols:
                column_mapping["invoice_date"] = date_cols[0]
                print(f"  Guessing 'invoice_date' might be: {date_cols[0]}")
    
    # Check again if we have all required columns
    missing_columns = [col for col in required_columns if col not in column_mapping]
    if missing_columns:
        raise ValueError(f"Could not identify these required columns: {missing_columns}")
    
    # Create a view with renamed columns
    renamed_df = df.rename(columns={v: k for k, v in column_mapping.items()})
    
    # Convert invoice_date to datetime
    if "invoice_date" in column_mapping:
        renamed_df["invoice_date"] = pd.to_datetime(renamed_df["invoice_date"], errors="coerce")
    
    # Drop rows with missing important values
    renamed_df = renamed_df.dropna(subset=["sku", "sales_value", "invoice_date"])
    
    # Convert numerical columns
    renamed_df["sales_value"] = pd.to_numeric(renamed_df["sales_value"], errors="coerce")
    
    # Select only the columns we're interested in
    columns_to_keep = [col for col in ["commercial_manager", "family", "subfamily", 
                                     "weight", "format", "sku", "sales_value", "invoice_date"] 
                      if col in renamed_df.columns]
    
    result_df = renamed_df[columns_to_keep]
    
    print(f"Processed data: {len(result_df)} rows remaining")
    
    return result_df

def get_mercado_especifico_skus() -> List[str]:
    """
    Identifica e retorna a lista de SKUs que pertencem ao mercado específico.
    Um SKU pertence ao mercado específico se:
    1. Contém "|" na coluna TxtBreveMaterial no arquivo BD_ATUALIZACAO.xlsx
    2. Tem vendas em 2024 no arquivo 2022-2025.xlsx
    3. Pertence a uma das 15 famílias específicas
    
    Returns:
        List[str]: Lista de SKUs do mercado específico
    """
    # Verificar se já temos os skus cacheados em um arquivo
    cache_file = 'data/skus_mercado_especifico.xlsx'
    if os.path.exists(cache_file):
        try:
            df = pd.read_excel(cache_file)
            # Garantir que os SKUs são retornados como strings
            return df['SKU'].astype(str).tolist()
        except Exception as e:
            print(f"Erro ao ler cache: {e}")
            # Se falhar ao ler o cache, continuamos com a extração normal
    
    # Retornar resultado da extração
    skus = extrair_skus_mercado_especifico()
    
    # Garantir que todos os SKUs são strings antes de retornar
    return [str(sku) for sku in skus]

def extrair_skus_mercado_especifico():
    """
    Extrai SKUs que atendem a 3 condições:
    1. Contêm '|' na coluna TxtBreveMaterial do arquivo BD_ATUALIZACAO.xlsx
    2. Têm vendas em 2024 no arquivo 2022-2025.xlsx
    3. Pertencem às 15 famílias específicas
    
    Returns:
        list: Lista de SKUs que atendem a todas as condições
    """
    # Lista de famílias a considerar
    familias = [
        'Cream Cracker', 'Maria', 'Wafer', 'Sortido', 'Cobertas de Chocolate',
        'Água e Sal', 'Digestiva', 'Recheada', 'Circus', 'Tartelete',
        'Torrada', 'Flocos de Neve', 'Integral', 'Mentol', 'Aliança'
    ]

    print('Iniciando extração de SKUs para mercado específico...')

    # 1. Ler SKUs com | no TxtBreveMaterial do arquivo BD_ATUALIZACAO.xlsx
    print('1. Lendo BD_ATUALIZACAO.xlsx...')
    skus_info = pd.read_excel('data/BD_ATUALIZACAO.xlsx', sheet_name='SKUs Inf.Geral')
    skus_com_pipe = skus_info[skus_info['TxtBreveMaterial'].str.contains('|', regex=False)]
    print(f'Encontrados {len(skus_com_pipe)} SKUs com | no TxtBreveMaterial')

    # 2. Ler dados de vendas do arquivo 2022-2025.xlsx usando a função adequada
    print('2. Lendo 2022-2025.xlsx para obter vendas...')
    try:
        vendas = load_sales_data('data/2022-2025.xlsx', sheet_name="BD_Vendas")
        print(f'Total de registros de vendas: {len(vendas)}')
    except Exception as e:
        print(f"Erro ao carregar dados de vendas: {e}")
        return []

    # 3. Filtrar para obter apenas vendas de 2024
    print('3. Filtrando vendas de 2024...')
    vendas['Ano'] = vendas['invoice_date'].dt.year
    vendas_2024 = vendas[vendas['Ano'] == 2024]
    print(f'Registros de vendas em 2024: {len(vendas_2024)}')

    # 4. Filtrar para obter apenas vendas das 15 famílias
    print('4. Filtrando para as 15 famílias...')
    vendas_familias = vendas_2024[vendas_2024['family'].isin(familias)]
    print(f'Registros de vendas em 2024 para as 15 famílias: {len(vendas_familias)}')

    # 5. Obter SKUs únicos das 15 famílias com vendas em 2024
    skus_familias_com_vendas = vendas_familias['sku'].unique()
    print(f'SKUs únicos das 15 famílias com vendas em 2024: {len(skus_familias_com_vendas)}')

    # 6. Obter a interseção: SKUs com | que pertencem às 15 famílias e têm vendas em 2024
    print('5. Calculando a interseção dos conjuntos...')
    skus_com_pipe_set = set(skus_com_pipe['Material'].astype(str))
    skus_vendas_set = set(map(str, skus_familias_com_vendas))

    skus_resultado = list(skus_com_pipe_set.intersection(skus_vendas_set))
    print(f'RESULTADO: {len(skus_resultado)} SKUs atendem a todas as condições')

    # 7. Criar um DataFrame com os resultados detalhados
    if len(skus_resultado) > 0:
        resultados = []
        for sku in skus_resultado:
            info = skus_com_pipe[skus_com_pipe['Material'].astype(str) == sku]
            if not info.empty:
                sku_rows = vendas_familias[vendas_familias['sku'].astype(str) == sku]
                familia_info = sku_rows['family'].iloc[0] if not sku_rows.empty else "Desconhecida"
                resultados.append({
                    'SKU': sku,
                    'Descrição': info.iloc[0]['TxtBreveMaterial'],
                    'Família': familia_info
                })
        
        resultados_df = pd.DataFrame(resultados)
        
        print('\nPrimeiros 10 SKUs que atendem a todas as condições:')
        if not resultados_df.empty:
            print(resultados_df.head(10))
            
            # Salvar resultados em Excel
            output_path = 'data/skus_mercado_especifico.xlsx'
            resultados_df.to_excel(output_path, index=False)
            print(f'\nResultados salvos em: {output_path}')
        else:
            print('Nenhum registro detalhado encontrado para os SKUs que atendem as condições')
    else:
        print('Nenhum SKU atende a todas as condições')
    
    return skus_resultado

if __name__ == "__main__":
    extrair_skus_mercado_especifico() 
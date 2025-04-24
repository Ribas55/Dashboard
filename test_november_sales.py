import pandas as pd
import os

def extract_november_2023_sales():
    """
    Extrai o total de vendas de cada família no mês de novembro de 2023.
    """
    # Caminho do arquivo Excel
    excel_file = os.path.join('data', '2022-2025.xlsx')
    
    # Verificar se o arquivo existe
    if not os.path.exists(excel_file):
        print(f"Erro: O arquivo {excel_file} não foi encontrado.")
        return
    
    # Carregar os dados diretamente do Excel
    print(f"Lendo arquivo Excel: {excel_file}")
    # Ler o Excel especificando a aba BD_Vendas
    df = pd.read_excel(excel_file, sheet_name='BD_Vendas')
    
    # Mostrar as colunas disponíveis
    print("\nColunas disponíveis no arquivo:")
    print(df.columns.tolist())
    
    # Converter a coluna de data para datetime
    df['Data do faturamento'] = pd.to_datetime(df['Data do faturamento'], dayfirst=True)
    
    # Filtrar para novembro de 2023
    november_2023 = df[
        (df['Data do faturamento'].dt.year == 2023) & 
        (df['Data do faturamento'].dt.month == 11)
    ]
    
    # Agrupar por família e calcular o total de vendas
    sales_by_family = november_2023.groupby('Familia')['Valor Kg'].sum().sort_values(ascending=False)
    
    # Formatar e exibir os resultados
    print("\nTotal de Vendas por Família - Novembro 2023")
    print("===========================================")
    for family, sales in sales_by_family.items():
        print(f"{family}: {sales:,.2f} Kg")
    
    # Exibir o total geral
    total_sales = sales_by_family.sum()
    print("\nTotal Geral:")
    print(f"Total: {total_sales:,.2f} Kg")

if __name__ == "__main__":
    extract_november_2023_sales() 
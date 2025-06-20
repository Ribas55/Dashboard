import os
import pandas as pd
import plotly.graph_objects as go
from src.data_loader import load_sales_data
from src.abc_xyz_analysis import get_abc_xyz_data, create_quadrant_chart
from src.mercado_especifico import get_mercado_especifico_skus

OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALLOWED_FAMILIES = [
    'Cream Cracker', 'Maria', 'Wafer', 'Sortido', 'Cobertas de Chocolate',
    'Água e Sal', 'Digestiva', 'Recheada', 'Circus', 'Tartelete',
    'Torrada', 'Flocos de Neve', 'Integral', 'Mentol', 'Aliança'
]

def filter_2024_skus(df):
    # Ensure SKU column is string type
    df = df.copy()
    df['sku'] = df['sku'].astype(str)
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    df_2024 = df[df['invoice_date'].dt.year == 2024]
    skus_2024 = set(df_2024[df_2024['sales_value'] > 0]['sku'].unique())
    return df[df['sku'].isin(skus_2024)]

def create_custom_abc_xyz_matrix(df, title):
    """Create a proper ABC/XYZ matrix like the reference image"""
    
    # Calculate matrix data
    classes = ['A', 'B', 'C']
    variations = ['X', 'Y', 'Z']
    
    # Initialize matrices
    count_matrix = {}
    percentage_matrix = {}
    
    # Total sales
    total_sales = df['total_sales'].sum() if 'total_sales' in df.columns else 0
    
    # Fill the matrices
    for cls in classes:
        for var in variations:
            category = f"{cls}{var}"
            category_df = df[(df['abc_class'] == cls) & (df['xyz_class'] == var)]
            
            count = len(category_df)
            sales_sum = category_df['total_sales'].sum() if 'total_sales' in category_df.columns else 0
            sales_percentage = (sales_sum / total_sales * 100) if total_sales > 0 else 0
            
            count_matrix[category] = count
            percentage_matrix[category] = sales_percentage
    
    # Colors for each row
    colors = {
        'A': '#82E0AA',  # Green
        'B': '#F7DC6F',  # Yellow  
        'C': '#EC7063'   # Red
    }
    
    # Prepare table data
    header_row = ['', 'X (<20%)', 'Y (20-50%)', 'Z (>50%)']
    
    # Create data for each column
    col1 = ['A (80%)', 'B (15%)', 'C (5%)']  # Row headers
    col2 = []  # X column
    col3 = []  # Y column  
    col4 = []  # Z column
    
    for cls in classes:
        for i, var in enumerate(variations):
            category = f"{cls}{var}"
            count = count_matrix[category]
            percentage = percentage_matrix[category]
            cell_text = f"{count} SKUs<br>{percentage:.1f}% of sales"
            
            if i == 0:  # X column
                col2.append(cell_text)
            elif i == 1:  # Y column
                col3.append(cell_text)
            else:  # Z column
                col4.append(cell_text)
    
    # Create fill colors for each column
    header_color = '#D3D3D3'
    fill_colors = [
        [header_color] + [colors[cls] for cls in classes],  # Column 1 (row headers)
        [header_color] + [colors[cls] for cls in classes],  # Column 2 (X)
        [header_color] + [colors[cls] for cls in classes],  # Column 3 (Y)
        [header_color] + [colors[cls] for cls in classes]   # Column 4 (Z)
    ]
    
    # Create the table
    fig = go.Figure(data=[go.Table(
        header=dict(
            values=header_row,
            fill_color=header_color,
            align='center',
            font=dict(size=14, color='black', family="Arial"),
            height=50,
            line=dict(color='black', width=2)
        ),
        cells=dict(
            values=[col1, col2, col3, col4],
            fill_color=fill_colors,
            align='center',
            font=dict(size=12, color='black', family="Arial"),
            height=80,
            line=dict(color='black', width=2)
        )
    )])
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=16, color='black', family="Arial")
        ),
        paper_bgcolor='white',
        plot_bgcolor='white',
        margin=dict(l=50, r=50, t=80, b=50),
        height=400,
        width=800
    )
    
    return fig

def english_bar(fig):
    fig.update_layout(
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(color='black', family="Arial, sans-serif"),
        xaxis_title='Number of SKUs',
        yaxis_title='ABC/XYZ Quadrant',
    )
    return fig

def run_abcxyz_analysis(df, mercado_especifico_skus, group_name, file_prefix):
    # Ensure SKU column is string type
    df = df.copy()
    df['sku'] = df['sku'].astype(str)
    
    if group_name == 'Normal':
        group_df = df[~df['sku'].isin(mercado_especifico_skus)]
    elif group_name == 'Specific Market':
        group_df = df[df['sku'].isin(mercado_especifico_skus)]
    else:
        group_df = df.copy()
    
    print(f"{group_name}: {len(group_df['sku'].unique())} SKUs with sales in 2024")
    
    if group_df.empty:
        print(f"No data for {group_name}")
        return

    # Run ABC/XYZ analysis
    abcxyz_df = get_abc_xyz_data(
        df=group_df,
        family_filter=None,
        subfamily_filter=None,
        market_type='Todos',
        mercado_especifico_skus=mercado_especifico_skus,
        a_threshold=0.8, b_threshold=0.95, x_threshold=0.2, y_threshold=0.5
    )

    if abcxyz_df.empty:
        print(f"No ABC/XYZ data for {group_name}")
        return

    print(f"ABC/XYZ analysis complete for {group_name}: {len(abcxyz_df)} SKUs classified")

    # Matrix - use custom function
    matrix_title = f"ABC/XYZ Classification Matrix ({group_name})"
    matrix_fig = create_custom_abc_xyz_matrix(abcxyz_df, matrix_title)
    matrix_fig.write_image(f'{OUTPUT_DIR}/abcxyz_matrix_{file_prefix}.png', scale=2)
    print(f"Saved matrix for {group_name} to {OUTPUT_DIR}/abcxyz_matrix_{file_prefix}.png")

    # Bar chart
    bar_fig = create_quadrant_chart(abcxyz_df)
    bar_fig = english_bar(bar_fig)
    bar_fig.write_image(f'{OUTPUT_DIR}/abcxyz_bar_{file_prefix}.png', scale=2)
    print(f"Saved bar chart for {group_name} to {OUTPUT_DIR}/abcxyz_bar_{file_prefix}.png")

    # Save details per quadrant
    with pd.ExcelWriter(f'{OUTPUT_DIR}/abcxyz_details_{file_prefix}.xlsx') as writer:
        for quadrant in ['AX','AY','AZ','BX','BY','BZ','CX','CY','CZ']:
            quad_df = abcxyz_df[abcxyz_df['abc_xyz_class'] == quadrant]
            if not quad_df.empty:
                quad_df.to_excel(writer, sheet_name=quadrant, index=False)
    print(f"Saved details for {group_name} to {OUTPUT_DIR}/abcxyz_details_{file_prefix}.xlsx")

def main():
    # Load sales data (only 15 families)
    sales_df = load_sales_data('data/2022-2025.xlsx')
    sales_df = sales_df[sales_df['family'].isin(ALLOWED_FAMILIES)]
    
    # Filter SKUs with at least one sale in 2024
    sales_df = filter_2024_skus(sales_df)
    print(f"Number of SKUs with sales in 2024: {len(sales_df['sku'].unique())}")
    
    # Get mercado específico SKUs (as strings)
    mercado_especifico_skus = [str(sku) for sku in get_mercado_especifico_skus()]
    print(f"Loaded {len(mercado_especifico_skus)} Mercado Específico SKUs from file.")
    
    # Mercado Específico SKUs with sales in 2024
    skus_2024 = set(sales_df['sku'].unique())
    skus_mercado_especifico_2024 = set(mercado_especifico_skus) & skus_2024
    print(f"Number of Mercado Específico SKUs with sales in 2024: {len(skus_mercado_especifico_2024)}")

    # Run analysis for each group
    for group, prefix in [('Normal', 'normal'), ('Specific Market', 'specific'), ('All', 'all')]:
        print(f"\n--- Running analysis for {group} ---")
        run_abcxyz_analysis(sales_df, mercado_especifico_skus, group, prefix)

    print("\nAnalysis complete. Check the 'output' folder for results.")

if __name__ == "__main__":
    main() 
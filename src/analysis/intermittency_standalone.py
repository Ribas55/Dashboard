import os
import pandas as pd
from src.data_loader import load_sales_data
from src.mercado_especifico import get_mercado_especifico_skus
from src.intermittency_analysis import get_intermittency_data, create_intermittency_matrix, create_quadrant_chart
import plotly.graph_objects as go
from datetime import datetime

# Output directory
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 15 allowed families
ALLOWED_FAMILIES = [
    'Cream Cracker', 'Maria', 'Wafer', 'Sortido', 'Cobertas de Chocolate',
    'Água e Sal', 'Digestiva', 'Recheada', 'Circus', 'Tartelete',
    'Torrada', 'Flocos de Neve', 'Integral', 'Mentol', 'Aliança'
]

# English quadrant/category names
EN_CATEGORIES = ["Smooth", "Intermittent", "Erratic", "Lumpy"]
EN_LABELS = {
    "sku": "SKU",
    "family": "Family",
    "subfamily": "Subfamily",
    "total_sales": "Sales (kg)",
    "cv2": "CV²",
    "adi": "ADI",
    "category": "Category"
}
CATEGORY_COLORS = {
    "Smooth": "green",
    "Intermittent": "blue",
    "Erratic": "yellow",
    "Lumpy": "red"
}

# Load sales data (only 15 families)
sales_df = load_sales_data('data/2022-2025.xlsx')
sales_df = sales_df[sales_df['family'].isin(ALLOWED_FAMILIES)]

# Filter SKUs with at least one sale in 2024
sales_df['invoice_date'] = pd.to_datetime(sales_df['invoice_date'])
sales_2024 = sales_df[sales_df['invoice_date'].dt.year == 2024]
skus_2024 = set(sales_2024[sales_2024['sales_value'] > 0]['sku'].unique())
print(f"Number of SKUs with sales in 2024: {len(skus_2024)}")
sales_df = sales_df[sales_df['sku'].isin(skus_2024)]

# Get mercado específico SKUs (as strings)
mercado_especifico_skus = [str(sku) for sku in get_mercado_especifico_skus()]
print(f"Loaded {len(mercado_especifico_skus)} Mercado Específico SKUs from file.")

# Ensure all SKUs in sales_df are strings
sales_df['sku'] = sales_df['sku'].astype(str)
skus_2024 = set(sales_df['sku'].unique())

# Mercado Específico SKUs with sales in 2024
skus_mercado_especifico_2024 = set(mercado_especifico_skus) & skus_2024
print(f"Number of Mercado Específico SKUs with sales in 2024: {len(skus_mercado_especifico_2024)}")

# Print samples for debugging
print("Sample Mercado Específico SKUs:", list(mercado_especifico_skus)[:5])
print("Sample SKUs with sales in 2024:", list(skus_2024)[:5])
print("Sample intersection SKUs:", list(skus_mercado_especifico_2024)[:5])

# Analysis for both market types
def run_analysis(market_type, skus_filter, file_prefix, plot_title, bar_title):
    if market_type == 'Normal':
        filtered_df = sales_df[~sales_df['sku'].isin(mercado_especifico_skus)]
    else:
        filtered_df = sales_df[sales_df['sku'].isin(mercado_especifico_skus)]
    filtered_df = filtered_df[filtered_df['sku'].isin(skus_filter)]
    print(f"{market_type}: {len(filtered_df['sku'].unique())} SKUs with sales in 2024")
    if filtered_df.empty:
        print(f"No data for {market_type} SKUs with sales in 2024.")
        # Still generate an empty bar chart
        categories = ['Smooth','Intermittent','Erratic','Lumpy']
        counts = [0,0,0,0]
        fig = go.Figure([go.Bar(x=categories, y=counts, marker_color=['green','blue','gold','red'])])
        fig.update_layout(template='plotly_white', title=bar_title, xaxis_title='Category', yaxis_title='Number of SKUs')
        fig.write_image(f'{OUTPUT_DIR}/bar_quadrant_{file_prefix}.png', scale=2)
        return
    intermittency_df = get_intermittency_data(filtered_df, market_type=market_type, mercado_especifico_skus=mercado_especifico_skus)
    # Ensure correct columns
    required_cols = {'sku','family','subfamily','total_sales','cv2','adi','category'}
    if not required_cols.issubset(set(intermittency_df.columns)):
        print(f"Missing columns in intermittency_df: {set(intermittency_df.columns)}")
        return
    # Save details per quadrant
    with pd.ExcelWriter(f'{OUTPUT_DIR}/detalhes_quadrante_{file_prefix}.xlsx') as writer:
        for cat in ['Smooth','Intermittent','Erratic','Lumpy']:
            cat_df = intermittency_df[intermittency_df['category'] == cat]
            cat_df[['sku','family','subfamily','total_sales','cv2','adi','category']].to_excel(writer, sheet_name=cat, index=False)
    # Intermittency matrix plot
    fig, _ = create_intermittency_matrix(intermittency_df, custom_title=plot_title)
    fig.update_layout(template='plotly_white')
    fig.write_image(f'{OUTPUT_DIR}/intermittency_{file_prefix}.png', scale=2)
    # Bar chart
    # Count SKUs per category
    categories = ['Smooth','Intermittent','Erratic','Lumpy']
    counts = [len(intermittency_df[intermittency_df['category']==cat]) for cat in categories]
    bar_fig = go.Figure([go.Bar(x=categories, y=counts, marker_color=['green','blue','gold','red'])])
    bar_fig.update_layout(template='plotly_white', title=bar_title, xaxis_title='Category', yaxis_title='Number of SKUs')
    bar_fig.write_image(f'{OUTPUT_DIR}/bar_quadrant_{file_prefix}.png', scale=2)

# Run for Normal
run_analysis(
    market_type='Normal',
    skus_filter=skus_2024 - set(mercado_especifico_skus),
    file_prefix='normal',
    plot_title='Intermittency Classification Matrix (Normal)',
    bar_title='SKU Count per Category (Normal)'
)
# Run for Specific Market
run_analysis(
    market_type='Mercado Específico',
    skus_filter=skus_mercado_especifico_2024,
    file_prefix='mercado_especifico',
    plot_title='Intermittency Classification Matrix (Specific Market)',
    bar_title='SKU Count per Category (Specific Market)'
)

print("Analysis complete. Check the 'output' folder for results.") 
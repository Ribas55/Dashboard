"""
Módulo para análise de intermitência, classificando SKUs em Smooth, Intermittent, Erratic e Lumpy
com base nos valores de CV² (Coeficiente de Variação ao quadrado) e ADI (Average Demand Interval).
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from src.mercado_especifico import get_mercado_especifico_skus

def get_intermittency_data(df, family_filter=None, subfamily_filter=None, market_type="Todos", 
                         mercado_especifico_skus=None, cv2_threshold=0.49, adi_threshold=1.32):
    """
    Calcula os valores de CV² e ADI para cada SKU e classifica por quadrante.
    
    Args:
        df (DataFrame): DataFrame com os dados de vendas
        family_filter (str, optional): Filtro de família
        subfamily_filter (str, optional): Filtro de subfamília
        market_type (str, optional): Filtro de tipo de mercado ('Todos', 'Normal', 'Mercado Específico')
        mercado_especifico_skus (list, optional): Lista de SKUs de mercado específico
        cv2_threshold (float, optional): Limite de CV² para classificação
        adi_threshold (float, optional): Limite de ADI para classificação
        
    Returns:
        DataFrame: DataFrame com a classificação de intermitência
    """
    # Apply family and subfamily filters if provided
    filtered_df = df.copy()
    
    if family_filter and family_filter != "Todas":
        filtered_df = filtered_df[filtered_df["family"] == family_filter]
    
    if subfamily_filter and subfamily_filter != "Todas":
        filtered_df = filtered_df[filtered_df["subfamily"] == subfamily_filter]
    
    # Ensure that mercado_especifico_skus is a list
    if mercado_especifico_skus is None:
        mercado_especifico_skus = []
    
    # Apply market type filter if needed
    if market_type == "Normal":
        filtered_df = filtered_df[~filtered_df["sku"].isin(mercado_especifico_skus)]
    elif market_type == "Mercado Específico":
        filtered_df = filtered_df[filtered_df["sku"].isin(mercado_especifico_skus)]
    
    # Group by SKU and resample to monthly frequency to identify zero demand periods
    # Convert to datetime if not already
    if pd.api.types.is_datetime64_any_dtype(filtered_df["invoice_date"]) == False:
        filtered_df["invoice_date"] = pd.to_datetime(filtered_df["invoice_date"])
    
    # Sort by date
    filtered_df = filtered_df.sort_values("invoice_date")
    
    # Get unique SKUs, families and subfamilies for reference
    skus = filtered_df["sku"].unique()
    sku_family_map = filtered_df.groupby("sku")["family"].first().to_dict()
    sku_subfamily_map = filtered_df.groupby("sku")["subfamily"].first().to_dict()
    
    # Initialize result dataframe
    result = []
    
    # Process each SKU
    for sku in skus:
        sku_data = filtered_df[filtered_df["sku"] == sku]
        
        # Calculate total sales
        total_sales = sku_data["sales_value"].sum()
        
        # Resample to monthly to identify zero demand periods
        monthly_data = sku_data.set_index("invoice_date").resample("M")["sales_value"].sum().reset_index()
        
        # Calculate ADI (Average Demand Interval)
        # Count non-zero months
        non_zero_months = (monthly_data["sales_value"] > 0).sum()
        
        # Avoid division by zero
        if non_zero_months > 0:
            adi = len(monthly_data) / non_zero_months
        else:
            adi = float('inf')  # If no demand, set ADI to infinity
        
        # Calculate CV² (Coefficient of Variation squared)
        if non_zero_months > 1:
            # Calculate only for months with demand
            demand_values = monthly_data[monthly_data["sales_value"] > 0]["sales_value"]
            mean = demand_values.mean()
            std = demand_values.std()
            
            # Avoid division by zero
            if mean > 0:
                cv2 = (std / mean) ** 2
            else:
                cv2 = float('inf')
        else:
            cv2 = float('inf')  # If only one or zero demand period
        
        # Classify based on thresholds
        if cv2 <= cv2_threshold and adi <= adi_threshold:
            category = "Smooth"
        elif cv2 <= cv2_threshold and adi > adi_threshold:
            category = "Intermittent"
        elif cv2 > cv2_threshold and adi <= adi_threshold:
            category = "Erratic"
        else:
            category = "Lumpy"
        
        # Add to results
        result.append({
            "sku": sku,
            "family": sku_family_map.get(sku, ""),
            "subfamily": sku_subfamily_map.get(sku, ""),
            "total_sales": total_sales,
            "cv2": cv2,
            "adi": adi,
            "category": category
        })
    
    # Convert to DataFrame
    result_df = pd.DataFrame(result)
    
    # Calculate percentages
    if not result_df.empty:
        total_sales_all = result_df["total_sales"].sum()
        result_df["percent_of_total"] = result_df["total_sales"] / total_sales_all
        result_df["percent_formatted"] = result_df["percent_of_total"].apply(lambda x: f"{x*100:.1f}%")
    
    # Sort by sales value
    if not result_df.empty:
        result_df = result_df.sort_values("total_sales", ascending=False)
    
    return result_df

def create_intermittency_matrix(data, custom_title=None):
    """
    Cria a matriz de intermitência com os quadrantes Smooth, Intermittent, Erratic e Lumpy.
    
    Args:
        data (DataFrame): DataFrame com os dados de intermitência
        custom_title (str, optional): Título personalizado para o gráfico
        
    Returns:
        tuple: (figura Plotly, estatísticas dos quadrantes)
    """
    if data.empty:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(title="Sem dados para exibir")
        return fig, {}
    
    # Extract CV² and ADI values
    cv2_values = data["cv2"].values
    adi_values = data["adi"].values
    
    # Find max values with a cap to avoid extreme outliers
    max_cv2 = min(np.percentile(cv2_values[cv2_values != float('inf')], 95) * 1.2, 2.0) if len(cv2_values[cv2_values != float('inf')]) > 0 else 2.0
    max_adi = min(np.percentile(adi_values[adi_values != float('inf')], 95) * 1.2, 3.0) if len(adi_values[adi_values != float('inf')]) > 0 else 3.0
    
    # Get threshold values from the first row (all rows should have the same threshold)
    if not data.empty:
        cv2_threshold = data.iloc[0].get("cv2_threshold", 0.49)
        adi_threshold = data.iloc[0].get("adi_threshold", 1.32)
    else:
        cv2_threshold = 0.49
        adi_threshold = 1.32
    
    # Create figure
    fig = go.Figure()
    
    # Add data points
    for category in ["Smooth", "Intermittent", "Erratic", "Lumpy"]:
        category_data = data[data["category"] == category]
        
        # Skip if no data for this category
        if category_data.empty:
            continue
        
        # Map colors to categories
        color_map = {
            "Smooth": "green",
            "Intermittent": "blue",
            "Erratic": "gold",
            "Lumpy": "red"
        }
        
        # Add scatter points
        fig.add_trace(go.Scatter(
            x=category_data["adi"],
            y=category_data["cv2"],
            mode="markers",
            name=category,
            marker=dict(
                color=color_map.get(category, "gray"),
                size=10,
                opacity=0.7
            ),
            customdata=category_data[["sku", "family", "subfamily", "total_sales"]],
            hovertemplate="<b>%{customdata[0]}</b><br>" +
                          "Família: %{customdata[1]}<br>" +
                          "Subfamília: %{customdata[2]}<br>" +
                          "Vendas: %{customdata[3]:.2f} kg<br>" +
                          "CV²: %{y:.2f}<br>" +
                          "ADI: %{x:.2f}"
        ))
    
    # Add threshold lines
    fig.add_shape(
        type="line",
        x0=adi_threshold,
        y0=0,
        x1=adi_threshold,
        y1=max_cv2,
        line=dict(color="gray", width=1, dash="dash")
    )
    
    fig.add_shape(
        type="line",
        x0=0,
        y0=cv2_threshold,
        x1=max_adi,
        y1=cv2_threshold,
        line=dict(color="gray", width=1, dash="dash")
    )
    
    # Add quadrant labels
    fig.add_annotation(
        x=adi_threshold/2,
        y=cv2_threshold/2,
        text="Smooth",
        showarrow=False,
        font=dict(color="green", size=16)
    )
    
    fig.add_annotation(
        x=adi_threshold + (max_adi - adi_threshold)/2,
        y=cv2_threshold/2,
        text="Intermittent",
        showarrow=False,
        font=dict(color="blue", size=16)
    )
    
    fig.add_annotation(
        x=adi_threshold/2,
        y=cv2_threshold + (max_cv2 - cv2_threshold)/2,
        text="Erratic",
        showarrow=False,
        font=dict(color="gold", size=16)
    )
    
    fig.add_annotation(
        x=adi_threshold + (max_adi - adi_threshold)/2,
        y=cv2_threshold + (max_cv2 - cv2_threshold)/2,
        text="Lumpy",
        showarrow=False,
        font=dict(color="red", size=16)
    )
    
    # Update layout
    title = custom_title if custom_title else "Matriz de Classificação de Intermitência"
    
    fig.update_layout(
        title=title,
        xaxis_title="Tempo Médio Entre Demandas (ADI)",
        yaxis_title="Coeficiente de Variação (CV²)",
        xaxis=dict(
            range=[0, max_adi],
            tickvals=[0, adi_threshold, max_adi],
            ticktext=["0", f"{adi_threshold}", f"{max_adi:.1f}"]
        ),
        yaxis=dict(
            range=[0, max_cv2],
            tickvals=[0, cv2_threshold, max_cv2],
            ticktext=["0", f"{cv2_threshold}", f"{max_cv2:.1f}"]
        ),
        legend_title="Categoria",
        template="plotly_dark",
        hovermode="closest",
        height=500
    )
    
    # Calculate statistics
    stats = {}
    
    # Count SKUs and sales percentage by category
    for category in ["Smooth", "Intermittent", "Erratic", "Lumpy"]:
        category_data = data[data["category"] == category]
        stats[category] = {
            "sku_count": len(category_data),
            "sales_percentage": category_data["total_sales"].sum() / data["total_sales"].sum() if not data.empty else 0
        }
    
    return fig, stats

def create_quadrant_chart(data):
    """
    Cria um gráfico de barras mostrando a quantidade de SKUs por quadrante.
    
    Args:
        data (DataFrame): DataFrame com os dados de intermitência
        
    Returns:
        plotly.graph_objects.Figure: Figura com o gráfico de barras
    """
    if data.empty:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(title="Sem dados para exibir")
        return fig
    
    # Count SKUs per category
    category_counts = data["category"].value_counts().reset_index()
    category_counts.columns = ["Categoria", "SKUs"]
    
    # Calculate sales percentage per category
    category_sales = data.groupby("category")["total_sales"].sum().reset_index()
    total_sales = category_sales["total_sales"].sum()
    category_sales["% Vendas"] = (category_sales["total_sales"] / total_sales * 100).round(0).astype(int)
    
    # Merge counts and sales
    category_data = pd.merge(category_counts, category_sales, left_on="Categoria", right_on="category", how="left")
    
    # Define order and colors
    category_order = ["Smooth", "Intermittent", "Erratic", "Lumpy"]
    category_colors = {
        "Smooth": "green",
        "Intermittent": "blue",
        "Erratic": "gold",
        "Lumpy": "red"
    }
    
    # Filter and order data
    category_data = category_data[category_data["Categoria"].isin(category_order)]
    category_data["Categoria"] = pd.Categorical(category_data["Categoria"], categories=category_order, ordered=True)
    category_data = category_data.sort_values("Categoria")
    
    # Create figure
    fig = go.Figure()
    
    # Add bars
    for i, row in category_data.iterrows():
        fig.add_trace(go.Bar(
            x=[row["Categoria"]],
            y=[row["SKUs"]],
            name=row["Categoria"],
            marker_color=category_colors.get(row["Categoria"], "gray"),
            text=[f"{row['SKUs']}<br>{row['% Vendas']}%"],
            textposition="auto",
            hovertemplate=f"<b>{row['Categoria']}</b><br>" +
                          f"SKUs: {row['SKUs']}<br>" +
                          f"% Vendas: {row['% Vendas']}%"
        ))
    
    # Update layout
    fig.update_layout(
        showlegend=False,
        xaxis_title="Categoria",
        yaxis_title="Quantidade de SKUs",
        template="plotly_dark",
        height=300,
        margin=dict(l=50, r=30, t=30, b=50)
    )
    
    return fig

def get_sku_details(data, top_n=5):
    """
    Retorna os detalhes dos top N SKUs por valor de vendas.
    
    Args:
        data (DataFrame): DataFrame com os dados de intermitência
        top_n (int, optional): Número de SKUs a retornar
        
    Returns:
        DataFrame: DataFrame com os detalhes dos SKUs
    """
    if data.empty:
        return pd.DataFrame()
    
    # Select relevant columns and sort by sales
    result = data.sort_values("total_sales", ascending=False).head(top_n)
    
    # Format CV² and ADI for display
    result = result.copy()
    result["cv2_formatted"] = result["cv2"].apply(lambda x: f"{x:.2f}" if x != float('inf') else "∞")
    result["adi_formatted"] = result["adi"].apply(lambda x: f"{x:.2f}" if x != float('inf') else "∞")
    
    return result

def get_subfamily_options(df, selected_family):
    """
    Retorna as opções de subfamília com base na família selecionada.
    
    Args:
        df (DataFrame): DataFrame com os dados
        selected_family (str): Família selecionada
        
    Returns:
        list: Lista de opções de subfamília
    """
    # Always include "Todas" option
    subfamily_options = ["Todas"]
    
    # Add subfamily options based on family filter
    if selected_family and selected_family != "Todas":
        subfamily_options.extend(sorted(df[df["family"] == selected_family]["subfamily"].unique()))
    else:
        subfamily_options.extend(sorted(df["subfamily"].unique()))
    
    return subfamily_options

def get_quadrant_interpretation(quadrant):
    """
    Retorna a interpretação para cada quadrante de intermitência.
    
    Args:
        quadrant (str): Nome do quadrante
        
    Returns:
        str: Interpretação do quadrante
    """
    interpretations = {
        "Smooth": "Padrão regular de vendas, baixa variabilidade. Ideal para previsões convencionais.",
        "Intermittent": "Períodos de zero demanda, mas valores consistentes. Usar métodos específicos (Croston).",
        "Erratic": "Demanda frequente mas com grandes variações em magnitude. Considerar média móvel.",
        "Lumpy": "Alta variabilidade e longos períodos sem vendas. Previsão desafiadora, usar modelos de bootstrapping."
    }
    
    return interpretations.get(quadrant, "") 
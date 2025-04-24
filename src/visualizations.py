"""
Module for data visualizations using Plotly.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List

def create_aggregated_time_series(
    df: pd.DataFrame, 
    group_by: str, 
    group_value: Optional[str] = None,
    date_col: str = "invoice_date", 
    value_col: str = "sales_value", 
    period: str = "M",
    normalize: bool = False,
    reference_date: Optional[pd.Timestamp] = None,
    show_only_total: bool = False
) -> go.Figure:
    """
    Create a time series chart for aggregated data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe with sales data
    group_by : str
        Column to group by (e.g., 'family', 'subfamily', 'sku')
    group_value : str, optional
        Specific value to filter by (if None, shows top 3)
    date_col : str
        Date column name
    value_col : str
        Value column name
    period : str
        Aggregation period ('D' for day, 'M' for month, 'Q' for quarter)
    normalize : bool
        Whether to normalize values to 0-1 scale
    reference_date : Optional[pd.Timestamp], optional
        Reference date for the range slider
    show_only_total : bool, default=False
        If True, only shows the total line (aggregated) without individual lines
        
    Returns:
    --------
    go.Figure
        Plotly figure with the time series
    """
    # Make a copy and ensure date column is datetime
    df_copy = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df_copy[date_col]):
        df_copy[date_col] = pd.to_datetime(df_copy[date_col], errors='coerce')
    
    # Create a period column for aggregation
    df_copy['period'] = df_copy[date_col].dt.to_period(period)
    
    # Filter by group value if provided
    if group_value is not None and group_by in df_copy.columns:
        df_copy = df_copy[df_copy[group_by] == group_value]
    
    # Aggregate by period and group
    if group_by in df_copy.columns:
        # If no specific group selected, show all groups instead of top 3
        if group_value is None:
            # Get all unique groups, no top 3 limit
            all_groups = df_copy[group_by].unique().tolist()
            
            # If there are too many groups (more than 10), limit to avoid overcrowding
            if len(all_groups) > 10:
                # Get top 10 by sales volume
                top_groups = df_copy.groupby(group_by)[value_col].sum().nlargest(10).index.tolist()
                df_filtered = df_copy[df_copy[group_by].isin(top_groups)]
            else:
                # Use all groups
                df_filtered = df_copy
            
            # Aggregate by period and group
            agg_data = df_filtered.groupby(['period', group_by])[value_col].sum().reset_index()
            
            # Pivot to have one column per group
            pivot_data = agg_data.pivot(index='period', columns=group_by, values=value_col)
            # Convert Period index to datetime index explicitly
            pivot_data.index = pd.DatetimeIndex([p.to_timestamp() for p in pivot_data.index])
            pivot_data = pivot_data.sort_index()
            
            # Normalize if requested
            if normalize and not pivot_data.empty:
                for col in pivot_data.columns:
                    min_val = pivot_data[col].min()
                    max_val = pivot_data[col].max()
                    if max_val > min_val:
                        pivot_data[col] = (pivot_data[col] - min_val) / (max_val - min_val)
            
            # Create the plot
            fig = go.Figure()
            
            # Se show_only_total for True, calcular e mostrar apenas a soma total
            if show_only_total:
                total_data = pivot_data.sum(axis=1)
                fig.add_trace(go.Scatter(
                    x=pivot_data.index, 
                    y=total_data,
                    name="Total",
                    mode='lines',
                    line=dict(color='#ffffff', width=3)
                ))
            else:
                # Adicionar linhas para cada grupo
                for col in pivot_data.columns:
                    fig.add_trace(go.Scatter(
                        x=pivot_data.index, 
                        y=pivot_data[col],
                        name=str(col),
                        mode='lines+markers'
                    ))
                
            title = f"Time Series by {group_by.capitalize()}"
            if normalize:
                title += " (Normalized)"
            if show_only_total:
                title += " (Total Only)"
            
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Normalized Value" if normalize else "Sales Value",
                legend_title=group_by.capitalize(),
                height=500,  # Increased height
                margin=dict(l=20, r=20, t=40, b=20),
            )
            
            # Make the mini chart lines gray  
            fig.update_xaxes(
                rangeslider=dict(
                    bgcolor='rgba(50, 50, 70, 0.7)',
                    bordercolor='rgba(70, 70, 90, 1)',
                    thickness=0.08
                )
            )
            
        else:
            # For a specific group, aggregate by period
            agg_data = df_copy.groupby('period')[value_col].sum().reset_index()
            agg_data['period'] = agg_data['period'].dt.to_timestamp()
            agg_data = agg_data.sort_values('period')
            
            # Normalize if requested
            if normalize and not agg_data.empty:
                min_val = agg_data[value_col].min()
                max_val = agg_data[value_col].max()
                if max_val > min_val:
                    agg_data['normalized'] = (agg_data[value_col] - min_val) / (max_val - min_val)
                    value_col = 'normalized'
            
            # Create the plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=agg_data['period'], 
                y=agg_data[value_col],
                name=str(group_value) if not show_only_total else "Total",
                mode='lines+markers'
            ))
                
            title = f"Time Series for {group_by.capitalize()}: {group_value}"
            if normalize:
                title += " (Normalized)"
            # Se for apenas total, ajustar o título
            if show_only_total:
                title = f"Total Sales for {group_by.capitalize()}"
            
            fig.update_layout(
                title=title,
                xaxis_title="Date",
                yaxis_title="Normalized Value" if normalize else "Sales Value",
                height=350,
                margin=dict(l=20, r=20, t=40, b=20),
            )
    else:
        # If group_by column not in dataframe, aggregate all data by period
        agg_data = df_copy.groupby('period')[value_col].sum().reset_index()
        agg_data['period'] = agg_data['period'].dt.to_timestamp()
        agg_data = agg_data.sort_values('period')
        
        # Normalize if requested
        if normalize and not agg_data.empty:
            min_val = agg_data[value_col].min()
            max_val = agg_data[value_col].max()
            if max_val > min_val:
                agg_data['normalized'] = (agg_data[value_col] - min_val) / (max_val - min_val)
                value_col = 'normalized'
        
        # Create the plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=agg_data['period'], 
            y=agg_data[value_col],
            name="Total",
            mode='lines+markers'
        ))
            
        title = "Time Series for All Data"
        if normalize:
            title += " (Normalized)"
            
        fig.update_layout(
            title=title,
            xaxis_title="Date",
            yaxis_title="Normalized Value" if normalize else "Sales Value",
            height=350,
            margin=dict(l=20, r=20, t=40, b=20),
        )
    
    # Add range slider with reference date of February 2025
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeslider=dict(
            bgcolor='rgba(50, 50, 70, 0.7)',
            bordercolor='rgba(70, 70, 90, 1)',
            thickness=0.08
        ),
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all", label="all")
            ]),
            # Set default x-axis range to start from February 2025 going backward
            x=1,
            y=0
        ),
        # Remove the default date range to show all data
        # range=[pd.Timestamp('2025-02-01') - pd.DateOffset(months=3), pd.Timestamp('2025-02-01')]
    )
    
    return fig

def timeseries_visualization(
    df: pd.DataFrame,
    sku: Optional[str] = None,
    normalize: bool = False
) -> go.Figure:
    """
    Create a time series visualization for a specific SKU.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sales data
    sku : str, optional
        The SKU to visualize. If None, the first SKU in the DataFrame is used.
    normalize : bool, default=False
        Whether to normalize the sales values to a 0-1 scale
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The Plotly figure object
    """
    if df.empty:
        # Return empty figure if no data
        fig = go.Figure()
        fig.update_layout(
            title="No data available",
            xaxis_title="Date",
            yaxis_title="Sales"
        )
        return fig
    
    # Make sure we have the required columns
    required_cols = ["sku", "invoice_date", "sales_value"]
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing}")
    
    # Use the first SKU if none is specified
    if sku is None or sku not in df["sku"].unique():
        sku = df["sku"].iloc[0]
        
    # Filter data for the selected SKU
    filtered_df = df[df["sku"] == sku].copy()
    
    # Aggregate data by date (sum of sales per day)
    date_df = filtered_df.groupby("invoice_date")["sales_value"].sum().reset_index()
    date_df = date_df.sort_values("invoice_date")
    
    # Get metadata for the SKU
    sku_name = ""
    family = ""
    if "family" in df.columns:
        family_values = df[df["sku"] == sku]["family"].unique()
        if len(family_values) > 0:
            family = family_values[0]
    
    # Normalize values if requested
    y_column = "sales_value"
    if normalize and not date_df.empty:
        min_val = date_df["sales_value"].min()
        max_val = date_df["sales_value"].max()
        
        if max_val > min_val:  # Avoid division by zero
            date_df["normalized_sales"] = (date_df["sales_value"] - min_val) / (max_val - min_val)
            y_column = "normalized_sales"
    
    # Create the figure
    title = f"Sales Time Series for SKU: {sku}"
    if family:
        title += f" (Family: {family})"
        
    if normalize:
        title += " - Normalized (0-1)"
    
    # Create the line chart
    fig = px.line(
        date_df, 
        x="invoice_date", 
        y=y_column,
        title=title
    )
    
    # Update layout for better appearance
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Normalized Sales" if normalize else "Sales Value",
        hovermode="x unified",
        template="plotly_white"
    )
    
    # Add markers to the line
    fig.update_traces(mode="lines+markers", marker=dict(size=5))
    
    # Add range slider with reference date of February 2025
    fig.update_xaxes(
        rangeslider_visible=True,
        rangeslider=dict(
            bgcolor='rgba(50, 50, 70, 0.7)',
            bordercolor='rgba(70, 70, 90, 1)',
            thickness=0.08
        ),
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=3, label="3m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all", label="all")
            ]),
            # Set default x-axis range to start from February 2025 going backward
            x=1,
            y=0
        ),
        # Remove the default date range to show all data
        # range=[pd.Timestamp('2025-02-01') - pd.DateOffset(months=3), pd.Timestamp('2025-02-01')]
    )
    
    return fig

def get_sku_dropdown_options(df: pd.DataFrame) -> List[dict]:
    """
    Get options for SKU dropdown based on available data.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with sales data
        
    Returns:
    --------
    List[dict]
        List of dictionaries with label and value for each SKU
    """
    if df.empty or "sku" not in df.columns:
        return []
    
    # Get unique SKUs
    skus = df["sku"].unique()
    
    # Create options list
    options = []
    for sku in skus:
        label = sku
        
        # Add family info if available
        if "family" in df.columns:
            family_values = df[df["sku"] == sku]["family"].unique()
            if len(family_values) > 0:
                label = f"{sku} ({family_values[0]})"
        
        options.append({"label": label, "value": sku})
    
    return sorted(options, key=lambda x: x["label"]) 
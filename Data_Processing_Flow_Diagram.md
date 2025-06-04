# Data Processing Flow Diagram
## SKU Sales Dashboard - Complete Data Pipeline

```mermaid
flowchart TD
    %% Data Sources
    A[Excel Files] --> B1[2022-2025.xlsx<br/>Main Sales Data]
    A --> B2[Orçamento2022-2025.xlsx<br/>Budget Data]
    A --> B3[skus_mercado_especifico.xlsx<br/>Special Market SKUs]
    
    %% Initial Data Loading - data_loader.py
    B1 --> C1[load_sales_data function]
    B2 --> C2[load_budget_data function]
    B3 --> C3[get_mercado_especifico_skus function]
    
    %% Data Loading Processing
    C1 --> D1[Column Mapping<br/>& Data Cleaning]
    C2 --> D2[Budget Aggregation<br/>& Date Processing]
    C3 --> D3[Special Market<br/>SKU List]
    
    %% Core Data Structure
    D1 --> E[Unified Sales DataFrame<br/>Columns: sku, family, subfamily,<br/>sales_value, invoice_date,<br/>commercial_manager, etc.]
    
    %% Filtering Layer - utils/filters.py
    E --> F1[filter_dataframe function]
    E --> F2[filter_date_range function]
    E --> F3[filter_active_skus function]
    
    %% Analysis Modules
    F1 --> G1[ABC/XYZ Analysis<br/>src/abc_xyz.py]
    F2 --> G2[Intermittency Analysis<br/>src/intermittency_analysis.py]
    F3 --> G3[Time Series Processing<br/>src/visualizations.py]
    F1 --> G4[Forecasting Methods<br/>src/forecasting.py]
    
    %% ABC/XYZ Processing
    G1 --> H1[classify_abc function<br/>Sales Volume Classification]
    G1 --> H2[classify_xyz function<br/>Demand Variability Classification]
    H1 --> H3[ABC Categories: A, B, C<br/>Based on cumulative sales %]
    H2 --> H4[XYZ Categories: X, Y, Z<br/>Based on coefficient of variation]
    H3 --> I1[create_abc_xyz_matrix function]
    H4 --> I1
    
    %% Intermittency Processing
    G2 --> J1[Calculate CV² & ADI<br/>for each SKU]
    J1 --> J2[Classify Demand Patterns:<br/>Smooth, Intermittent,<br/>Erratic, Lumpy]
    J2 --> J3[create_intermittency_matrix function]
    
    %% Time Series Processing
    G3 --> K1[create_aggregated_time_series function]
    K1 --> K2[Period Aggregation<br/>Daily/Monthly/Quarterly]
    K2 --> K3[Optional Normalization<br/>utils/helpers.py]
    K3 --> K4[Time Series Visualization]
    
    %% Forecasting Processing
    G4 --> L1[Method Selection<br/>ARIMA, SES, SMA, TSB,<br/>XGBoost, Linear Regression]
    L1 --> L2[generate_forecasts function]
    L2 --> L3[Forecast Calculations<br/>src/analysis/ modules]
    L3 --> L4[Forecast Results DataFrame]
    
    %% Visualization Layer - src/visualizations.py
    I1 --> M1[Interactive Plotly Charts]
    J3 --> M2[Intermittency Matrix Chart]
    K4 --> M3[Time Series Line Charts]
    L4 --> M4[Forecast Visualizations]
    
    %% Streamlit Pages Layer
    M1 --> N1[abc_xyz_page.py]
    M2 --> N2[intermittency_page.py]
    M3 --> N3[time_series.py]
    M3 --> N4[aggregated_series.py]
    M4 --> N5[forecasting_methods_page.py]
    M4 --> N6[weighted_forecast_page.py]
    
    %% Main Application
    N1 --> O[app.py<br/>Main Streamlit App]
    N2 --> O
    N3 --> O
    N4 --> O
    N5 --> O
    N6 --> O
    
    %% User Interface
    O --> P[Interactive Dashboard<br/>with Sidebar Navigation]
    
    %% Styling
    classDef sourceFiles fill:#e1f5fe
    classDef processing fill:#f3e5f5
    classDef analysis fill:#e8f5e8
    classDef visualization fill:#fff3e0
    classDef ui fill:#fce4ec
    
    class A,B1,B2,B3 sourceFiles
    class C1,C2,C3,D1,D2,D3,E,F1,F2,F3 processing
    class G1,G2,G3,G4,H1,H2,H3,H4,I1,J1,J2,J3,K1,K2,K3,K4,L1,L2,L3,L4 analysis
    class M1,M2,M3,M4,N1,N2,N3,N4,N5,N6 visualization
    class O,P ui
```

## Detailed Data Processing Steps

### 1. Data Sources & Loading
- **Primary Data**: `2022-2025.xlsx` contains main sales data with columns:
  - `Gestor Comercial`, `Familia`, `Sub Familia`, `Gramagem`, `Formato`
  - `Material Mapeado` (SKU), `Valor Kg` (sales value), `Data do faturamento` (date)
- **Budget Data**: `Orçamento2022-2025.xlsx` for forecast comparison
- **Special Market**: `skus_mercado_especifico.xlsx` for market segmentation

### 2. Data Transformation Pipeline

#### 2.1 Column Standardization (`data_loader.py`)
```python
column_mapping = {
    'Data do faturamento': 'invoice_date',
    'Familia': 'family',
    'Sub Familia': 'subfamily', 
    'Material Mapeado': 'sku',
    'Valor Kg': 'sales_value',
    'Gestor Comercial': 'commercial_manager'
}
```

#### 2.2 Data Types & Cleaning
- Convert dates to `pd.datetime`
- Handle missing values
- Filter by 15 specific families (as per workspace rules)
- Validate data integrity

#### 2.3 Filtering Layer (`utils/filters.py`)
- **Date Range Filtering**: `filter_date_range()`
- **Multi-column Filtering**: `filter_dataframe()`
- **Active SKU Filtering**: Based on sales activity in specified periods

### 3. Analysis Modules

#### 3.1 ABC/XYZ Classification (`src/abc_xyz.py`)
```python
# ABC Analysis - Sales Volume
A: 0-80% of total sales (high volume)
B: 80-95% of total sales (medium volume)  
C: 95-100% of total sales (low volume)

# XYZ Analysis - Demand Variability
X: CV ≤ 20% (stable demand)
Y: 20% < CV ≤ 50% (moderate variability)
Z: CV > 50% (high variability)
```

#### 3.2 Intermittency Analysis (`src/intermittency_analysis.py`)
```python
# Classification based on CV² and ADI
Smooth: CV² ≤ 0.49 & ADI ≤ 1.32
Intermittent: CV² ≤ 0.49 & ADI > 1.32
Erratic: CV² > 0.49 & ADI ≤ 1.32
Lumpy: CV² > 0.49 & ADI > 1.32
```

#### 3.3 Time Series Processing (`src/visualizations.py`)
- **Aggregation**: Daily → Monthly → Quarterly
- **Normalization**: 0-1 scaling using `utils/helpers.py`
- **Interactive Charts**: Plotly with range sliders and zoom

#### 3.4 Forecasting Pipeline (`src/forecasting.py`)
```python
Methods Available:
- ARIMA(p,d,q)
- Simple Exponential Smoothing (SES)
- Simple Moving Average (SMA)
- Teunter-Syntetos-Boylan (TSB)
- XGBoost with lags
- Linear Regression with lags
- Weighted SKU Ponderation
```

### 4. Visualization & UI Layer

#### 4.1 Streamlit Pages Architecture
- **Overview** (`pages/overview.py`): KPIs and main trends
- **Time Series** (`pages/time_series.py`): Individual SKU analysis
- **Aggregated Series** (`pages/aggregated_series.py`): Group-level analysis
- **ABC/XYZ** (`pages/abc_xyz_page.py`): Classification matrices
- **Intermittency** (`pages/intermittency_page.py`): Demand pattern analysis
- **Forecasting** (`pages/forecasting_methods_page.py`): Prediction models

#### 4.2 Interactive Features
- **Dynamic Filtering**: Family, subfamily, date ranges
- **Real-time Updates**: Charts update based on filter selections
- **Export Capabilities**: Download filtered data and charts
- **Performance Optimization**: Caching for large datasets

### 5. Data Flow Control

#### 5.1 Session State Management
```python
st.session_state.sales_data     # Main dataset
st.session_state.data_loaded    # Loading status
st.session_state.filtered_data  # Current filtered view
st.session_state.analysis_cache # Cached analysis results
```

#### 5.2 Error Handling & Validation
- **File Validation**: Check required columns exist
- **Data Quality**: Handle missing dates, negative values
- **Performance**: Lazy loading for large datasets
- **User Feedback**: Progress bars and error messages

### 6. Key Libraries & Dependencies

#### 6.1 Core Processing
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computations
- **plotly**: Interactive visualizations
- **streamlit**: Web application framework

#### 6.2 Advanced Analytics
- **statsmodels**: Time series forecasting (ARIMA, ETS)
- **xgboost**: Machine learning forecasting
- **scikit-learn**: Linear regression and preprocessing

#### 6.3 Data Handling
- **openpyxl**: Excel file processing
- **datetime**: Date/time manipulations
- **typing**: Type hints for better code quality

This comprehensive flow ensures:
1. **Data Integrity**: Consistent column mapping and validation
2. **Performance**: Efficient filtering and caching mechanisms
3. **Flexibility**: Multiple analysis methods and visualization options
4. **User Experience**: Interactive filters and real-time updates
5. **Maintainability**: Modular structure with clear separation of concerns 
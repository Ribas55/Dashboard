# Data Processing Flow Diagram
## SKU Sales Dashboard - Data Processing Pipeline

```mermaid
%%{init: {'theme':'default', 'themeVariables': {'fontFamily': 'Arial', 'fontSize': '14px', 'lineColor': '#000000', 'primaryColor': '#ffffff', 'primaryTextColor': '#000000', 'primaryBorderColor': '#333333', 'background': '#ffffff', 'mainBkg': '#ffffff', 'cScale0': '#4285f4', 'cScale1': '#34a853', 'cScale2': '#ff9800', 'cScale3': '#9c27b0'}}}%%
flowchart TD
    %% Data Extraction
    A[Raw Data Sources] --> B[Data Loading<br/>pandas.read_excel]
    
    %% Data Processing
    B --> C[Column Mapping<br/>pandas.DataFrame.rename]
    C --> D[Data Type Conversion<br/>pandas.to_datetime]
    D --> E[Data Cleaning<br/>Handle missing values]
    
    %% Data Transformation
    E --> F[Data Filtering<br/>pandas.DataFrame.query]
    F --> G[Data Aggregation<br/>pandas.groupby]
    
    %% Analysis Processing
    G --> H1[ABC Analysis<br/>numpy calculations]
    G --> H2[XYZ Analysis<br/>Coefficient of Variation]
    G --> H3[Time Series Processing<br/>pandas.resample]
    G --> H4[Intermittency Analysis<br/>Statistical calculations]
    
    %% Statistical Calculations
    H1 --> I1[Sales Volume Classification<br/>numpy.cumsum]
    H2 --> I2[Demand Variability<br/>numpy.std, numpy.mean]
    H3 --> I3[Period Aggregation<br/>pandas.Period]
    H4 --> I4[CV² and ADI Calculation<br/>numpy mathematical functions]
    
    %% Results
    I1 --> J[Processed DataFrames<br/>Ready for Visualization]
    I2 --> J
    I3 --> J
    I4 --> J
    
    %% Visualization Preparation
    J --> K[Chart Data Preparation<br/>plotly.graph_objects]
    K --> L[Interactive Visualizations<br/>plotly.express]
    
    %% Styling with vibrant clean colors and black borders
    classDef extraction fill:#4285f4,stroke:#000000,stroke-width:3px,color:#ffffff
    classDef processing fill:#34a853,stroke:#000000,stroke-width:3px,color:#ffffff
    classDef analysis fill:#ff9800,stroke:#000000,stroke-width:3px,color:#ffffff
    classDef output fill:#9c27b0,stroke:#000000,stroke-width:3px,color:#ffffff
    
    class A,B extraction
    class C,D,E,F,G processing
    class H1,H2,H3,H4,I1,I2,I3,I4 analysis
    class J,K,L output
```

## Data Processing Steps

### 1. Data Extraction & Loading
- **pandas**: Read data from various sources
- **Column standardization**: Map original column names to standard format

### 2. Data Processing
- **Data type conversion**: Convert dates using `pandas.to_datetime()`
- **Data cleaning**: Handle missing values and outliers
- **Filtering**: Apply business rules using `pandas.query()`

### 3. Data Transformation
- **Aggregation**: Group data by SKU, family, time periods using `pandas.groupby()`
- **Time series preparation**: Create period indexes with `pandas.resample()`

### 4. Statistical Analysis
- **ABC Classification**: Calculate cumulative percentages using `numpy.cumsum()`
- **XYZ Classification**: Compute coefficient of variation with `numpy.std()` and `numpy.mean()`
- **Intermittency Analysis**: Calculate CV² and ADI using mathematical functions
- **Time series metrics**: Statistical calculations for demand patterns

### 5. Output Preparation
- **Data structuring**: Prepare processed DataFrames for visualization
- **Chart data formatting**: Transform data for plotly visualization requirements

## Key Libraries Used

| Library | Purpose | Main Functions |
|---------|---------|----------------|
| **pandas** | Data manipulation | `read_excel()`, `groupby()`, `resample()`, `to_datetime()` |
| **numpy** | Numerical calculations | `std()`, `mean()`, `cumsum()`, mathematical operations |
| **plotly** | Visualization | `graph_objects`, `express` for interactive charts |

## Processing Flow Summary

1. **Extract** → Load raw data using pandas
2. **Transform** → Clean, map columns, convert data types  
3. **Aggregate** → Group and summarize data by business dimensions
4. **Analyze** → Apply statistical methods for classification
5. **Prepare** → Format data for visualization and reporting 
# SKU Sales Dashboard

An interactive Python dashboard for visualization and analysis of sales time series by SKU.

## Project Structure

```
project-root/
│
├── app.py                     # Entry point (Streamlit app)
├── requirements.txt           # Dependencies
├── README.md
│
├── data/
│   └── 2022-2025.xlsx         # Sales data
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Excel/CSV file reading
│   ├── preprocess.py          # Aggregations, normalizations, transformations
│   ├── abc_xyz.py             # ABC/XYZ analysis logic
│   ├── intermitencia.py       # Classification logic (smooth, lumpy...)
│   └── visualizations.py      # Functions for creating charts (Plotly)
│
└── utils/
    ├── filters.py             # Data filters
    └── helpers.py             # Helper functions (e.g., normalization)
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit app:
```
streamlit run app.py
```

## Features

- Load and analyze sales data by SKU
- Interactive filters and visualizations
- ABC/XYZ analysis
- Intermittency classification (smooth, irregular, lumpy, erratic)
- Time series visualization and analysis 
import os
import streamlit as st

# Ocultar todas as páginas Python da interface
st.set_page_config(
    page_title="Análise de Vendas",
    page_icon="📊",
    initial_sidebar_state="expanded"
)

# Redirecionar para o app.py
os.system("streamlit run app.py") 
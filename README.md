# Dashboard de Análise de Vendas - SKU Intelligence Platform

## 📊 Visão Geral

Este projeto é uma **plataforma interativa de análise de vendas** desenvolvida em Python que utiliza **Streamlit** para criar dashboards dinâmicos de análise de séries temporais, classificação ABC/XYZ, análise de intermitência e métodos de previsão para SKUs (Stock Keeping Units).

O sistema foi especialmente projetado para o setor de **produtos alimentícios** (biscoitos), oferecendo insights estratégicos para gestão de inventário, planejamento de produção e otimização de vendas.

## 🎯 Objetivos do Projeto

- **Visualização Interativa**: Dashboards responsivos com filtros dinâmicos
- **Análise ABC/XYZ**: Classificação de SKUs por volume de vendas e variabilidade de demanda
- **Análise de Intermitência**: Categorização do comportamento de demanda (smooth, intermittent, erratic, lumpy)
- **Previsão Avançada**: Múltiplos métodos de forecasting com comparação de performance
- **Segmentação de Mercado**: Análise separada para mercado normal vs. mercado específico
- **Insights Estratégicos**: Recomendações automáticas baseadas em classificações

## 🚀 Funcionalidades Principais

### 1. **Página Inicial (Overview)**
- Métricas gerais do portfólio
- Resumo executivo de vendas por família
- Indicadores chave de performance (KPIs)
- Visualizações de tendências anuais

### 2. **Análise ABC/XYZ**
- **Matriz ABC/XYZ**: Visualização 3x3 interativa com 9 quadrantes
- **Gráfico de Barras**: Distribuição de SKUs por categoria
- **Filtros Dinâmicos**: Por família, subfamília e tipo de mercado
- **Detalhes dos SKUs**: Tabela com top performers e estatísticas
- **Interpretações Automáticas**: Recomendações estratégicas por quadrante

### 3. **Séries Temporais**
- **Visualizações Interativas**: Gráficos de linha com zoom e pan
- **Normalização**: Opção de visualizar dados normalizados
- **Filtros Múltiplos**: Seleção de SKUs, famílias e períodos
- **Agregação Temporal**: Visão mensal, trimestral e anual
- **Comparações**: Análise side-by-side de múltiplos SKUs

### 4. **Análise de Intermitência**
- **Classificação em 4 Categorias**:
  - **Smooth**: Demanda regular e constante
  - **Intermittent**: Demanda ocasional mas previsível
  - **Erratic**: Demanda irregular e variável
  - **Lumpy**: Demanda esporádica com grandes volumes
- **Gráficos de Distribuição**: Scatter plots ADI vs CV²
- **Métricas Especializadas**: Average Demand Interval (ADI) e Squared Coefficient of Variation (CV²)

### 5. **Métodos de Previsão**
- **Simple Moving Average (SMA)**
- **Single Exponential Smoothing (SES)**
- **Triple Exponential Smoothing (Holt-Winters)**
- **ARIMA**: Modelos auto-regressivos integrados
- **Linear Regression**: Regressão linear com tendências
- **XGBoost**: Machine learning para previsões avançadas
- **Métricas de Avaliação**: MAE, RMSE, MAPE para cada método

### 6. **Previsão Ponderada Customizada**
- **Combinação de Métodos**: Weighted ensemble de múltiplos algoritmos
- **Pesos Customizáveis**: Ajuste manual dos pesos por método
- **Validação Cruzada**: Avaliação de performance out-of-sample
- **Comparação Visual**: Gráficos comparativos entre métodos

### 7. **Comparação de Resultados**
- **Ranking de Métodos**: Performance por SKU e por método
- **Análise Estatística**: Testes de significância entre métodos
- **Visualizações Comparativas**: Heatmaps e gráficos de performance
- **Recomendações**: Método ótimo por SKU baseado em métricas

## 🛠️ Instalação e Configuração

### Pré-requisitos
- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### 1. Clone o Repositório
```bash
git clone [URL_DO_REPOSITORIO]
cd Dashboard_Vieira
```

### 2. Instale as Dependências
```bash
pip install -r requirements.txt
```

### 3. Verifique a Estrutura de Dados
Certifique-se de que o arquivo `data/2022-2025.xlsx` está presente com a estrutura correta:
- **Gestor Comercial**: Responsável pelas vendas
- **Familia**: Categoria principal do produto
- **Sub Familia**: Subcategoria do produto  
- **Gramagem**: Peso/tamanho do produto
- **Formato**: Tipo de embalagem
- **Material Mapeado**: SKU único do produto
- **Valor Kg**: Valor das vendas (variável dependente)
- **Data do faturamento**: Data da transação (componente temporal)

### 4. Execute o Dashboard
```bash
streamlit run app.py
```

O dashboard estará disponível em `http://localhost:8501`

## 📁 Estrutura do Projeto

```
Dashboard_Vieira/
├── app.py                          # Aplicação principal Streamlit
├── requirements.txt                # Dependências Python
├── README.md                       # Documentação do projeto
│
├── data/                           # Dados de entrada
│   ├── 2022-2025.xlsx             # Dados principais de vendas
│   ├── BD_ATUALIZACAO.xlsx        # Base de dados de atualização
│   ├── Orçamento2022-2025.xlsx    # Dados orçamentários
│   └── skus_mercado_especifico.xlsx # SKUs do mercado específico
│
├── src/                            # Módulos de análise
│   ├── __init__.py
│   ├── data_loader.py              # Carregamento e preparação de dados
│   ├── abc_xyz.py                  # Classificação ABC/XYZ genérica
│   ├── abc_xyz_analysis.py         # Análise ABC/XYZ para dashboard
│   ├── intermittency_analysis.py   # Análise de intermitência
│   ├── forecasting.py              # Métodos de previsão
│   ├── mercado_especifico.py       # Lógica de mercado específico
│   ├── visualizations.py           # Funções de visualização
│   └── analysis/                   # Módulos especializados
│       ├── arima_forecast.py       # Previsão ARIMA
│       ├── ses_forecast.py         # Exponential Smoothing
│       ├── linear_regression_forecast.py # Regressão Linear
│       ├── xgboost_forecast.py     # XGBoost
│       ├── sma_forecast.py         # Simple Moving Average
│       ├── tsb_forecast.py         # TSB
│       └── custom_weighted_forecast.py # Previsão ponderada
│
├── pages/                          # Páginas do dashboard
│   ├── __init__.py
│   ├── overview.py                 # Página inicial
│   ├── abc_xyz_page.py            # Análise ABC/XYZ
│   ├── time_series.py             # Séries temporais
│   ├── aggregated_series.py       # Séries agregadas
│   ├── intermittency_page.py      # Análise de intermitência
│   ├── forecasting_methods_page.py # Métodos de previsão
│   ├── weighted_forecast_page.py   # Previsão ponderada
│   └── results_comparison_page.py  # Comparação de resultados
│
├── utils/                          # Utilitários
│   ├── __init__.py
│   ├── filters.py                  # Filtros de dados
│   └── helpers.py                  # Funções auxiliares
│
└── output/                         # Resultados e exports
    ├── *.png                       # Gráficos exportados
    └── *.xlsx                      # Tabelas exportadas
```

## 🧮 Metodologia Matemática

### **Análise ABC (Classificação por Volume)**

**1. Cálculo de Vendas Totais por SKU:**
```
Total_Sales_i = Σ(sales_value_i) para todos os períodos do ano
```

**2. Percentual Individual:**
```
Percentage_i = Total_Sales_i / Σ(Total_Sales_all_SKUs)
```

**3. Percentual Cumulativo:**
```
Cumulative_Percentage_i = Σ(Percentage_j) para j = 1 até i (ordenado decrescente)
```

**4. Regras de Classificação ABC:**
- **Classe A**: Cumulative_Percentage ≤ 80% (produtos estratégicos)
- **Classe B**: 80% < Cumulative_Percentage ≤ 95% (produtos importantes)
- **Classe C**: Cumulative_Percentage > 95% (produtos de baixo volume)

### **Análise XYZ (Classificação por Variabilidade)**

**1. Agregação Mensal:**
```
Monthly_Sales_i,m = Σ(sales_value_i) para o mês m
```

**2. Média Mensal:**
```
μ_i = (1/12) × Σ(Monthly_Sales_i,m) para m = 1 até 12
```

**3. Desvio Padrão Amostral:**
```
σ_i = √[(1/11) × Σ(Monthly_Sales_i,m - μ_i)²]
```
*Nota: Usa ddof=1 (graus de liberdade = 11) para estimativa não-viesada*

**4. Coeficiente de Variação:**
```
CV_i = σ_i / μ_i  (quando μ_i > 0)
CV_i = ∞         (quando μ_i = 0)
```

**5. Regras de Classificação XYZ:**
- **Classe X**: CV ≤ 20% (demanda regular)
- **Classe Y**: 20% < CV ≤ 50% (variabilidade moderada)
- **Classe Z**: CV > 50% (alta variabilidade)

### **Análise de Intermitência**

**1. Average Demand Interval (ADI):**
```
ADI = Número_total_períodos / Número_períodos_com_demanda
```

**2. Squared Coefficient of Variation (CV²):**
```
CV² = (σ / μ)²
```

**3. Classificação de Intermitência:**
- **Smooth**: ADI ≤ 1.32 e CV² ≤ 0.49
- **Intermittent**: ADI > 1.32 e CV² ≤ 0.49  
- **Erratic**: ADI ≤ 1.32 e CV² > 0.49
- **Lumpy**: ADI > 1.32 e CV² > 0.49

## 🎛️ Guia de Uso

### **1. Navegação Principal**
Use a **sidebar esquerda** para navegar entre as páginas:
- Selecione a página desejada no menu dropdown
- Cada página tem filtros específicos na sidebar
- Gráficos são interativos (zoom, pan, seleção)

### **2. Filtros Disponíveis**
- **Família**: Filtra por categoria de produto (15 famílias disponíveis)
- **Subfamília**: Filtra por subcategoria (dinâmico baseado na família)
- **Tipo de Mercado**: Normal, Mercado Específico, ou Todos
- **Ano**: Seleciona o ano de análise (2022-2025)
- **SKUs**: Seleção múltipla de produtos específicos

### **3. Interpretação de Resultados**

#### **Matriz ABC/XYZ - 9 Quadrantes:**
- **AX**: Alto volume, baixa variabilidade → Produtos estratégicos
- **AY**: Alto volume, média variabilidade → Monitoramento ativo
- **AZ**: Alto volume, alta variabilidade → Atenção especial
- **BX**: Médio volume, baixa variabilidade → Gestão equilibrada
- **BY**: Médio volume, média variabilidade → Revisões periódicas
- **BZ**: Médio volume, alta variabilidade → Avaliação cautelosa
- **CX**: Baixo volume, baixa variabilidade → Produtos de nicho
- **CY**: Baixo volume, média variabilidade → Candidatos a consolidação
- **CZ**: Baixo volume, alta variabilidade → Candidatos à descontinuação

#### **Análise de Intermitência:**
- **Smooth**: Reabastecimento regular e previsível
- **Intermittent**: Políticas de stock de segurança
- **Erratic**: Análise de causas da variabilidade
- **Lumpy**: Estratégias just-in-time ou sob demanda

### **4. Exportação de Resultados**
- Gráficos podem ser salvos usando o menu do Plotly (📷)
- Tabelas são exportáveis para Excel
- Resultados são automaticamente salvos na pasta `output/`

## 🔧 Configurações Avançadas

### **Personalização de Thresholds**
No código `src/abc_xyz_analysis.py`, você pode ajustar:
```python
a_threshold: float = 0.8    # Limite A/B (padrão: 80%)
b_threshold: float = 0.95   # Limite B/C (padrão: 95%)
x_threshold: float = 0.2    # Limite X/Y (padrão: 20%)
y_threshold: float = 0.5    # Limite Y/Z (padrão: 50%)
```

### **Adição de Novas Famílias**
Edite a lista no workspace rule:
```python
familia_permitidas = [
    "Cream Cracker", "Maria", "Wafer", "Sortido",
    "Cobertas de Chocolate", "Água e Sal", "Digestiva",
    "Recheada", "Circus", "Tartelete", "Torrada",
    "Flocos de Neve", "Integral", "Mentol", "Aliança"
]
```

### **Novos Métodos de Previsão**
Para adicionar um novo método:
1. Crie um arquivo em `src/analysis/novo_metodo_forecast.py`
2. Implemente a função seguindo o padrão dos métodos existentes
3. Adicione a importação em `src/forecasting.py`
4. Inclua no menu da página de previsão

## 📊 Métricas de Performance

### **Métricas de Previsão**
- **MAE** (Mean Absolute Error): Erro médio absoluto
- **RMSE** (Root Mean Square Error): Raiz do erro quadrático médio  
- **MAPE** (Mean Absolute Percentage Error): Erro percentual absoluto médio
- **Accuracy**: Precisão da previsão (1 - MAPE)

### **Benchmarks de Performance**
- **Excelente**: MAPE < 10%
- **Bom**: 10% ≤ MAPE < 20%
- **Aceitável**: 20% ≤ MAPE < 30%
- **Necessita Melhoria**: MAPE ≥ 30%

## 🐛 Troubleshooting

### **Problemas Comuns**

**1. Erro de Carregamento de Dados:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/2022-2025.xlsx'
```
**Solução**: Verifique se o arquivo existe e está na pasta `data/`

**2. Erro de Memória:**
```
MemoryError: Unable to allocate array
```
**Solução**: Reduza o período de análise ou filtre por família específica

**3. Gráficos Não Aparecem:**
**Solução**: Verifique se há dados para os filtros selecionados

**4. Performance Lenta:**
**Solução**: Use filtros mais específicos e evite selecionar "Todos" quando desnecessário

### **Logs e Debug**
Ative o modo debug adicionando ao início de `app.py`:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔮 Roadmap de Desenvolvimento

### **Próximas Funcionalidades**
- [ ] **Dashboard Executivo**: KPIs em tempo real
- [ ] **Alertas Automáticos**: Notificações de anomalias
- [ ] **API RESTful**: Integração com outros sistemas
- [ ] **Análise de Sazonalidade**: Detecção automática de padrões
- [ ] **Machine Learning Avançado**: Modelos de deep learning
- [ ] **Otimização de Estoque**: Recomendações de reorder points
- [ ] **Análise de Rentabilidade**: Integração com dados de custo

### **Melhorias Técnicas**
- [ ] **Cache Inteligente**: Otimização de performance
- [ ] **Testes Automatizados**: Cobertura de testes unitários
- [ ] **Documentação API**: Swagger/OpenAPI
- [ ] **Containerização**: Docker deployment
- [ ] **CI/CD Pipeline**: Automatização de deploys

## 📞 Suporte e Contribuições

### **Como Contribuir**
1. Fork o repositório
2. Crie uma branch para sua feature (`git checkout -b feature/nova-funcionalidade`)
3. Commit suas mudanças (`git commit -am 'Adiciona nova funcionalidade'`)
4. Push para a branch (`git push origin feature/nova-funcionalidade`)
5. Crie um Pull Request

### **Reportar Bugs**
Use as Issues do GitHub com:
- Descrição detalhada do problema
- Passos para reproduzir
- Screenshots (se aplicável)
- Informações do ambiente (OS, Python version, etc.)

## 📄 Licença

Este projeto está sob licença [MIT](LICENSE). Veja o arquivo LICENSE para mais detalhes.

## 🙏 Agradecimentos

Desenvolvido para otimização de análise de vendas no setor alimentício, com foco em biscoitos e produtos relacionados. Especial agradecimento à equipe de planejamento e gestão comercial pelas especificações e requisitos do projeto.

---

**Versão**: 1.0.0  
**Última Atualização**: Janeiro 2025  
**Python**: 3.8+  
**Framework**: Streamlit 1.28+ 
# Sentinel Risk Intelligence

**AI-Enhanced Quantitative Portfolio Risk Platform**

Sentinel Risk Intelligence is a modular financial risk analytics platform that combines **quantitative portfolio risk modeling** with **AI-assisted market context analysis**.

The system allows users to analyze individual assets or portfolios, perform stress testing, evaluate risk drivers, and optionally generate AI-assisted market risk briefs based on real-time news.

Built as a **full-stack data application** using Python, Streamlit, and modern data science tooling.

---

# Key Features

## Quantitative Risk Engine

- Historical **Value-at-Risk (VaR)**
- **Conditional VaR (Expected Shortfall)**
- **Annualized Volatility**
- **Maximum Drawdown**
- **Stress Testing & Scenario Analysis**
- **Portfolio Risk Contribution**
- **Diversification Metrics**

---

## Portfolio Analytics

- Multi-asset portfolio analysis
- Portfolio volatility estimation
- Asset-level risk contribution
- Weight vs risk contribution comparison
- Interactive visualization

---

## AI Market Intelligence

Optional AI-assisted analysis using an LLM:

- Market headline summarization
- Risk classification
- Sentiment scoring
- Explainable AI risk adjustment

The system also includes a **free RSS-based news ingestion pipeline** to automatically fetch recent market headlines.

---

## Executive Reporting

Sentinel can generate downloadable **Executive Risk Reports** containing:

- Volatility metrics
- VaR / CVaR analysis
- Risk score summary
- Risk interpretation

Reports are exported as **HTML documents**.

---

# System Architecture

The project follows a modular architecture:

Hier ist das komplette README als ein einziger Markdown-Block, ohne Screenshot-Sektion und ohne zusätzliche Markierungen — einfach komplett kopieren und in README.md einfügen.

# Sentinel Risk Intelligence

**AI-Enhanced Quantitative Portfolio Risk Platform**

Sentinel Risk Intelligence is a modular financial risk analytics platform that combines **quantitative portfolio risk modeling** with **AI-assisted market context analysis**.

The system allows users to analyze individual assets or portfolios, perform stress testing, evaluate risk drivers, and optionally generate AI-assisted market risk briefs based on real-time news.

Built as a **full-stack data application** using Python, Streamlit, and modern data science tooling.

---

# Key Features

## Quantitative Risk Engine

- Historical **Value-at-Risk (VaR)**
- **Conditional VaR (Expected Shortfall)**
- **Annualized Volatility**
- **Maximum Drawdown**
- **Stress Testing & Scenario Analysis**
- **Portfolio Risk Contribution**
- **Diversification Metrics**

---

## Portfolio Analytics

- Multi-asset portfolio analysis
- Portfolio volatility estimation
- Asset-level risk contribution
- Weight vs risk contribution comparison
- Interactive visualization

---

## AI Market Intelligence

Optional AI-assisted analysis using an LLM:

- Market headline summarization
- Risk classification
- Sentiment scoring
- Explainable AI risk adjustment

The system also includes a **free RSS-based news ingestion pipeline** to automatically fetch recent market headlines.

---

## Executive Reporting

Sentinel can generate downloadable **Executive Risk Reports** containing:

- Volatility metrics
- VaR / CVaR analysis
- Risk score summary
- Risk interpretation

Reports are exported as **HTML documents**.

---

# System Architecture

The project follows a modular architecture:


src/
│
├── sentinel
│ ├── data # Market data loading
│ ├── portfolio # Portfolio construction & returns
│ ├── risk # Risk models (VaR, CVaR, stress testing)
│ ├── ai # AI market context & news ingestion
│ └── reporting # Executive report generation
│
└── app
└── streamlit_app.py


Architecture layers:

1. **Data Layer**  
   Market data ingestion via `yfinance`

2. **Portfolio Layer**  
   Portfolio return calculations and asset weighting

3. **Risk Engine**  
   Quantitative risk metrics and stress testing

4. **AI Layer**  
   Optional LLM-based market intelligence

5. **Reporting Layer**  
   Executive risk report generation

6. **Interface Layer**  
   Interactive Streamlit dashboard

---

# Technology Stack

- Python  
- Pandas  
- NumPy  
- SciPy  
- Streamlit  
- yfinance  
- pytest  
- OpenAI API *(optional)*  
- RSS News Feeds  

---

# Running the Project

Follow these steps to run Sentinel Risk Intelligence locally.

## 1. Clone the Repository

```bash
git clone https://github.com/yourusername/sentinel-risk-intelligence.git
cd sentinel-risk-intelligence
```
## 2. Install Dependencies

```bash
pip install -r requirements.txt
pip install -e .
```
## 3. (Optional) Configure AI Features

use your OPENAI KEY in the .env File. If no API key is provided, the application will still run and the AI module will fall back to manual mode.

## 4. Run the Application

```bash
python -m streamlit run src/app/streamlit_app.py
```

## 5. Running Tests

```bash
python -m pytest


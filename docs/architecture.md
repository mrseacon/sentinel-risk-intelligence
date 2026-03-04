# Sentinel Risk Intelligence Architecture

Sentinel is structured as a modular risk analytics platform.

Core layers:

1. Data Layer
Responsible for market data ingestion using yfinance.

2. Portfolio Layer
Handles asset returns and portfolio construction.

3. Risk Engine
Computes:
- Volatility
- Value-at-Risk
- CVaR
- Stress Testing
- Risk Contribution

4. AI Layer
Provides optional LLM-based market risk summaries.

5. Reporting Layer
Generates executive risk reports.

6. Interface Layer
Interactive Streamlit dashboard.
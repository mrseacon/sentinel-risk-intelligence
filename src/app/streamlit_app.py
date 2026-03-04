import streamlit as st

from sentinel.data.loader import load_price_data
from sentinel.portfolio.portfolio import calculate_returns
from sentinel.risk.metrics import annualized_volatility
from sentinel.risk.var import historical_var

st.title("Sentinel Risk Intelligence")

ticker = st.text_input("Ticker", "AAPL")

if st.button("Run Analysis"):

    data = load_price_data(ticker)

    returns = calculate_returns(data["Adj Close"])

    vol = annualized_volatility(returns)

    var = historical_var(returns)

    st.metric("Annualized Volatility", f"{vol:.2%}")

    st.metric("Historical VaR (95%)", f"{var:.2%}")
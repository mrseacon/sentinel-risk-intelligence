import streamlit as st

from sentinel.data.loader import load_price_data
from sentinel.portfolio.portfolio import calculate_returns
from sentinel.risk.metrics import annualized_volatility, max_drawdown
from sentinel.risk.var import historical_var, historical_cvar
from sentinel.risk.stress import apply_market_shock
from sentinel.risk.scoring import compute_risk_score

st.set_page_config(page_title="Sentinel Risk Intelligence", layout="wide")
st.title("Sentinel Risk Intelligence")

ticker = st.text_input("Ticker", "AAPL")
confidence = st.slider("Confidence level", 0.90, 0.99, 0.95, 0.01)

if st.button("Run Analysis"):
    data = load_price_data(ticker)
    prices = data["Adj Close"]
    rets = calculate_returns(prices)

    vol = annualized_volatility(rets)
    dd = max_drawdown(prices)

    var = historical_var(rets, confidence)
    cvar = historical_cvar(rets, confidence)

    score = compute_risk_score(
        vol_annual=vol,
        max_dd=dd,
        var_95=var,
        cvar_95=cvar,
        concentration_hhi=None,  # portfolio later
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Annualized Volatility", f"{vol:.2%}")
    col2.metric("Max Drawdown", f"{dd:.2%}")
    col3.metric(f"Historical VaR ({int(confidence*100)}%)", f"{var:.2%}")
    col4.metric(f"Historical CVaR ({int(confidence*100)}%)", f"{cvar:.2%}")

    st.subheader("Risk Score")
    st.metric("Score (0–100)", f"{score.score} ({score.label})")
    if score.drivers:
        st.write("Top risk drivers:")
        for d in score.drivers:
            st.write(f"- {d}")

    st.subheader("Stress Test")
    shock = st.slider("Market shock (%)", -30, 0, -10, 1)
    stress = apply_market_shock(prices, shock / 100)
    st.write(
        {
            "scenario": stress.scenario,
            "shocked_last_price": round(stress.shocked_last_price, 2),
            "pnl_pct": f"{stress.pnl_pct:.2%}",
        }
    )
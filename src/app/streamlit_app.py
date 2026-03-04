import streamlit as st

from sentinel.data.loader import load_price_data
from sentinel.portfolio.portfolio import calculate_returns
from sentinel.risk.metrics import annualized_volatility, max_drawdown
from sentinel.risk.var import historical_var, historical_cvar
from sentinel.risk.stress import apply_market_shock
from sentinel.risk.scoring import compute_risk_score
from sentinel.data.loader import load_multiple_assets
from sentinel.portfolio.returns import (
    calculate_asset_returns,
    portfolio_returns,
)
from sentinel.risk.portfolio_risk import portfolio_volatility
from sentinel.risk.risk_contribution import portfolio_risk_contribution
from sentinel.ai.llm_client import LLMClient
from sentinel.ai.market_context import build_market_context_prompt, parse_market_context
from sentinel.ai.risk_adjustment import compute_ai_adjustment
from sentinel.reporting.report_generator import generate_risk_report

prices = load_multiple_assets(list(portfolio.keys()))

returns = calculate_asset_returns(prices)

port_ret = portfolio_returns(returns, portfolio)

vol = portfolio_volatility(returns, portfolio)

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

portfolio = {
    "AAPL": 0.4,
    "MSFT": 0.3,
    "SPY": 0.3,
}
rc = portfolio_risk_contribution(returns, portfolio)

st.subheader("Risk Contribution")

st.bar_chart(rc)

tabs = st.tabs(["Quant Overview", "AI Market Brief"])

with tabs[1]:
    st.subheader("AI Market Brief (Hybrid)")
    st.caption("Runs without API key in manual mode. If OPENAI_API_KEY is set, it will generate automatically.")

    portfolio_tickers = ["AAPL", "MSFT", "SPY"]  # replace with your actual portfolio keys
    st.write("Portfolio tickers:", ", ".join(portfolio_tickers))

    manual_headlines = st.text_area(
        "Paste market headlines / notes (manual mode fallback)",
        value="Example: Fed hints at higher rates; tech stocks volatile; oil prices rise.",
        height=140,
    )

    client = LLMClient()

    use_llm = st.checkbox(
        "Use LLM (requires OPENAI_API_KEY + openai package)", value=False
    )

    if st.button("Generate Market Brief"):
        if use_llm:
            try:
                prompt = build_market_context_prompt(manual_headlines, portfolio_tickers)
                resp = client.generate(prompt)
                context = parse_market_context(resp.text)
                st.success(f"Generated via {resp.provider}")
            except Exception as e:
                st.warning(f"LLM mode failed, falling back to manual parsing. Reason: {e}")
                # Manual fallback: simple default context
                context = None
        else:
            context = None

        if context is None:
            st.info("Manual mode: showing your headlines as context bullets.")
            context = parse_market_context(
                """
                {
                  "bullets": [
                    "Manual mode: pasted headlines were used as-is.",
                    "Add OPENAI_API_KEY for automated summarization.",
                    "Ensure headlines are relevant to your holdings.",
                    "Use 3-10 short headlines for best results.",
                    "Optional: paste macro + sector + company-level items."
                  ],
                  "classification": "Mixed",
                  "sentiment": 0,
                  "confidence": 0.3
                }
                """.strip()
            )

        st.write("### Key Points")
        for b in context.bullets:
            st.write(f"- {b}")

        st.write("### Signal")
        st.write(
            {
                "classification": context.classification,
                "sentiment": context.sentiment,
                "confidence": round(context.confidence, 2),
            }
        )

        adj = compute_ai_adjustment(context)
        st.write("### AI Risk Adjustment (transparent)")
        st.write({"delta_score": adj.delta_score, "rationale": adj.rationale})

report = generate_risk_report(
    ticker,
    vol,
    var,
    cvar,
    score.score,
)

st.download_button(
    "Download Risk Report",
    report,
    file_name="sentinel_risk_report.html",
)
st.subheader("Returns Distribution")

st.bar_chart(returns)
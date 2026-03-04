import streamlit as st

from sentinel.data.loader import load_price_data, load_multiple_assets
from sentinel.portfolio.portfolio import calculate_returns
from sentinel.portfolio.returns import calculate_asset_returns, portfolio_returns
from sentinel.risk.metrics import annualized_volatility, max_drawdown
from sentinel.risk.portfolio_risk import portfolio_volatility
from sentinel.risk.risk_contribution import portfolio_risk_contribution
from sentinel.risk.scoring import compute_risk_score
from sentinel.risk.stress import apply_market_shock
from sentinel.risk.var import historical_var, historical_cvar
from sentinel.ai.llm_client import LLMClient
from sentinel.ai.market_context import build_market_context_prompt, parse_market_context
from sentinel.ai.risk_adjustment import compute_ai_adjustment
from sentinel.reporting.report_generator import generate_risk_report
from sentinel.ai.news_loader import (
    fetch_portfolio_headlines,
    build_headlines_text,
    build_markdown_news_list,
)


st.set_page_config(page_title="Sentinel Risk Intelligence", layout="wide")
st.title("Sentinel Risk Intelligence")
st.caption("Advanced Quantitative & AI-Enhanced Financial Risk Platform")

tabs = st.tabs(["Single Asset", "Portfolio", "AI Market Brief"])

# ----------------------------
# Tab 1: Single Asset Analysis
# ----------------------------
with tabs[0]:
    st.subheader("Single Asset Risk Analysis")

    ticker = st.text_input("Ticker", "AAPL", key="single_ticker")
    confidence = st.slider("Confidence level", 0.90, 0.99, 0.95, 0.01, key="single_conf")

    run_single = st.button("Run Single Asset Analysis", key="run_single")

    if run_single:
        data = load_price_data(ticker)
        # robust: use Adj Close if present, else Close
        if "Adj Close" in data.columns:
            prices = data["Adj Close"].dropna()
        else:
            prices = data["Close"].dropna()

        rets = calculate_returns(prices)

        vol = annualized_volatility(rets)
        dd = max_drawdown(prices)
        var_ = historical_var(rets, confidence)
        cvar_ = historical_cvar(rets, confidence)

        score = compute_risk_score(
            vol_annual=vol,
            max_dd=dd,
            var_95=var_,
            cvar_95=cvar_,
            concentration_hhi=None,
        )

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Annualized Volatility", f"{vol:.2%}")
        col2.metric("Max Drawdown", f"{dd:.2%}")
        col3.metric(f"Historical VaR ({int(confidence*100)}%)", f"{var_:.2%}")
        col4.metric(f"Historical CVaR ({int(confidence*100)}%)", f"{cvar_:.2%}")

        st.subheader("Risk Score")
        st.metric("Score (0–100)", f"{score.score} ({score.label})")
        if score.drivers:
            st.write("Top risk drivers:")
            for d in score.drivers:
                st.write(f"- {d}")

        st.subheader("Stress Test")
        shock = st.slider("Market shock (%)", -30, 0, -10, 1, key="single_shock")
        stress = apply_market_shock(prices, shock / 100)
        st.write(
            {
                "scenario": stress.scenario,
                "shocked_last_price": round(stress.shocked_last_price, 2),
                "pnl_pct": f"{stress.pnl_pct:.2%}",
            }
        )

        st.subheader("Price Chart")
        st.line_chart(prices)

        st.subheader("Daily Returns")
        st.line_chart(rets)

        st.subheader("Executive Risk Report")
        report_html = generate_risk_report(
            ticker=ticker,
            volatility=vol,
            var=var_,
            cvar=cvar_,
            risk_score=score.score,
        )

        st.download_button(
            "Download Risk Report (HTML)",
            report_html,
            file_name="sentinel_risk_report.html",
            mime="text/html",
            key="download_single_report",
        )

# ----------------------------
# Tab 2: Portfolio Analysis
# ----------------------------
with tabs[1]:
    st.subheader("Portfolio Risk Analysis")

    st.caption("Edit tickers/weights below (JSON format). Example: {\"AAPL\": 0.4, \"MSFT\": 0.3, \"SPY\": 0.3}")

    portfolio_text = st.text_area(
        "Portfolio weights (JSON)",
        value='{"AAPL": 0.4, "MSFT": 0.3, "SPY": 0.3}',
        height=90,
        key="portfolio_json",
    )

    start_date = st.text_input("Start date (YYYY-MM-DD)", "2018-01-01", key="portfolio_start")

    run_portfolio = st.button("Run Portfolio Analysis", key="run_portfolio")

    if run_portfolio:
        import json

        try:
            portfolio = json.loads(portfolio_text)
            if not isinstance(portfolio, dict) or len(portfolio) < 2:
                raise ValueError("Portfolio must be a JSON object with at least 2 tickers.")
            portfolio = {str(k).upper(): float(v) for k, v in portfolio.items()}
        except Exception as e:
            st.error(f"Invalid portfolio JSON. Please fix and retry. Details: {e}")
            st.stop()

        tickers = list(portfolio.keys())

        prices_df = load_multiple_assets(tickers, start=start_date)
        returns_df = calculate_asset_returns(prices_df)

        port_ret = portfolio_returns(returns_df, portfolio)
        port_vol = portfolio_volatility(returns_df, portfolio)

        st.write("### Portfolio Summary")
        col1, col2 = st.columns(2)
        col1.metric("Portfolio Annualized Volatility", f"{port_vol:.2%}")
        col2.metric("Average Daily Return", f"{port_ret.mean():.3%}")

        st.write("### Portfolio Returns")
        st.line_chart(port_ret)

        st.write("### Asset Prices")
        st.line_chart(prices_df)

        st.write("### Risk Contribution (Variance-based)")
        rc = portfolio_risk_contribution(returns_df, portfolio)
        st.bar_chart(rc)

        st.write("### Weights vs Risk Contribution")
        weights_series = st.session_state.get("weights_series")
        # create a local weights series for display
        import pandas as pd

        w = pd.Series(portfolio)
        w = w / w.sum()
        compare = pd.DataFrame({"weight": w, "risk_contribution": rc}).fillna(0.0)
        st.dataframe(compare)

# ----------------------------
# Tab 3: AI Market Brief (Hybrid + Auto News + Caching)
# ----------------------------
with tabs[2]:
    st.subheader("AI Market Brief (Hybrid)")
    st.caption(
        "Works without API key in manual mode. "
        "Optionally fetches latest headlines via free RSS (cached). "
        "With OPENAI_API_KEY + openai installed, it can generate automatically."
    )

    portfolio_tickers = st.text_input(
        "Tickers context (comma separated)",
        "AAPL,MSFT,SPY",
        key="ai_tickers",
    )
    tickers = [t.strip().upper() for t in portfolio_tickers.split(",") if t.strip()]

    # ---- Cached news fetch (15 min TTL) ----
    @st.cache_data(ttl=900, show_spinner=False)
    def cached_fetch_news(tickers_tuple: tuple[str, ...]):
        return fetch_portfolio_headlines(list(tickers_tuple))

    st.markdown("### News Ingestion (Free RSS)")
    auto_news = st.checkbox(
        "Fetch latest news automatically (RSS, free)",
        value=True,
        key="ai_auto_news",
    )

    if "ai_news_items" not in st.session_state:
        st.session_state["ai_news_items"] = []
    if "ai_fetched_headlines" not in st.session_state:
        st.session_state["ai_fetched_headlines"] = ""

    col_a, col_b = st.columns([1, 1])
    with col_a:
        fetch_clicked = st.button("Fetch News", key="ai_fetch_news")
    with col_b:
        clear_clicked = st.button("Clear Fetched News", key="ai_clear_news")

    if clear_clicked:
        st.session_state["ai_news_items"] = []
        st.session_state["ai_fetched_headlines"] = ""
        st.success("Fetched news cleared.")

    if fetch_clicked and auto_news:
        try:
            items = cached_fetch_news(tuple(tickers))
            st.session_state["ai_news_items"] = items

            # For LLM prompt:
            fetched_text = build_headlines_text(items, max_items=12)
            st.session_state["ai_fetched_headlines"] = fetched_text

            st.success(f"Fetched {len(items)} headlines (cached).")
        except Exception as e:
            st.warning(
                f"News fetch failed. You can still paste headlines manually. Reason: {e}"
            )

    items = st.session_state.get("ai_news_items", [])
    fetched_block = st.session_state.get("ai_fetched_headlines", "").strip()

    if items:
        st.markdown("#### Latest Headlines (clickable)")
        st.markdown(build_markdown_news_list(items, max_items=12))
        st.caption("Tip: Use the links to quickly verify relevance and timing.")
    else:
        st.info("No fetched headlines yet. Click 'Fetch News' to load current headlines.")

    # --- Manual headlines ---
    st.markdown("### Manual Headlines / Notes")
    manual_headlines = st.text_area(
        "Paste additional headlines or notes",
        value="Fed hints at higher rates; tech stocks volatile; oil prices rise.",
        height=140,
        key="ai_headlines",
    )

    # Combine auto + manual into one prompt input
    combined_headlines = manual_headlines.strip()
    if fetched_block:
        combined_headlines = (fetched_block + "\n\n" + combined_headlines).strip()

    # --- LLM toggle ---
    use_llm = st.checkbox(
        "Use LLM (requires OPENAI_API_KEY)",
        value=False,
        key="ai_use_llm",
    )

    st.markdown("### Generate Market Brief")
    if st.button("Generate Market Brief", key="ai_generate"):
        context = None

        if use_llm:
            try:
                client = LLMClient()
                prompt = build_market_context_prompt(combined_headlines, tickers)
                resp = client.generate(prompt)
                context = parse_market_context(resp.text)
                st.success(f"Generated via {resp.provider}")
            except Exception as e:
                st.warning(f"LLM mode failed, using manual fallback. Reason: {e}")

        if context is None:
            context = parse_market_context(
                """
                {
                  "bullets": [
                    "Manual mode: headlines were used without LLM summarization.",
                    "Add OPENAI_API_KEY to enable automated market risk briefs.",
                    "Include macro + sector + company-level headlines for best results.",
                    "Keep headlines short and relevant to your holdings.",
                    "Use this brief as qualitative context, not as a trading signal."
                  ],
                  "classification": "Mixed",
                  "sentiment": 0,
                  "confidence": 0.3
                }
                """.strip()
            )
            st.info("Manual fallback context used.")

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
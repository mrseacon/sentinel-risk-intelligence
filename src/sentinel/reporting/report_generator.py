from datetime import datetime


def generate_risk_report(
    ticker: str,
    volatility: float,
    var: float,
    cvar: float,
    risk_score: int,
):

    now = datetime.now().strftime("%Y-%m-%d")

    html = f"""
    <html>
    <head>
    <title>Sentinel Risk Intelligence Report</title>
    </head>

    <body>

    <h1>Sentinel Risk Intelligence</h1>
    <h2>Portfolio Risk Report</h2>

    <p><b>Date:</b> {now}</p>
    <p><b>Asset:</b> {ticker}</p>

    <h3>Risk Metrics</h3>

    <ul>
        <li>Volatility: {volatility:.2%}</li>
        <li>Value-at-Risk (95%): {var:.2%}</li>
        <li>CVaR (Expected Shortfall): {cvar:.2%}</li>
        <li>Risk Score: {risk_score}</li>
    </ul>

    <h3>Interpretation</h3>

    <p>
    This report summarizes the current risk profile of the analyzed asset.
    The metrics combine historical return analysis with stress testing
    and risk scoring to provide a structured overview.
    </p>

    </body>
    </html>
    """

    return html


def save_report(html, filename="risk_report.html"):

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html)

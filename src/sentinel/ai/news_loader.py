from __future__ import annotations

import time
import urllib.parse
from collections.abc import Iterable
from dataclasses import dataclass

import feedparser
import requests


@dataclass(frozen=True)
class NewsItem:
    title: str
    source: str
    published: str | None
    link: str | None


def _google_news_rss_url(query: str) -> str:
    q = urllib.parse.quote(query)
    # hl=en-US: English output; gl=US region; ceid=US:en
    return f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"


def fetch_google_news_rss(
    query: str,
    limit: int = 10,
    timeout: int = 10,
) -> list[NewsItem]:
    """
    Fetch headlines from Google News RSS for a query (free, no key).
    """
    url = _google_news_rss_url(query)

    headers = {
        "User-Agent": "SentinelRiskIntelligence/1.0 (+https://github.com/your-repo)"
    }

    resp = requests.get(url, headers=headers, timeout=timeout)
    resp.raise_for_status()

    feed = feedparser.parse(resp.text)
    items: list[NewsItem] = []

    for entry in feed.entries[:limit]:
        title = getattr(entry, "title", "").strip()
        link = getattr(entry, "link", None)
        published = getattr(entry, "published", None)

        # Google News titles often look like: "Headline - Source"
        source = "Google News"
        if " - " in title:
            parts = title.rsplit(" - ", 1)
            if len(parts) == 2:
                title, source = parts[0].strip(), parts[1].strip()

        if title:
            items.append(
                NewsItem(
                    title=title,
                    source=source,
                    published=published,
                    link=link,
                )
            )

    return items


def build_headlines_text(items: Iterable[NewsItem], max_items: int = 10) -> str:
    """
    Convert news items into a compact text block suitable for LLM prompting.
    """
    lines = []
    for i, it in enumerate(list(items)[:max_items], start=1):
        meta = f"{it.source}"
        if it.published:
            meta += f" | {it.published}"
        lines.append(f"{i}. {it.title} ({meta})")
    return "\n".join(lines).strip()


def fetch_portfolio_headlines(
    tickers: list[str],
    limit_per_ticker: int = 4,
    throttle_seconds: float = 0.4,
) -> list[NewsItem]:
    """
    Fetch a small set of headlines per ticker + a macro query.
    Throttled to be polite to RSS endpoints.
    """
    all_items: list[NewsItem] = []

    # Macro / market context first
    macro_queries = [
        "US stock market risk",
        "Federal Reserve rates market impact",
        "inflation outlook market",
    ]
    for q in macro_queries:
        all_items.extend(fetch_google_news_rss(q, limit=3))
        time.sleep(throttle_seconds)

    # Ticker/company context
    for t in tickers:
        all_items.extend(fetch_google_news_rss(t, limit=limit_per_ticker))
        time.sleep(throttle_seconds)

    # Deduplicate by title
    seen = set()
    deduped: list[NewsItem] = []
    for it in all_items:
        key = it.title.lower()
        if key not in seen:
            deduped.append(it)
            seen.add(key)

    return deduped[: min(25, len(deduped))]


def build_markdown_news_list(items: Iterable[NewsItem], max_items: int = 10) -> str:
    """
    Create a markdown list with clickable links for Streamlit.
    """
    md_lines: list[str] = []
    for i, it in enumerate(list(items)[:max_items], start=1):
        title = it.title.replace("\n", " ").strip()
        source = (it.source or "Source").strip()

        if it.link:
            md_lines.append(f"{i}. [{title}]({it.link}) — *{source}*")
        else:
            md_lines.append(f"{i}. {title} — *{source}*")

    return "\n".join(md_lines).strip()


def classify_headline_bucket(title: str, tickers: list[str]) -> str:
    """
    Very lightweight classifier for UI filtering.
    Returns: "macro" or "company"
    """
    t = title.upper()
    if any(sym.upper() in t for sym in tickers):
        return "company"
    return "macro"

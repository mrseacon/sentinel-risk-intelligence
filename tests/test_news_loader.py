from sentinel.ai.news_loader import (
    NewsItem,
    build_headlines_text,
    build_markdown_news_list,
)


def test_build_headlines_text():
    items = [
        NewsItem(
            title="AAPL earnings beat expectations",
            source="Reuters",
            published=None,
            link="https://example.com/a",
        ),
        NewsItem(
            title="Fed signals rate pause",
            source="Bloomberg",
            published="Mon, 01 Jan 2026",
            link="https://example.com/b",
        ),
    ]
    text = build_headlines_text(items, max_items=10)
    assert "AAPL earnings" in text
    assert "Fed signals" in text


def test_build_markdown_news_list_contains_links():
    items = [
        NewsItem(
            title="AAPL earnings beat expectations",
            source="Reuters",
            published=None,
            link="https://example.com/a",
        ),
    ]
    md = build_markdown_news_list(items, max_items=10)
    assert "[" in md and "](" in md  # markdown link pattern
    assert "https://example.com/a" in md

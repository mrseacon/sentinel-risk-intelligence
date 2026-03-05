from sentinel.ai.market_context import parse_market_context


def test_parse_market_context_valid_json():
    text = """
    {
      "bullets": ["a","b","c","d","e"],
      "classification": "Macro",
      "sentiment": -1,
      "confidence": 0.8
    }
    """.strip()

    ctx = parse_market_context(text)
    assert ctx.sentiment == -1
    assert 0.0 <= ctx.confidence <= 1.0
    assert len(ctx.bullets) == 5

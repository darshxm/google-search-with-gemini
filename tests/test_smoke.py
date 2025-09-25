from gemini_grounding import __version__, extract_markdown_links

def test_version():
    assert isinstance(__version__, str)

def test_extract_links():
    md = "See [1](https://a.com) and [2](https://b.com)"
    out = extract_markdown_links(md)
    assert out == [(1, "https://a.com"), (2, "https://b.com")]


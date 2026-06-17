from app.preprocessing.cleaner import clean_text, remove_html, remove_urls


def test_remove_html():
    assert remove_html("<p>Hello</p>") == " Hello "


def test_remove_urls():
    text = "Visit https://example.com for more"
    assert "https" not in remove_urls(text)


def test_clean_text_empty():
    assert clean_text("") == ""
    assert clean_text("   ") == ""

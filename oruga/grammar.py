"""Optional post-hoc grammar correction.

Replacing words by synonyms can introduce small grammatical slips (agreement,
articles, ...). The legacy scripts ran the result through LanguageTool. That
logic lived in a duplicated ``correct_mistakes`` function; it now lives here and
degrades gracefully to a no-op when ``language_tool_python`` (which needs Java)
is unavailable.
"""

from __future__ import annotations

__all__ = ["correct_grammar", "is_available"]

_TOOL = None


def is_available() -> bool:
    try:
        import language_tool_python  # noqa: F401
        return True
    except Exception:
        return False


def _get_tool(lang: str = "en-US"):
    global _TOOL
    if _TOOL is None:
        import language_tool_python
        _TOOL = language_tool_python.LanguageTool(lang)
    return _TOOL


def correct_grammar(text: str, lang: str = "en-US") -> str:
    """Return ``text`` with LanguageTool's top suggestion applied to each match.

    If LanguageTool is not installed/usable, the input text is returned
    unchanged so pipelines never break.
    """
    try:
        tool = _get_tool(lang)
        matches = [m for m in tool.check(text) if m.replacements]
    except Exception:
        return text

    chars = list(text)
    for m in matches:
        start, end = m.offset, m.offset + m.errorLength
        replacement = m.replacements[0]
        chars[start] = replacement
        for i in range(start + 1, end):
            chars[i] = ""
    return "".join(chars)

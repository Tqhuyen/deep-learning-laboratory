from __future__ import annotations

import os

import pytest

from laboratory.agents.environment import load_environment, parse_environment_file, variable


def test_parses_comments_exports_quotes_and_inline_comments():
    text = """
# a comment
GROQ_API_KEY=gsk_123   # trailing comment
export MISTRAL_API_KEY="abc # not a comment"
CUSTOM_BASE_URL='http://localhost:11434/v1'
ESCAPED="line\\nbreak \\"quoted\\""
EMPTY=
URL_WITH_HASH=http://host/#anchor
"""
    pairs = parse_environment_file(text)
    assert pairs == {
        "GROQ_API_KEY": "gsk_123",
        "MISTRAL_API_KEY": "abc # not a comment",
        "CUSTOM_BASE_URL": "http://localhost:11434/v1",
        "ESCAPED": 'line\nbreak "quoted"',
        "EMPTY": "",
        "URL_WITH_HASH": "http://host/#anchor",
    }


@pytest.mark.parametrize("line", ["NOT AN ASSIGNMENT", "1BAD=x", 'OPEN="unterminated', 'X="a" junk'])
def test_malformed_lines_name_their_line_number(line):
    with pytest.raises(ValueError, match="line 2"):
        parse_environment_file(f"OK=1\n{line}\n")


def test_existing_variables_win_unless_overridden(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text("LAB_TEST_KEY=from-file\nLAB_TEST_OTHER=file-only\n", encoding="utf-8")
    monkeypatch.setenv("LAB_TEST_KEY", "from-shell")
    monkeypatch.setenv("LAB_TEST_OTHER", "registered so teardown removes it")
    monkeypatch.delenv("LAB_TEST_OTHER")

    load_environment(path)
    assert variable(os.environ, "LAB_TEST_KEY") == "from-shell"
    assert variable(os.environ, "LAB_TEST_OTHER") == "file-only"

    load_environment(path, override=True)
    assert variable(os.environ, "LAB_TEST_KEY") == "from-file"


def test_missing_file_is_not_an_error(tmp_path):
    assert load_environment(tmp_path / "absent.env") == {}


def test_blank_values_count_as_unset():
    assert variable({"KEY": "   "}, "KEY") is None
    assert variable({"KEY": " value "}, "KEY") == "value"

"""Credentials from the process environment and a repository-level ``.env``.

Keys never live in code, configs or run artifacts. :func:`load_environment`
reads ``KEY=VALUE`` lines from ``.env`` at the repository root (copy
``.env.example`` to start) into ``os.environ``. Variables that are already set
win, so shell exports and CI secrets override the file.

Accepted syntax, one assignment per line::

    # comment
    GROQ_API_KEY=gsk_...
    export MISTRAL_API_KEY="abc"      # quotes and a trailing comment
    CUSTOM_BASE_URL='http://localhost:11434/v1'
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_ESCAPES = {"n": "\n", "t": "\t", '"': '"', "\\": "\\"}


def repository_environment_file() -> Path:
    """``<repository>/.env``; the file may not exist."""
    return Path(__file__).resolve().parents[2] / ".env"


def _unquote(value: str, line_number: int) -> str:
    quote = value[0]
    characters: list[str] = []
    index = 1
    while index < len(value):
        character = value[index]
        if character == quote:
            rest = value[index + 1 :].strip()
            if rest and not rest.startswith("#"):
                raise ValueError(f".env line {line_number}: unexpected text after the closing quote")
            return "".join(characters)
        if quote == '"' and character == "\\" and index + 1 < len(value):
            index += 1
            characters.append(_ESCAPES.get(value[index], "\\" + value[index]))
        else:
            characters.append(character)
        index += 1
    raise ValueError(f".env line {line_number}: missing closing {quote}")


def parse_environment_file(text: str) -> dict[str, str]:
    """Parse ``.env`` text into ``{name: value}``; malformed lines raise ``ValueError``."""
    pairs: dict[str, str] = {}
    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        name, separator, value = line.partition("=")
        name = name.strip()
        if not separator or not _NAME.match(name):
            raise ValueError(f".env line {line_number}: expected NAME=value, got {raw_line!r}")
        value = value.strip()
        if value[:1] in ("'", '"'):
            value = _unquote(value, line_number)
        else:
            # An unquoted value ends at " #" so trailing comments stay out of keys.
            comment = re.search(r"\s#", value)
            if comment:
                value = value[: comment.start()].rstrip()
        pairs[name] = value
    return pairs


def load_environment(path: Path | str | None = None, *, override: bool = False) -> dict[str, str]:
    """Copy the pairs from ``path`` (default ``<repository>/.env``) into ``os.environ``.

    Returns the parsed pairs. A missing file is not an error: keyless providers
    work without one. Existing variables are kept unless ``override`` is set.
    """
    path = Path(path) if path is not None else repository_environment_file()
    if not path.exists():
        return {}
    pairs = parse_environment_file(path.read_text(encoding="utf-8"))
    for name, value in pairs.items():
        if override or name not in os.environ:
            os.environ[name] = value
    return pairs


def variable(environment: Mapping[str, str], name: str) -> str | None:
    """``environment[name]`` with surrounding whitespace removed; empty means unset."""
    value = environment.get(name, "").strip()
    return value or None

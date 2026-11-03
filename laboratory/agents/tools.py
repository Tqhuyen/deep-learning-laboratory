"""Tools an agent can call, described once and usable by every architecture.

A :class:`Tool` carries a JSON-schema description, so the same object is sent
as a native function definition (OpenAI ``tools``) *and* rendered as text for
prompt-based agents such as ReAct, which run on models without function
calling. A tool failure becomes the observation ``"Error: ..."`` instead of an
exception: the agent sees its mistake and can recover, and recovering is part
of what an architecture comparison measures.
"""

from __future__ import annotations

import ast
import difflib
import json
import math
import operator
import re
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: Mapping[str, Any]
    """JSON schema of the arguments object."""
    function: Callable[..., Any]

    def schema(self) -> dict[str, Any]:
        """OpenAI function-calling definition."""
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": dict(self.parameters)},
        }

    def signature(self) -> str:
        """``calculator(expression: string): ...`` for text prompts."""
        properties = self.parameters.get("properties", {})
        required = set(self.parameters.get("required", ()))
        arguments = ", ".join(
            f"{name}: {spec.get('type', 'any')}{'' if name in required else ' (optional)'}"
            for name, spec in properties.items()
        )
        return f"{self.name}({arguments}): {self.description}"

    def _arguments(self, arguments: str | Mapping[str, Any]) -> dict[str, Any]:
        properties = self.parameters.get("properties", {})
        if isinstance(arguments, str):
            text = arguments.strip()
            try:
                value = json.loads(text) if text else {}
            except json.JSONDecodeError:
                value = None
            if not isinstance(value, dict):
                # Prompt-based agents often write the bare argument: "Action Input: 17 * 23".
                if len(properties) != 1:
                    raise ValueError("arguments must be a JSON object")
                value = {next(iter(properties)): text.strip("'\"") if value is None else value}
            arguments = value
        missing = [name for name in self.parameters.get("required", ()) if name not in arguments]
        if missing:
            raise ValueError(f"missing argument(s) {', '.join(missing)}")
        unknown = sorted(set(arguments) - set(properties))
        if properties and unknown:
            raise ValueError(f"unknown argument(s) {', '.join(unknown)}; expected {', '.join(properties)}")
        return dict(arguments)

    def invoke(self, arguments: str | Mapping[str, Any]) -> str:
        """Run the tool; argument and runtime errors come back as ``"Error: ..."``."""
        try:
            keyword_arguments = self._arguments(arguments)
        except ValueError as error:
            return f"Error: {error}"
        try:
            return str(self.function(**keyword_arguments))
        except Exception as error:  # noqa: BLE001 - the observation is the agent's feedback
            return f"Error: {type(error).__name__}: {error}"


class Toolbox:
    """The tools available to one agent, addressable by name."""

    def __init__(self, tools: Iterable[Tool]) -> None:
        self.tools = tuple(tools)
        self._by_name = {tool.name: tool for tool in self.tools}
        if len(self._by_name) != len(self.tools):
            raise ValueError("tool names must be unique")

    def __iter__(self) -> Iterator[Tool]:
        return iter(self.tools)

    def __len__(self) -> int:
        return len(self.tools)

    @property
    def names(self) -> list[str]:
        return [tool.name for tool in self.tools]

    def describe(self) -> str:
        return "\n".join(f"- {tool.signature()}" for tool in self.tools)

    def run(self, name: str, arguments: str | Mapping[str, Any]) -> str:
        tool = self._by_name.get(name.strip()) or self._by_name.get(name.strip().lower())
        if tool is None:
            return f"Error: unknown tool {name!r}; available tools: {', '.join(self.names)}"
        return tool.invoke(arguments)


# --- calculator -----------------------------------------------------------------

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS: dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt, "log": math.log, "log10": math.log10, "exp": math.exp,
    "sin": math.sin, "cos": math.cos, "tan": math.tan, "abs": abs, "round": round,
    "min": min, "max": max, "floor": math.floor, "ceil": math.ceil,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}


def _evaluate(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _evaluate(node.left), _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and (abs(right) > 1000 or (abs(left) > 1 and abs(right) * math.log10(abs(left)) > 300)):
            raise ValueError("power is too large")
        return _BINARY[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_evaluate(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS and not node.keywords:
        return _FUNCTIONS[node.func.id](*(_evaluate(argument) for argument in node.args))
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    raise ValueError(f"unsupported expression: {ast.unparse(node)}")


def evaluate_arithmetic(expression: str) -> float:
    """Evaluate arithmetic safely: numbers, operators, parentheses and a few math functions.

    Writing conventions models use are accepted: ``1,234`` thousands
    separators, ``×``, ``÷``, ``−`` and ``^`` for powers.
    """
    text = expression.strip().rstrip("=").strip()
    text = text.replace("×", "*").replace("÷", "/").replace("−", "-").replace("^", "**")
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    if len(text) > 500:
        raise ValueError("expression is longer than 500 characters")
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        raise ValueError(f"not an arithmetic expression: {expression!r}") from None
    return _evaluate(tree.body)


def format_number(value: float) -> str:
    """Integers print exactly; other values with 10 significant digits."""
    if isinstance(value, float) and math.isfinite(value) and value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def calculator() -> Tool:
    return Tool(
        name="calculator",
        description=(
            "Evaluate an arithmetic expression exactly. Supports + - * / // % ** and parentheses, "
            "plus sqrt, log, exp, sin, cos, round, min, max."
        ),
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "for example (3141 - 2718) * 59 / 3"}},
            "required": ["expression"],
        },
        function=lambda expression: format_number(evaluate_arithmetic(expression)),
    )


# --- lookup ---------------------------------------------------------------------


def lookup_tool(
    entries: Mapping[str, str],
    *,
    name: str = "lookup",
    description: str = "Look up an encyclopedia entry by its title (a person, place or thing) and return its text.",
) -> Tool:
    """Exact-title lookup over ``entries`` that forgives case, articles and small typos."""
    titles = {title.lower(): title for title in entries}

    def lookup(title: str) -> str:
        key = re.sub(r"^(the|a|an)\s+", "", title.strip().strip("'\"").lower())
        if key in titles:
            return entries[titles[key]]
        close = difflib.get_close_matches(key, list(titles), n=3, cutoff=0.8)
        if close:
            return entries[titles[close[0]]]
        containing = [original for lowered, original in titles.items() if key and (key in lowered or lowered in key)]
        if len(containing) == 1:
            return entries[containing[0]]
        suggestions = containing or [titles[s] for s in difflib.get_close_matches(key, list(titles), n=3, cutoff=0.5)]
        hint = f" Similar titles: {', '.join(suggestions)}." if suggestions else ""
        return f"No entry titled {title!r}.{hint}"

    return Tool(
        name=name,
        description=description,
        parameters={
            "type": "object",
            "properties": {"title": {"type": "string", "description": "exact entry title, e.g. Velmora Bridge"}},
            "required": ["title"],
        },
        function=lookup,
    )

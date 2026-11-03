from __future__ import annotations

import pytest

from laboratory.agents.tools import Tool, Toolbox, calculator, evaluate_arithmetic, lookup_tool


@pytest.mark.parametrize(
    ("expression", "value"),
    [
        ("48763 * 2917", 142241671),
        ("48,763 × 2,917", 142241671),
        ("(3141 − 2718) * 59 ÷ 3", 8319),
        ("2^31 - 1", 2147483647),
        ("1 + 2 * 3 ** 2", 19),
        ("-sqrt(16) + max(1, 2)", -2),
        ("17 * 23 =", 391),
    ],
)
def test_calculator_reads_the_way_models_write_arithmetic(expression, value):
    assert evaluate_arithmetic(expression) == value


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('echo hi')", "(1).__class__", "open('x')", "[1, 2]", "lambda: 1", "9 ** 9999", "x + 1"],
)
def test_calculator_rejects_everything_but_arithmetic(expression):
    with pytest.raises(ValueError):
        evaluate_arithmetic(expression)


def test_calculator_tool_formats_results_and_reports_errors_as_observations():
    tool = calculator()
    assert tool.invoke({"expression": "7200 / 37.5"}) == "192"
    assert tool.invoke({"expression": "0.17 * 8460 + 1234"}) == "2672.2"
    assert tool.invoke({"expression": "1 / 0"}).startswith("Error: ZeroDivisionError")


def test_arguments_may_be_json_or_a_bare_value_for_one_parameter_tools():
    tool = calculator()
    assert tool.invoke('{"expression": "2 + 2"}') == "4"
    assert tool.invoke("2 + 2") == "4"
    assert tool.invoke('"2 + 2"') == "4"
    assert tool.invoke("{}") == "Error: missing argument(s) expression"
    assert tool.invoke('{"expression": "1", "unit": "m"}').startswith("Error: unknown argument(s) unit")


def test_multi_parameter_tools_require_a_json_object():
    tool = Tool("add", "add", {"type": "object", "properties": {"a": {}, "b": {}}, "required": ["a", "b"]}, lambda a, b: a + b)
    assert tool.invoke('{"a": 1, "b": 2}') == "3"
    assert tool.invoke("1 2") == "Error: arguments must be a JSON object"


def test_toolbox_routes_by_name_and_lists_tools_for_text_prompts():
    toolbox = Toolbox([calculator()])
    assert toolbox.run("calculator", "6 * 7") == "42"
    assert toolbox.run("search", "x").startswith("Error: unknown tool 'search'; available tools: calculator")
    assert "calculator(expression: string)" in toolbox.describe()
    with pytest.raises(ValueError):
        Toolbox([calculator(), calculator()])


ENTRIES = {"Kestrel Bay": "harbour town", "Kestrel Bay Lighthouse": "38 metres", "Velmora Bridge": "completed 1911"}


@pytest.mark.parametrize(
    ("title", "result"),
    [
        ("velmora bridge", "completed 1911"),
        ("The Velmora Bridge", "completed 1911"),
        ("Velmora", "completed 1911"),
        ("Kestrel Bay Lighthuose", "38 metres"),
    ],
)
def test_lookup_forgives_case_articles_and_typos(title, result):
    assert lookup_tool(ENTRIES).invoke({"title": title}) == result


def test_ambiguous_lookup_returns_suggestions_not_an_entry():
    result = lookup_tool(ENTRIES).invoke({"title": "Kestrel"})
    assert result.startswith("No entry titled 'Kestrel'.")
    assert "Kestrel Bay Lighthouse" in result

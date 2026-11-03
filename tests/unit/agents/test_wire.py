from __future__ import annotations

import json

import pytest

from laboratory.agents.messages import Message, ToolCall
from laboratory.agents.providers import resolve_endpoint
from laboratory.agents.tools import calculator
from laboratory.agents.wire import (
    PROXY_ERROR_PREFIX,
    ProviderError,
    chat_request,
    classify,
    exhausted_for,
    parse_chat_response,
    parse_responses_payload,
    responses_request,
    same_model,
    seconds_until,
)

KEYS = {name: "key" for name in ("NVIDIA_API_KEY", "COHERE_API_KEY", "MOONDREAM_API_KEY", "ACLIDE_API_KEY", "GROQ_API_KEY")}
CONVERSATION = [
    Message.system("be brief"),
    Message.user("what is 2+2?"),
    Message.assistant("", [ToolCall("call_1", "calculator", '{"expression": "2+2"}')]),
    Message.tool(ToolCall("call_1", "calculator", "{}"), "4"),
]


def endpoint(provider, model="m", **overrides):
    return resolve_endpoint(provider, model, {**KEYS}, **overrides)


def test_chat_messages_carry_only_wire_fields():
    body = chat_request(endpoint("groq"), CONVERSATION)
    assert body["messages"][2] == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "calculator", "arguments": '{"expression": "2+2"}'}}],
    }
    assert body["messages"][3] == {"role": "tool", "tool_call_id": "call_1", "content": "4"}


def test_single_tool_call_providers_forbid_parallel_calls_only_when_tools_are_sent():
    assert chat_request(endpoint("nvidia"), CONVERSATION, [calculator()])["parallel_tool_calls"] is False
    assert "parallel_tool_calls" not in chat_request(endpoint("nvidia"), CONVERSATION)
    assert "parallel_tool_calls" not in chat_request(endpoint("groq"), CONVERSATION, [calculator()])


def test_ai_horde_floors_and_defaults_max_tokens():
    horde = resolve_endpoint("aihorde", "m", {})
    assert chat_request(horde, CONVERSATION, max_tokens=5)["max_tokens"] == 16
    assert chat_request(horde, CONVERSATION)["max_tokens"] == 512
    assert "max_tokens" not in chat_request(endpoint("groq"), CONVERSATION)


def test_cohere_tool_schemas_lose_rejected_keywords_at_every_depth():
    tool = calculator()
    nested = type(tool)(
        tool.name,
        tool.description,
        {"$schema": "x", "type": "object", "additionalProperties": False,
         "properties": {"expression": {"type": "object", "additionalProperties": False}}},
        tool.function,
    )
    sent = json.dumps(chat_request(endpoint("cohere"), CONVERSATION, [nested])["tools"])
    assert "additionalProperties" not in sent and "$schema" not in sent
    assert "additionalProperties" in json.dumps(chat_request(endpoint("groq"), CONVERSATION, [nested])["tools"])


def test_moondream_renames_max_tokens_and_drops_stop_sequences():
    body = chat_request(endpoint("moondream", tools=False), [Message.user("hi")], max_tokens=64, stop=["\nObservation:"])
    assert body["max_completion_tokens"] == 64 and "max_tokens" not in body and "stop" not in body


def test_tools_sent_to_a_tool_less_endpoint_are_rejected_before_the_network():
    with pytest.raises(ProviderError) as raised:
        chat_request(endpoint("groq", tools=False), CONVERSATION, [calculator()])
    assert raised.value.kind == "bad_request"


def test_responses_api_encodes_tool_turns_as_typed_items():
    body = responses_request(endpoint("aclide"), CONVERSATION, [calculator()], max_tokens=100)
    assert body["input"][2] == {"type": "function_call", "call_id": "call_1", "name": "calculator", "arguments": '{"expression": "2+2"}'}
    assert body["input"][3] == {"type": "function_call_output", "call_id": "call_1", "output": "4"}
    assert body["tools"][0]["type"] == "function" and body["tools"][0]["name"] == "calculator"
    assert body["max_output_tokens"] == 100 and body["store"] is False


def test_chat_response_normalisation():
    payload = {
        "model": "served-model",
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {
                "content": [{"type": "text", "text": "<think>plan</think>"}, {"type": "text", "text": "ok"}],
                "tool_calls": [{"function": {"name": "calculator", "arguments": {"expression": "1+1"}}}],
            },
        }],
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
    }
    message, reasoning, finish, usage, served = parse_chat_response(payload)
    assert (message.content, reasoning, finish, served) == ("ok", "plan", "tool_calls", "served-model")
    assert message.tool_calls[0] == ToolCall("call_0", "calculator", '{"expression": "1+1"}')
    assert (usage.prompt_tokens, usage.completion_tokens) == (7, 3)


def test_separate_reasoning_fields_win_over_think_tags():
    _, reasoning, *_ = parse_chat_response({"choices": [{"message": {"content": "pong", "reasoning_content": "hidden"}}]})
    assert reasoning == "hidden"
    message, reasoning, *_ = parse_chat_response({"choices": [{"message": {"content": "trace</think>answer"}}]})
    assert (reasoning, message.content) == ("trace", "answer")


@pytest.mark.parametrize(
    "payload",
    [
        {"error": {"message": "upstream exploded"}},
        {"choices": []},
        {"choices": [{"message": {"content": PROXY_ERROR_PREFIX + "502)** upstream"}}]},
        ["not", "an", "object"],
    ],
)
def test_successful_status_with_a_broken_body_is_transient(payload):
    with pytest.raises(ProviderError) as raised:
        parse_chat_response(payload)
    assert raised.value.kind == "transient"


def test_responses_payload_collects_text_calls_and_reasoning():
    payload = {
        "status": "incomplete",
        "output": [
            {"type": "reasoning", "summary": [{"text": "think"}]},
            {"type": "message", "content": [{"type": "output_text", "text": "partial"}]},
        ],
        "usage": {"input_tokens": 4, "output_tokens": 2},
    }
    message, reasoning, finish, usage, _ = parse_responses_payload(payload)
    assert (message.content, reasoning, finish, usage.total_tokens) == ("partial", "think", "length", 6)

    call = {"output": [{"type": "function_call", "call_id": "c9", "name": "lookup", "arguments": '{"title": "x"}'}]}
    message, _, finish, _, _ = parse_responses_payload(call)
    assert finish == "tool_calls" and message.tool_calls[0].identifier == "c9"
    with pytest.raises(ProviderError):
        parse_responses_payload({"status": "failed"})


NOW = 1_800_000_000.0


@pytest.mark.parametrize(
    ("status", "headers", "body", "kind", "retry_after"),
    [
        (429, {"retry-after": "38"}, {"error": {"message": "slow down"}}, "rate_limited", 38.0),
        (429, {}, {"error": {"message": "insufficient balance (1008)"}}, "unavailable", None),
        (503, {}, {"error": {"message": "all busy", "code": "get_channel_failed"}}, "rate_limited", None),
        (503, {}, {"error": {"message": "no such name", "code": "model_not_found"}}, "unavailable", None),
        (401, {}, {"error": {"message": "who are you"}}, "unauthorized", None),
        (400, {}, {"error": {"message": "API key not valid. Please pass a valid API key."}}, "unauthorized", None),
        (403, {}, {"error": {"message": "This is a premium model"}}, "unavailable", None),
        (404, {}, {"detail": "Not Found"}, "unavailable", None),
        (400, {}, {"error": {"message": "The model `x` does not exist"}}, "unavailable", None),
        (400, {}, {"error": {"message": "context length exceeded"}}, "bad_request", None),
        (502, {}, "<html>bad gateway</html>", "transient", None),
    ],
)
def test_classification_decides_what_the_router_does(status, headers, body, kind, retry_after):
    raw = (json.dumps(body) if not isinstance(body, str) else body).encode()
    error = classify(status, headers, raw, NOW)
    assert error.kind == kind
    assert error.retry_after == retry_after


@pytest.mark.parametrize(
    ("value", "seconds"),
    [
        ("38", 38.0),
        ("1m30s", 90.0),
        ("59.56s", 59.56),
        ("120ms", 0.12),
        ("2h", 7200.0),
        (str(NOW + 25), 25.0),
        (str((NOW + 25) * 1000), 25.0),
        ("Fri, 15 Jan 2027 08:00:25 GMT", 25.0),
    ],
)
def test_reset_headers_in_every_common_format(value, seconds):
    assert seconds_until(value, NOW) == pytest.approx(seconds)


def test_unreadable_reset_header_is_ignored():
    assert seconds_until("soon", NOW) is None


def test_zero_remaining_requests_means_wait_for_the_window():
    assert exhausted_for({"x-ratelimit-remaining-requests": "0", "x-ratelimit-reset-requests": "2m"}, NOW) == 120.0
    assert exhausted_for({"x-ratelimit-remaining-minute": "0"}, NOW) == 60.0
    assert exhausted_for({"x-ratelimit-remaining": "3", "x-ratelimit-reset": "10"}, NOW) is None


def test_model_identity_ignores_vendor_prefix_and_free_suffix_only():
    assert same_model("deepseek/deepseek-v4-flash:free", "deepseek-v4-flash")
    assert same_model("Meta-Llama-3_3-70B", "meta-llama-3.3-70b")
    assert not same_model("llama-3.3-70b", "minimax-m2.5-free")
    assert not same_model("llama-3.3-70b", None)

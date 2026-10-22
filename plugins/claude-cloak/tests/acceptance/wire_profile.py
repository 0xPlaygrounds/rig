"""Allowlisted application-wire structure and a verifier for original request bytes."""
import hashlib
import json
import re
from audit import markers, fallback_record

HEADER_PATTERNS = {
    "user-agent": r"claude-cli/\d+\.\d+\.\d+ \(external, (?:cli|claude-desktop)\)",
    "anthropic-version": r"\d{4}-\d{2}-\d{2}",
    "x-stainless-package-version": r"\d+\.\d+\.\d+", "x-stainless-runtime-version": r"v\d+\.\d+\.\d+",
    "x-stainless-lang": r"js", "x-stainless-runtime": r"node|bun", "x-stainless-retry-count": r"\d{1,3}",
    "x-stainless-timeout": r"\d{1,7}", "x-stainless-os": r"Linux|MacOS|Windows|FreeBSD",
    "x-stainless-arch": r"x64|arm64|x86", "content-type": r"application/json(?:; charset=utf-8)?",
    "content-encoding": r"gzip|br|deflate|zstd|identity", "accept": r"application/json|text/event-stream",
    "accept-encoding": r"(?:gzip|deflate|br|zstd|identity)(?:, ?(?:gzip|deflate|br|zstd|identity))*",
    "x-app": r"cli", "anthropic-dangerous-direct-browser-access": r"true|false",
}
BODY_KEYS = {"model", "messages", "system", "tools", "metadata", "max_tokens", "thinking", "context_management", "compaction", "output_config", "fallbacks", "anthropic_beta", "stream", "temperature", "top_p", "top_k", "stop_sequences", "speed", "tool_choice", "cache_control", "container", "diagnostics"}
BLOCK_TYPES = {"text", "image", "thinking", "redacted_thinking", "tool_use", "tool_result", "document", "server_tool_use", "web_search_tool_result", "tool_addition", "tool_removal"}
CORRELATION_HEADERS = (
    "x-claude-code-session-id", "x-claude-code-request-class", "x-claude-code-prompt-id",
    "x-claude-code-agent-id", "x-claude-code-parent-agent-id", "x-claude-code-agent-type",
    "x-claude-code-prev-tool-durations", "x-client-request-id", "x-cc-context-compacted",
    "x-cc-compaction-request", "x-claude-code-compaction", "x-claude-code-context-compacted",
    "x-claude-code-signature", "x-anthropic-additional-protection",
    "x-claude-code-wiggly-dove",
    "x-claude-remote-session-origin", "x-client-app", "x-claude-remote-container-id", "x-claude-remote-session-id",
)
HEADER_ROLES = {
    "x-claude-code-request-class": ("main", "subagent", "workflow", "compaction", "auxiliary", "side_reply"),
    "x-claude-code-agent-type": ("teammate", "custom"),
}
STANDARD_HEADERS = {
    "authorization", "proxy-authorization", "cookie", "set-cookie", "x-api-key",
    "host", "content-length", "transfer-encoding", "connection", "keep-alive", "proxy-connection",
    "accept-language", "cache-control", "pragma", "origin", "referer", "expect", "te", "trailer",
    "upgrade", "date", "via", "forwarded", "range", "if-match", "if-none-match", "if-modified-since",
    "if-unmodified-since", "if-range", "content-language", "content-location", "content-range",
}


def correlation_headers(body, headers, evidence):
    normalized = {name.lower(): value for name, value in headers.items() if isinstance(name, str)}
    safe = {}
    for name in CORRELATION_HEADERS:
        value = normalized.get(name)
        item = {"present": value is not None}
        if isinstance(value, str):
            item["alias"] = evidence.alias(value)
            if name in HEADER_ROLES:
                item["role"] = value if value in HEADER_ROLES[name] else "other"
        safe[name] = item
    known = set(HEADER_PATTERNS) | set(CORRELATION_HEADERS) | STANDARD_HEADERS | {"anthropic-beta"}
    result = {"correlation_headers": safe, "extra_header_count": sum(name not in known for name in normalized)}
    session = normalized.get("x-claude-code-session-id")
    if session is not None:
        metadata = body.get("metadata")
        try:
            identity = json.loads(metadata.get("user_id", "")) if isinstance(metadata, dict) else None
        except (ValueError, TypeError):
            identity = None
        result["session_header_matches_metadata"] = isinstance(session, str) and isinstance(identity, dict) and isinstance(identity.get("session_id"), str) and session == identity["session_id"]
    return result


def request_role(messages):
    last = next((message for message in reversed(messages) if isinstance(message, dict) and message.get("role") in ("user", "assistant")), {})
    return "tool_continuation" if any(isinstance(block, dict) and block.get("type") == "tool_result" for block in last.get("blocks", [])) else "prompt"


def shape(body, headers, evidence):
    safe_headers = {}
    for name, pattern in HEADER_PATTERNS.items():
        value = headers.get(name)
        if value is not None:
            safe_headers[name] = value if isinstance(value, str) and re.fullmatch(pattern, value) else "unexpected"
    betas = headers.get("anthropic-beta", "").split(",")
    safe_headers["anthropic-beta"] = sorted({b.strip() if re.fullmatch(r"[a-z][a-z0-9-]{1,80}-(?:\d{4}-\d{2}-\d{2}|\d{8})", b.strip()) else "unexpected" for b in betas if b.strip()})
    system = body.get("system", [])
    if isinstance(system, str):
        system = [{"type": "text", "text": system}]
    def block(block):
        if not isinstance(block, dict):
            return {"type": "other"}
        kind = block.get("type")
        item = {"type": kind if kind in BLOCK_TYPES else "other"}
        if kind in ("text", "thinking"):
            text = block.get("text" if kind == "text" else "thinking", "")
            item["chars"] = len(text) if isinstance(text, str) else 0
        if kind == "thinking":
            item["signed"] = bool(block.get("signature"))
            item["response_only_estimate"] = "estimated_tokens" in block
        if kind == "tool_result":
            item["is_error_present"] = "is_error" in block
            item["is_error"] = block.get("is_error") is True
            content = block.get("content")
            item["content_kind"] = "string" if isinstance(content, str) else "array" if isinstance(content, list) else "other"
            item["content_alias"] = evidence.alias(json.dumps(content, sort_keys=True))
        return item
    messages = []
    for message in body.get("messages", []):
        content = message.get("content", [])
        messages.append({"role": message.get("role") if message.get("role") in ("user", "assistant", "system", "developer") else "other",
                         "content_kind": "string" if isinstance(content, str) else "array",
                         "blocks": [block(b) for b in content] if isinstance(content, list) else [{"type": "text", "chars": len(content)}] if isinstance(content, str) else []})
    tools = []
    for tool in body.get("tools", []):
        tools.append({"name": tool.get("name") if tool.get("name") in ("Read", "read", "_read", "DeferredToolPlaceholder") else "unexpected",
                      "schema_sha256": hashlib.sha256(json.dumps(tool.get("input_schema"), sort_keys=True).encode()).hexdigest(),
                      "strict": tool.get("strict") is True, "deferred": tool.get("defer_loading") is True})
    thinking = body.get("thinking", {})
    return {"headers": safe_headers, **correlation_headers(body, headers, evidence), **fallback_record(body), "body_keys": sorted(k if k in BODY_KEYS else "unexpected" for k in body),
            "call_role": request_role(messages),
            "system": [block(b) for b in system] if isinstance(system, list) else [], "messages": messages, "tools": tools,
            "thinking_type": thinking.get("type") if thinking.get("type") in ("adaptive", "enabled", "disabled", "between_tools") else "other",
            "thinking_binding": isinstance(thinking.get("block_binding"), dict),
            "thinking_display": thinking.get("display") if thinking.get("display") in ("updates", "summarized", "omitted") else "other",
            "thinking_budget": thinking.get("budget_tokens") if type(thinking.get("budget_tokens")) is int else None,
            "max_tokens": body.get("max_tokens") if type(body.get("max_tokens")) is int else None,
            "effort": body.get("output_config", {}).get("effort") if body.get("output_config", {}).get("effort") in ("low", "medium", "high", "max") else "other",
            "tool_choice": body.get("tool_choice", {}).get("type") if body.get("tool_choice", {}).get("type") in ("auto", "none", "any", "tool") else "other",
            "context_management": "context_management" in body, "compaction": "compaction" in body,
            "markers": markers(body), "diagnostics_previous_message_present": body.get("diagnostics", {}).get("previous_message_id") is not None,
            "metadata_user_id": isinstance(body.get("metadata", {}).get("user_id"), str)}


def checksum_record(raw, verifier=None):
    """The verifier receives original decoded bytes and returns only True/False/None.

    A separately reviewed module may provide verify(bytes). No checksum recipe is
    inferred from reserialized or redacted bodies. The default establishes no parity.
    """
    if not isinstance(raw, bytes):
        raise ValueError("checksum verification requires original bytes")
    result = verifier(raw) if verifier else None
    if result not in (True, False, None):
        raise ValueError("checksum verifier must return a boolean or None")
    return {"decoded_bytes": len(raw), "checksum": "match" if result is True else "mismatch" if result is False else "unknown"}

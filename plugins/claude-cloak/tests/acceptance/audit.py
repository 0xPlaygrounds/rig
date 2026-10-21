"""Private structural evidence for the optional live acceptance runner.

This is an observer, not a replacement for Rig's Anthropic decoder.
"""
import hashlib
import hmac
import json
import os
import re
import tempfile
from pathlib import Path
import zlib

MODEL = "claude-opus-5-5"
FACTS = {"harbor": "amber-lantern-731", "garden": "violet-otter-482",
         "observatory": "silver-pine-963", "archive": "copper-moon-254",
         "station": "indigo-crane-615"}
AUTO_FACT = "teal-comet-826"
AUTO_INPUT = "Newest retained message. Remember comet = " + AUTO_FACT + ".\n" + ("New neutral padding with no additional facts.\n" * 2400) + "\nReply only READY."
PHASE_NAMES = ("fresh_prefix", "fresh_prefix_repeat", "seed", "warm", "read_success", "read_error", "effort", "neutral", "manual_compact", "manual_recall", "rewarm", "fork", "resume", "auto_seed", "auto_trigger", "auto_recall", "auto_rewarm")


def reject_symlinks(path):
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError("symlink evidence paths are not supported")


def private_text(path, text):
    path = Path(path)
    reject_symlinks(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".acceptance-", delete=False) as out:
            temporary = Path(out.name)
            out.write(text)
        # Replacing a directory entry never follows a newly substituted symlink.
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def private_json(path, value):
    private_text(path, json.dumps(value, sort_keys=True, indent=2))


class Evidence:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        keyfile = self.root / ".pseudonym-key"
        try:
            fd = os.open(keyfile, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "wb") as out:
                out.write(os.urandom(32))
        self.key = keyfile.read_bytes()
        if len(self.key) != 32:
            raise ValueError("invalid pseudonym key")

    def alias(self, value):
        if value is None:
            return None
        return "private:" + hmac.new(self.key, str(value).encode(), hashlib.sha256).hexdigest()[:24]

    def save(self, record):
        # Callers must pass only the structural allowlisted records below.
        with os.fdopen(os.open(self.root / "audit.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "a") as out:
            out.write(json.dumps(record, sort_keys=True) + "\n")


def markers(body):
    """Only protocol marker slots count; schema and tool input keys are data."""
    found = []

    def add(block, path):
        if not isinstance(block, dict) or block.get("cache_control") is None:
            return
        control = block["cache_control"]
        control = control if isinstance(control, dict) else {}
        ttl = control.get("ttl", "5m")
        found.append({"path": path + "/cache_control",
                      "type_valid": control.get("type") == "ephemeral",
                      "ttl": ttl if ttl in ("5m", "1h") else "invalid"})

    for section in ("tools", "system"):
        blocks = body.get(section, [])
        if isinstance(blocks, list):
            for index, block in enumerate(blocks):
                add(block, f"/{section}/{index}")
    for index, message in enumerate(body.get("messages", [])):
        content = message.get("content", []) if isinstance(message, dict) else []
        if isinstance(content, list):
            for block_index, block in enumerate(content):
                add(block, f"/messages/{index}/content/{block_index}")
    add(body, "")
    return found


def usage(value):
    if not isinstance(value, dict):
        return {}
    result = {key: value[key] for key in ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
              if type(value.get(key)) is int and value[key] >= 0}
    creation = value.get("cache_creation")
    if isinstance(creation, dict):
        result["cache_creation"] = {key: creation[key] for key in ("ephemeral_5m_input_tokens", "ephemeral_1h_input_tokens")
                                    if type(creation.get(key)) is int and creation[key] >= 0}
    return result


def request_record(body, evidence):
    # Never persist headers, text, tools' schemas/inputs, signed blocks or raw JSON.
    identity = body.get("metadata", {}).get("user_id")
    try:
        identity = json.loads(identity) if isinstance(identity, str) else {}
    except ValueError:
        identity = {}
    if not isinstance(identity, dict):
        identity = {}
    controls = markers(body)
    billing = next((b.get("text", "") for b in body.get("system", []) if isinstance(b, dict) and b.get("text", "").startswith("x-anthropic-billing-header:")), "")
    fingerprint = re.search(r"cc_version=\d+\.\d+\.\d+\.([0-9a-f]{3})", billing)
    newest = next((m.get("content", "") for m in reversed(body.get("messages", [])) if m.get("role") == "user"), "")
    newest_text = newest if isinstance(newest, str) else "".join(b.get("text", "") for b in newest if isinstance(b, dict))
    results = [block for message in body.get("messages", []) for block in (message.get("content", []) if isinstance(message.get("content"), list) else []) if isinstance(block, dict) and block.get("type") == "tool_result"]
    return {"model_ok": body.get("model") == MODEL, "requested_model": MODEL if body.get("model") == MODEL else "unexpected", "markers": controls,
            "human_fingerprint": evidence.alias(fingerprint.group(1)) if fingerprint else None,
            "tools_read_only": all(t.get("name") in ("read", "_read") for t in body.get("tools", [])),
            "tool_results": len(results), "tool_errors": sum(bool(b.get("is_error")) for b in results),
            "input_chars": len(json.dumps(body.get("messages", []))),
            "newest_probe_intact": AUTO_INPUT in newest_text,
            "effort": body.get("output_config", {}).get("effort") if body.get("output_config", {}).get("effort") in ("low", "medium", "high", "max") else "unknown",
            "max_tokens": body.get("max_tokens") if type(body.get("max_tokens")) is int else None,
            "tool_count": len(body.get("tools", [])), "message_count": len(body.get("messages", [])),
            "session": evidence.alias(identity.get("session_id")),
            "device": evidence.alias(identity.get("device_id"))}


class SSE:
    """Bounded incremental SSE observer; unsupported encoding is an explicit error."""
    def __init__(self, encoding="identity", limit=2 * 1024 * 1024, second_read_file=None):
        if encoding not in ("identity", "gzip", "deflate"):
            raise ValueError("unsupported content encoding")
        self.decoder = None if encoding == "identity" else zlib.decompressobj(31 if encoding == "gzip" else 15)
        self.pending = b""
        self.limit = limit
        self.text = ""
        self.result = {"terminal": False, "message_stop": False, "model_ok": None, "usage": {}}
        self.message_id = None
        self.second_read_file = second_read_file

    def feed(self, chunk):
        data = self.decoder.decompress(chunk, self.limit + 1) if self.decoder else chunk
        if self.decoder and self.decoder.unconsumed_tail:
            raise ValueError("decompressed chunk exceeds bound")
        self.pending += data
        if len(self.pending) > self.limit:
            raise ValueError("SSE buffer exceeds bound")
        # Preserve a trailing CR until the next chunk arrives.
        normalized = self.pending.replace(b"\r\n", b"\n")
        while b"\n\n" in normalized:
            frame, normalized = normalized.split(b"\n\n", 1)
            payload = b"\n".join(line[5:].removeprefix(b" ") for line in frame.split(b"\n") if line.startswith(b"data:"))
            if payload:
                self.consume(json.loads(payload))
        self.pending = normalized

    def consume(self, event):
        kind = event.get("type")
        if kind == "message_start":
            message = event.get("message", {})
            self.message_id = message.get("id")
            self.result["model_ok"] = message.get("model") == MODEL
            self.result["served_model"] = MODEL if message.get("model") == MODEL else "unexpected"
            self.result["usage"].update(usage(message.get("usage")))
        elif kind == "message_delta":
            self.result["usage"].update(usage(event.get("usage")))
            reason = event.get("delta", {}).get("stop_reason")
            if reason is not None:
                self.result["terminal"] = True
                self.result["successful_stop"] = reason in ("end_turn", "tool_use", "stop_sequence")
                self.result["final_answer"] = reason in ("end_turn", "stop_sequence")
        elif kind == "message_stop":
            self.result["message_stop"] = True
        elif kind == "error":
            self.result["provider_error"] = True
        elif kind == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "text_delta":
                self.text += delta.get("text", "")
                if len(self.text) > self.limit:
                    raise ValueError("response text exceeds bound")

    def finish(self, cancelled=False):
        result = dict(self.result)
        result["facts"] = {key: value in self.text for key, value in FACTS.items()}
        result["auto_fact"] = AUTO_FACT in self.text
        if self.second_read_file:
            pattern = re.escape(self.second_read_file) + r"\s*\|\s*(?:failure|failed|error)\s*\|\s*1\b"
            result["second_read"] = {"filename": self.second_read_file in self.text,
                                     "failed_once": bool(re.search(pattern, self.text, re.IGNORECASE))}
        result["close"] = "cancel_after_terminal" if cancelled and result["terminal"] else "truncated" if not result["terminal"] else "complete"
        result["compression_complete"] = self.decoder is None or self.decoder.eof
        result["pending_frame"] = bool(self.pending.strip())
        return result


def effect_records(item):
    """Accept ordinary records and the session log's record envelope."""
    record = item.get("record", item)
    if not isinstance(record, dict):
        return []
    result = record.get("outcome", {}).get("Ok", {})
    if not isinstance(result, dict) or result.get("outcome") != "completion":
        return []
    return [(record, result)]


def import_effects(root, evidence):
    count = 0
    for path in Path(root).glob("*/files/effects.jsonl"):
        for line in path.read_text().splitlines():
            for record, result in effect_records(json.loads(line)):
                evidence.save({"kind": "effect", "request_id": evidence.alias(result.get("provider_request_id")),
                               "effect_id": evidence.alias(record.get("id")),
                               "purpose": record.get("purpose") if record.get("purpose") in ("summary", "compaction", "completion") else "other",
                               "usage": usage(result.get("raw", {}).get("usage", {}))})
                count += 1
    return count


def structural_record(record):
    """Treat imported JSON as untrusted, including values under familiar keys."""
    if not isinstance(record, dict):
        return {}
    safe = {}
    for key in ("call", "max_tokens", "tool_count", "message_count", "tool_results", "tool_errors", "input_chars", "status"):
        if type(record.get(key)) is int and 0 <= record[key] <= 2**63 - 1:
            safe[key] = record[key]
    for key in ("model_ok", "requested_model_ok", "tools_read_only", "newest_probe_intact", "terminal", "message_stop", "successful_stop", "final_answer", "provider_error", "auto_fact", "compression_complete", "pending_frame", "parse_error", "transport_error"):
        if type(record.get(key)) is bool:
            safe[key] = record[key]
    enums = {"kind": ("request", "headers", "response", "effect"), "scenario": (*PHASE_NAMES, "unknown"),
             "requested_model": (MODEL, "unexpected"), "served_model": (MODEL, "unexpected"),
             "effort": ("low", "medium", "high", "max", "unknown"),
             "purpose": ("summary", "compaction", "completion", "other"),
             "close": ("complete", "cancel_after_terminal", "truncated"),
             "local_guard": ("model", "tool_roster", "output_cap", "marker_budget", "invalid_request", "request_cap", "deadline", "credential_like_input"),
             "capture_unknown": ("unsupported_encoding",)}
    for key, choices in enums.items():
        if isinstance(record.get(key), str) and record[key] in choices:
            safe[key] = record[key]
    for key in ("request_id", "message_id", "effect_id", "session", "device", "human_fingerprint"):
        value = record.get(key)
        if isinstance(value, str) and re.fullmatch(r"private:[0-9a-f]{24}", value):
            safe[key] = value
    if isinstance(record.get("usage"), dict):
        safe["usage"] = usage(record["usage"])
    for key, fields in (("facts", FACTS), ("second_read", ("filename", "failed_once"))):
        if isinstance(record.get(key), dict):
            safe[key] = {field: record[key][field] for field in fields if type(record[key].get(field)) is bool}
    if isinstance(record.get("markers"), list):
        safe["markers"] = []
        for marker in record["markers"]:
            if not isinstance(marker, dict):
                continue
            item = {"type_valid": marker.get("type_valid") is True,
                    "ttl": marker.get("ttl") if marker.get("ttl") in ("5m", "1h") else "invalid"}
            path = marker.get("path")
            if isinstance(path, str) and re.fullmatch(r"(?:/(?:tools|system)/[0-9]+|/messages/[0-9]+/content/[0-9]+)?/cache_control", path):
                item["path"] = path
            safe["markers"].append(item)
    return safe


def analyze(records):
    records = [structural_record(record) for record in records]
    records = [record for record in records if record.get("kind") == "effect" or "call" in record]
    calls = {}
    effects = {r["request_id"]: r for r in records if r.get("kind") == "effect" and r.get("request_id")}
    for record in records:
        if record.get("kind") != "effect":
            calls.setdefault(record["call"], {}).update(record)
    rows = []
    for call, row in calls.items():
        controls = row.get("markers", [])
        ttl = [m["ttl"] for m in controls]
        valid = len(controls) <= 4 and all(m["type_valid"] and m["ttl"] in ("5m", "1h") for m in controls)
        valid = valid and not any(ttl[i] == "5m" and "1h" in ttl[i + 1:] for i in range(len(ttl)))
        status = "unknown"
        if row.get("local_guard"):
            status = "fail"
        elif row.get("status") is not None and row["status"] != 200:
            status = "fail"
        elif row.get("capture_unknown"):
            status = "unknown"
        elif row.get("close") == "complete" and (row.get("compression_complete") is False or row.get("pending_frame")):
            status = "fail"
        elif row.get("parse_error") or row.get("provider_error") or row.get("close") == "truncated":
            status = "fail"
        elif row.get("terminal") and row.get("successful_stop") and row.get("model_ok") and row.get("requested_model_ok") and row.get("status") == 200 and valid:
            status = "pass" if all(key in row.get("usage", {}) for key in ("cache_read_input_tokens", "cache_creation_input_tokens")) else "unknown"
        elif row.get("terminal"):
            status = "fail" if row.get("model_ok") is False or row.get("requested_model_ok") is False or row.get("successful_stop") is False or not valid else "unknown"
        effect = effects.get(row.get("request_id"))
        row.update(call=call, classification=status, markers_valid=valid,
                   effect_correlated=effect is not None,
                   effect_purpose=effect.get("purpose") if effect else None,
                   effect_usage=effect.get("usage") if effect else None,
                   effect_id=effect.get("effect_id") if effect else None)
        rows.append(row)
    return {"schema": 1, "calls": rows, "classification": "fail" if any(r["classification"] == "fail" for r in rows) else "pass" if rows and all(r["classification"] == "pass" for r in rows) else "unknown",
            "limits": ["Structural observer, not a checksum verifier or a provider invoice.", "Only executed phases are evidence; the small fresh-session control does not prove compaction.", "Missing evidence is unknown; successful HTTP alone is insufficient."]}


def report(root):
    root = Path(root)
    for name in ("audit.jsonl", "policy.json", "matrix.json", "fresh_prefix.json", "fresh_prefix_repeat.json", "summary.json", "REPORT.md"):
        reject_symlinks(root / name)
    path = root / "audit.jsonl"
    result = analyze([json.loads(line) for line in path.read_text().splitlines()] if path.exists() else [])
    if (root / "matrix.json").exists():
        imported = json.loads((root / "matrix.json").read_text())
        result["scenario_checks"] = {phase: imported[phase] if imported[phase] in ("pass", "fail", "unknown") else "unknown" for phase in PHASE_NAMES if isinstance(imported, dict) and phase in imported}
        statuses = list(result["scenario_checks"].values())
        result["classification"] = "fail" if "fail" in statuses or result["classification"] == "fail" else "unknown" if "unknown" in statuses else result["classification"]
    elif (root / "policy.json").exists():
        checks = {}
        for phase in ("fresh_prefix", "fresh_prefix_repeat"):
            calls = [row for row in result["calls"] if row.get("scenario") == phase]
            outcome_file = root / (phase + ".json")
            outcome = json.loads(outcome_file.read_text()) if outcome_file.exists() else {}
            checks[phase] = "unknown" if not calls or not outcome or any(row["classification"] == "unknown" for row in calls) else "pass" if (
                outcome.get("exit_code") == 0 and not outcome.get("timed_out")
                and all(row["classification"] == "pass" for row in calls)
                and all(calls[-1].get("facts", {}).get(key) for key in FACTS)
                and all(key in calls[-1].get("usage", {}) for key in ("cache_read_input_tokens", "cache_creation_input_tokens"))
                and (phase != "fresh_prefix_repeat" or calls[-1]["usage"]["cache_read_input_tokens"] > 0)
            ) else "fail"
        result["scenario_checks"] = checks
        result["classification"] = "fail" if "fail" in checks.values() or result["classification"] == "fail" else "unknown" if "unknown" in checks.values() else result["classification"]
    private_json(root / "summary.json", result)
    text = ["# Claude Cloak acceptance", "", "Overall: " + result["classification"], ""]
    text += [f"- Call {r['call']}: {r['classification']}; close={r.get('close', 'unknown')}; cache={r.get('usage', {})}" for r in result["calls"]]
    text += ["", "Scenarios: " + json.dumps(result.get("scenario_checks", {}))]
    text += ["", "Limits:"] + ["- " + limit for limit in result["limits"]]
    private_text(root / "REPORT.md", "\n".join(text) + "\n")
    return result

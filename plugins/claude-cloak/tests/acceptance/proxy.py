"""mitmdump addon. No flow dumps; only audit.py's structural summaries persist."""
import json
import os
import re
from pathlib import Path
import sys
import time
import zlib
import importlib.util

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit import Evidence, PHASE_NAMES, SSE, request_record
from ledger import Ledger
from wire_profile import shape, checksum_record
from mitmproxy import http


class Audit:
    def __init__(self):
        self.root = Path(os.environ["CLOAK_ACCEPTANCE_ROOT"])
        self.evidence = Evidence(self.root)
        self.policy = json.loads((self.root / "policy.json").read_text())
        self.ledger = Ledger(self.policy["ledger"]) if self.policy.get("ledger") else None
        self.verifier = None
        if self.policy.get("checksum_verifier"):
            spec = importlib.util.spec_from_file_location("acceptance_checksum", self.policy["checksum_verifier"])
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.verifier = module.verify
        path = self.root / "audit.jsonl"
        self.sequence = max((r.get("call", 0) for r in
                             (json.loads(line) for line in path.read_text().splitlines())), default=0) if path.exists() else 0

    def request(self, flow):
        # Authentication is handled normally by the built agent, not this addon.
        if flow.request.host != "api.anthropic.com" or flow.request.path.split("?", 1)[0] != "/v1/messages":
            return
        self.sequence += 1
        reason = None
        try:
            phase = (self.root / "phase.txt").read_text().strip()
            client = (self.root / "client.txt").read_text().strip()
            if phase not in PHASE_NAMES or client not in ("rig", "claude"):
                reason = "invalid_request"
        except (OSError, UnicodeError):
            phase, client, reason = "unknown", "rig", "invalid_request"
        client = client if client in ("rig", "claude") else "rig"
        state = {"call": self.sequence, "kind": "request", "client": client, "role": "primary", "scenario": phase if phase in PHASE_NAMES else "unknown"}
        flow.metadata["acceptance"] = state
        try:
            body = json.loads(flow.request.get_content())
            state.update(request_record(body, self.evidence))
            if client == "claude":
                state["tools_read_only"] = all(t.get("name") in ("Read", "DeferredToolPlaceholder") for t in body.get("tools", []))
            state["requested_model_ok"] = state.pop("model_ok")
            if client == "claude" and body.get("model") in self.policy.get("blocked_auxiliary_models", []):
                state["role"] = "auxiliary"
            elif not state["requested_model_ok"]:
                state["role"] = "unexpected"
            # Inspect decoded JSON, including tool results, but never auth headers.
            # Re-encoding catches JSON-escaped token characters without persisting them.
            if re.search(r"sk-ant-[A-Za-z0-9_-]{8,}", json.dumps(body, ensure_ascii=False)):
                reason = "credential_like_input"
            elif not state["requested_model_ok"]:
                reason = "auxiliary_model" if state["role"] == "auxiliary" else "model"
            elif not state["fallback_shape_valid"] or any(model != "claude-opus-5-5" for model in state["requested_fallback_models"]):
                reason = "fallback_model"
            elif not state["tools_read_only"] or (state["tool_count"] != 0 and (not (self.policy.get("matrix") or self.policy.get("tool_result_probe")) or phase in ("fresh_prefix", "fresh_prefix_repeat"))):
                reason = "tool_roster"
            elif not state["max_tokens"] or not 0 < state["max_tokens"] <= self.policy["output_tokens"]:
                reason = "output_cap"
            elif len(state["markers"]) > 4:
                reason = "marker_budget"
            raw = flow.request.get_content()
            profile = {"call": self.sequence, "client": client, "scenario": state["scenario"], "role": state["role"],
                       **shape(body, flow.request.headers, self.evidence), **checksum_record(raw, self.verifier)}
            with os.fdopen(os.open(self.root / "profiles.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600), "a") as output:
                output.write(json.dumps(profile, sort_keys=True) + "\n")
            raw_dir = os.environ.get("CLOAK_RAW_CAPTURE_DIR")
            if raw_dir:
                destination = Path(raw_dir).resolve()
                if Path("/dev/shm") not in destination.parents:
                    raise ValueError("raw capture must remain in private tmpfs")
                with os.fdopen(os.open(destination / f"{client}-{self.sequence}.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600), "wb") as output:
                    output.write(raw)
        except (ValueError, TypeError, AttributeError, OSError):
            reason = "invalid_request"
        if self.ledger:
            try:
                _, _, budget_reason = self.ledger.attempt(self.policy["run_id"], self.policy["requests"])
            except (ValueError, OSError):
                budget_reason = "aggregate_cap"
            if budget_reason:
                reason = budget_reason
        elif self.sequence > self.policy["requests"]:
            reason = "request_cap"
        if time.time() >= self.policy["deadline"]:
            reason = "deadline"
        if self.policy.get("startup_only"):
            reason = "startup_guard"
        if reason:
            state["local_guard"] = reason
            flow.response = http.Response.make(400, b'{"type":"error","error":{"type":"invalid_request_error","message":"local acceptance guard rejected request"}}', {"content-type": "application/json", "x-should-retry": "false"})
        # Explicitly negotiate supported compression rather than silently treating br/zstd as SSE.
        flow.request.headers["accept-encoding"] = "gzip, deflate, identity"
        self.evidence.save(state)

    def responseheaders(self, flow):
        state = flow.metadata.get("acceptance")
        if state is None:
            return
        state.update(status=flow.response.status_code,
                     request_id=self.evidence.alias(flow.response.headers.get("request-id") or flow.response.headers.get("x-request-id")))
        self.evidence.save({**state, "kind": "headers"})
        if "text/event-stream" not in flow.response.headers.get("content-type", ""):
            return
        try:
            target = "facts.txt" if state["scenario"] == "read_success" else self.policy.get("second_read_file") if state["scenario"] == "read_error" else None
            parser = SSE(flow.response.headers.get("content-encoding", "identity").lower(), second_read_file=self.policy.get("second_read_file"),
                         expected_read_target=target, read_tool=("Read",) if state["client"] == "claude" else ("read", "_read"),
                         workspace=os.environ.get("CLOAK_WORKSPACE_" + state["client"].upper()))
        except ValueError:
            state["capture_unknown"] = "unsupported_encoding"
            return
        flow.metadata["acceptance_parser"] = parser

        def stream(chunk):
            if chunk and not state.get("parse_error"):
                try:
                    parser.feed(chunk)
                except (ValueError, UnicodeError, TypeError, AttributeError, zlib.error):
                    state["parse_error"] = True
            return chunk
        flow.response.stream = stream

    def finish(self, flow, cancelled=False):
        state = flow.metadata.get("acceptance")
        if state is None:
            return
        parser = flow.metadata.get("acceptance_parser")
        if parser:
            state.update(parser.finish(cancelled))
            state["message_id"] = self.evidence.alias(parser.message_id)
        elif cancelled:
            state["close"] = "truncated"
        # Never persist the exception string: proxies and providers can echo secrets.
        self.evidence.save({**state, "kind": "response", "transport_error": cancelled})

    def response(self, flow):
        self.finish(flow)

    def error(self, flow):
        self.finish(flow, cancelled=True)


addons = [Audit()]

"""mitmdump addon. No flow dumps; only audit.py's structural summaries persist."""
import json
import os
import re
from pathlib import Path
import sys
import time
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit import Evidence, PHASE_NAMES, SSE, request_record
from mitmproxy import http


class Audit:
    def __init__(self):
        self.root = Path(os.environ["CLOAK_ACCEPTANCE_ROOT"])
        self.evidence = Evidence(self.root)
        self.policy = json.loads((self.root / "policy.json").read_text())
        path = self.root / "audit.jsonl"
        self.sequence = max((r.get("call", 0) for r in
                             (json.loads(line) for line in path.read_text().splitlines())), default=0) if path.exists() else 0

    def request(self, flow):
        # Authentication is handled normally by the built agent, not this addon.
        if flow.request.host != "api.anthropic.com" or flow.request.path.split("?", 1)[0] != "/v1/messages":
            return
        self.sequence += 1
        phase = (self.root / "phase.txt").read_text().strip()
        state = {"call": self.sequence, "kind": "request", "scenario": phase if phase in PHASE_NAMES else "unknown"}
        flow.metadata["acceptance"] = state
        reason = None
        try:
            body = json.loads(flow.request.get_content())
            state.update(request_record(body, self.evidence))
            state["requested_model_ok"] = state.pop("model_ok")
            # Inspect decoded JSON, including tool results, but never auth headers.
            # Re-encoding catches JSON-escaped token characters without persisting them.
            if re.search(r"sk-ant-[A-Za-z0-9_-]{8,}", json.dumps(body, ensure_ascii=False)):
                reason = "credential_like_input"
            elif not state["requested_model_ok"]:
                reason = "model"
            elif not state["tools_read_only"] or (state["tool_count"] != 0 and (not self.policy.get("matrix") or phase in ("fresh_prefix", "fresh_prefix_repeat"))):
                reason = "tool_roster"
            elif not state["max_tokens"] or not 0 < state["max_tokens"] <= self.policy["output_tokens"]:
                reason = "output_cap"
            elif len(state["markers"]) > 4:
                reason = "marker_budget"
        except (ValueError, TypeError, AttributeError):
            reason = "invalid_request"
        if self.sequence > self.policy["requests"]:
            reason = "request_cap"
        if time.time() >= self.policy["deadline"]:
            reason = "deadline"
        if reason:
            state["local_guard"] = reason
            flow.response = http.Response.make(429, b'{"error":{"type":"local_acceptance_guard","message":"bounded experiment rejected request"}}', {"content-type": "application/json"})
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
            parser = SSE(flow.response.headers.get("content-encoding", "identity").lower(), second_read_file=self.policy.get("second_read_file"))
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

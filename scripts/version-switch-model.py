"""Scripted stand-in for an Ollama server, for scripts/version-switch.sh.

Every chat answer is `FAKE-MODEL turns: <the user turns it was sent>`, so a
resumed session shows up as earlier turns in the request. A turn whose last
user message contains `[hold]` is answered only once HOLD_FILE exists, which
keeps that agent running mid-turn while another version uses the same home.
Each request is one JSON line in LOG. Standard library only.

usage: python3 -I scripts/version-switch-model.py PORT LOG HOLD_FILE
"""

import json
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT, LOG, HOLD = int(sys.argv[1]), sys.argv[2], sys.argv[3]
MODEL = "qwen3"


def log(entry):
    with open(LOG, "a", encoding="utf-8") as out:
        out.write(json.dumps(entry) + "\n")


def text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, body, content_type="application/json"):
        data = (json.dumps(body) + "\n").encode()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def missing(self):
        log({"unknown": self.command + " " + self.path})
        self.send_response(404)
        self.end_headers()

    def do_GET(self):
        if self.path.startswith("/api/tags"):
            return self.reply({"models": [{"name": MODEL, "model": MODEL}]})
        if self.path.startswith("/api/version"):
            return self.reply({"version": "0.9.0"})
        if self.path.startswith("/v1/models"):
            return self.reply({"object": "list", "data": [{"id": MODEL, "object": "model"}]})
        return self.missing()

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        request = json.loads(self.rfile.read(length) or b"{}")
        if self.path.startswith("/api/show"):
            return self.reply({
                "capabilities": ["completion", "tools"],
                "model_info": {"general.context_length": 131072},
            })
        if self.path.startswith("/v1/chat/completions"):
            return self.chat(request)
        return self.missing()

    def chat(self, request):
        turns = [text(m.get("content")) for m in request.get("messages", []) if m.get("role") == "user"]
        held = bool(turns) and "[hold]" in turns[-1]
        log({"chat": turns, "held": held})
        while held and not os.path.exists(HOLD):
            time.sleep(0.2)
        answer = "FAKE-MODEL turns: " + " | ".join(turns)
        usage = {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        base = {"id": "fake", "created": 0, "model": MODEL}
        if not request.get("stream"):
            message = {"role": "assistant", "content": answer}
            choice = {"index": 0, "message": message, "finish_reason": "stop"}
            return self.reply({**base, "object": "chat.completion", "choices": [choice], "usage": usage})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        chunks = (
            {"index": 0, "delta": {"role": "assistant", "content": answer}, "finish_reason": None},
            {"index": 0, "delta": {}, "finish_reason": "stop"},
        )
        for number, choice in enumerate(chunks):
            chunk = {**base, "object": "chat.completion.chunk", "choices": [choice]}
            if number == 1:
                chunk["usage"] = usage
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()
        self.close_connection = True


ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()

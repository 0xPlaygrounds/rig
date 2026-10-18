"""A small client of the rig agent server (rigd), standard library only.

    from rig_client import Client
    rig = Client()                                    # $RIG_HOME or ~/.rig
    s = rig.call("rig/session/start", cwd=".", prompt="say hi")["sessionId"]
    print(rig.wait(s)["answer"])
    for event in rig.events(s, after=0, follow=False):
        print(event["seq"], event["kind"])

The wire is JSON-RPC 2.0, one JSON object per line, every frame carrying
"rig": 1 (docs/protocol.md). Keyword arguments are snake_case and sent as
camelCase (session_id -> sessionId). Mutating methods get a fresh
idempotency key unless you pass key=..., and a call lost with its
connection is resent with the same key, so it is applied once. events()
resumes after a lost connection from the last seq it yielded and yields
each event once, in order.
"""

import json
import os
import socket
import subprocess
import time
import uuid

PROTOCOL = 1
MUTATING = {"rig/session/start", "rig/session/send", "rig/session/answer",
            "rig/session/stop", "rig/session/reload"}


class RigError(Exception):
    """An error answer: .code, .message, .data (with "next", the fix)."""

    def __init__(self, error):
        self.code = error.get("code")
        self.message = error.get("message", "")
        self.data = error.get("data") or {}
        super().__init__(f"{self.message} ({self.code})"
                         + (f"; next: {self.data['next']}" if "next" in self.data else ""))


def _camel(name):
    head, *rest = name.split("_")
    return head + "".join(word.title() for word in rest)


class Client:
    def __init__(self, path=None, name="rig_client.py", retry_for=30.0):
        """path: the rig home (default $RIG_HOME, else ~/.rig)."""
        self.home = path or os.environ.get("RIG_HOME") or os.path.expanduser("~/.rig")
        self.name, self.retry_for = name, retry_for
        self.next_id, self.sock = 0, None
        self.queued = {}  # subscription -> [event] read while waiting for a response
        self._connect()

    # The wire.

    def _connect(self):
        endpoint = os.path.join(self.home, "run", "rig.sock")
        deadline = time.monotonic() + self.retry_for
        started = False
        while True:
            try:
                self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                self.sock.connect(endpoint)
                break
            except OSError:
                self.sock.close()
                if not started:
                    # Any rig command starts the server when none runs.
                    subprocess.run(["rig", "agents", "--json"], capture_output=True,
                                   env={**os.environ, "RIG_HOME": self.home})
                    started = True
                if time.monotonic() > deadline:
                    raise RigError({"code": -1, "message": f"no rig server at {endpoint}",
                                    "data": {"next": "run `rig server status`"}})
                time.sleep(0.1)
        self.file = self.sock.makefile("rwb")
        self.queued = {}
        with open(os.path.join(self.home, "run", "token")) as f:
            token = f.read().strip()
        self.hello = self._request("rig/hello", {
            "token": token, "client": {"name": self.name, "version": "1"},
            "rig": {"min": PROTOCOL, "max": PROTOCOL}})

    def _send(self, frame):
        self.file.write(json.dumps({"jsonrpc": "2.0", "rig": PROTOCOL, **frame}).encode() + b"\n")
        self.file.flush()

    def _read(self):
        line = self.file.readline()
        if not line:
            raise ConnectionError("the rig server closed the connection")
        return json.loads(line)

    def _request(self, method, params):
        self.next_id += 1
        self._send({"id": self.next_id, "method": method, "params": params})
        while True:
            frame = self._read()
            if frame.get("method") == "rig/event":
                self.queued.setdefault(frame["params"].get("subscription"), []).append(frame["params"])
            elif frame.get("id") == self.next_id:
                if "error" in frame:
                    raise RigError(frame["error"])
                return frame["result"]

    def _event(self, subscription):
        """The next event of `subscription`, queueing other streams' events."""
        while True:
            queued = self.queued.get(subscription)
            if queued:
                return queued.pop(0)
            frame = self._read()
            if frame.get("method") == "rig/event":
                self.queued.setdefault(frame["params"].get("subscription"), []).append(frame["params"])

    def _reconnect(self):
        try:
            self.sock.close()
        except OSError:
            pass
        self._connect()

    # The API.

    def call(self, method, **params):
        """Calls `method`; returns its result or raises RigError."""
        params = {_camel(k): v for k, v in params.items() if v is not None}
        if method in MUTATING:
            params.setdefault("key", str(uuid.uuid4()))
        try:
            return self._request(method, params)
        except (ConnectionError, OSError):
            self._reconnect()
            return self._request(method, params)

    def wait(self, session, timeout_ms=None):
        """Waits for the session's latest command: {"status", "answer"?, "ask"?, ...}.
        status is answered, needs_approval, needs_answer, failed, stopped or timed_out."""
        return self.call("rig/session/wait", session_id=session, timeout_ms=timeout_ms)

    def events(self, session, after=0, follow=True):
        """The session's events after seq `after`, each once, in order; then live
        ones when `follow`. Without `follow` it ends at the journal's current end."""
        last = after
        while True:
            try:
                sub = self._request("rig/events/subscribe",
                                    {"sessionId": session, "afterSeq": last, "follow": follow})
                end = sub["nextSeq"] - 1
                while follow or last < end:
                    event = self._event(sub["subscription"])
                    if event["seq"] > last:
                        last = event["seq"]
                        yield event
                return
            except (ConnectionError, OSError):
                self._reconnect()

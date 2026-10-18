"""A small client of the rig agent server (rigd), standard library only.

    from rig_client import Client
    rig = Client()                                    # $RIG_HOME or ~/.rig
    s = rig.call("rig/session/start", cwd=".", prompt="say hi")["sessionId"]
    print(rig.wait(s)["answer"])
    for event in rig.events(s, after=0, follow=False):
        print(event["seq"], event["kind"])

The wire is JSON-RPC 2.0, one JSON object per line, every frame carrying
"rig": 1 (docs/protocol.md). Keyword arguments are snake_case and sent as
camelCase (session_id -> sessionId), and a relative `cwd` is made absolute
(the server refuses relative paths). Mutating methods get a fresh
idempotency key unless you pass key=..., and a call lost with its
connection is resent with the same key, so it is applied once; a call with
no key that changes something (a plugin's x/ method) is not resent: it
raises RigError saying it may have run. events()
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
# Calls that change nothing, so one lost with its connection is resent
# without a key. Any other call is resent only when it carries a key.
RESENDABLE = {"rig/hello", "rig/session/wait", "rig/session/list", "rig/session/read",
              "rig/events/subscribe", "rig/events/unsubscribe", "rig/events/read",
              "rig/session/attach", "rig/session/detach"}
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
    def __init__(self, path=None, name="rig_client.py", retry_for=30.0, token=None, autostart=True):
        """path: the rig home (default $RIG_HOME, else ~/.rig). token: the hello
        token (default: run/token). autostart: start the server when none runs."""
        self.home = path or os.environ.get("RIG_HOME") or os.path.expanduser("~/.rig")
        self.name, self.retry_for, self.token, self.autostart = name, retry_for, token, autostart
        self.next_id, self.sock, self.generation = 0, None, 0
        self.queued = {}  # subscription -> [event] read while waiting for a response
        self._connect()

    # The wire.

    def _connect(self):
        """Connects and says hello, retrying with backoff for `retry_for`
        seconds: a server that is starting may refuse or reset either step."""
        endpoint = os.path.join(self.home, "run", "rig.sock")
        deadline = time.monotonic() + self.retry_for
        started, pause = False, 0.02
        while True:
            try:
                self._open(endpoint)
                return
            except (ConnectionError, OSError) as error:
                if self.sock:
                    self.sock.close()
                if self.autostart and not started:
                    # Any rig command starts the server when none runs.
                    subprocess.run(["rig", "agents", "--json"], capture_output=True,
                                   env={**os.environ, "RIG_HOME": self.home})
                    started = True
                if time.monotonic() > deadline:
                    raise RigError({"code": -1, "message": f"no rig server at {endpoint} ({error})",
                                    "data": {"next": "run `rig server status`"}}) from None
                time.sleep(pause)
                pause = min(pause * 2, 1.0)

    def _open(self, endpoint):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(endpoint)
        self.file = self.sock.makefile("rwb")
        self.queued = {}
        self.generation += 1
        token = self.token
        if token is None:
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
        frame = json.loads(line)
        version = frame.get("rig")
        if version != PROTOCOL:
            # The older side updates: this client when the server is newer.
            newer = isinstance(version, int) and version > PROTOCOL
            raise RigError({"code": -32010, "message": f"the rig server speaks wire version {version}",
                            "data": {"update": "client" if newer else "server", "supported": [PROTOCOL],
                                     "next": "install a newer rig, or the newest clients/python/rig_client.py"
                                     if newer else "install a newer rig where the server runs"}})
        return frame

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
        if isinstance(params.get("cwd"), str):
            # The server takes absolute paths only: "." is this process's.
            params["cwd"] = os.path.abspath(params["cwd"])
        return self._call_wire(method, params)

    def _call_wire(self, method, params):
        """Calls `method` with params as they go on the wire."""
        deadline = time.monotonic() + self.retry_for
        while True:
            try:
                return self._request(method, params)
            except (ConnectionError, OSError):
                if time.monotonic() > deadline:
                    raise
                self._reconnect()
                if "key" not in params and method not in RESENDABLE:
                    raise RigError({"code": -1, "message": (
                        f"the connection to the rig server dropped before {method} answered, "
                        "so it may have run"),
                        "data": {"next": "check its effect before calling it again"}}) from None
                # Resent with the same key, so it is applied once.

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
                end, generation = sub["nextSeq"] - 1, self.generation
                while follow or last < end:
                    if self.generation != generation:
                        break  # a call reconnected: resubscribe after `last`
                    event = self._event(sub["subscription"])
                    if event["seq"] > last:
                        last = event["seq"]
                        yield event
                else:
                    return
            except (ConnectionError, OSError):
                self._reconnect()

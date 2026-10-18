"""rig_client.py against a fake server that drops connections the way a
restarting rigd does: python3 -m unittest clients/python/test_rig_client.py"""

import json
import os
import shutil
import socket
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rig_client import Client, RigError  # noqa: E402


class Fake:
    """Answers hello, rig/session/list and rig/events/subscribe (3 events).
    `drops` lists, per connection, what kills it: "hello" (closed on the
    hello, unanswered), "call" (closed on the first call, unanswered),
    "event:N" (closed after N events), or None."""

    def __init__(self, drops):
        self.home = tempfile.mkdtemp(prefix="rcpy-", dir=os.environ.get("TMPDIR"))
        os.makedirs(os.path.join(self.home, "run"), mode=0o700)
        with open(os.path.join(self.home, "run", "token"), "w") as f:
            f.write("t")
        self.drops, self.calls, self.connections = list(drops), [], 0
        self.listener = socket.socket(socket.AF_UNIX)
        self.listener.bind(os.path.join(self.home, "run", "rig.sock"))
        self.listener.listen()
        threading.Thread(target=self.accept, daemon=True).start()

    def accept(self):
        while True:
            try:
                conn, _ = self.listener.accept()
            except OSError:
                return
            drop = self.drops.pop(0) if self.drops else None
            self.connections += 1
            threading.Thread(target=self.serve, args=(conn, drop), daemon=True).start()

    def serve(self, conn, drop):
        with conn, conn.makefile("rwb") as f:
            def send(frame):
                f.write(json.dumps({"jsonrpc": "2.0", "rig": 1, **frame}).encode() + b"\n")
                f.flush()
            for line in f:
                request = json.loads(line)
                method = request["method"]
                if method == "rig/hello":
                    if drop == "hello":
                        return
                    send({"id": request["id"], "result": {"rig": 1, "clientId": "c"}})
                    continue
                self.calls.append(request)
                if drop == "call":
                    return
                if method == "rig/events/subscribe":
                    after = request["params"]["afterSeq"]
                    send({"id": request["id"], "result": {"subscription": 1, "nextSeq": 4}})
                    for sent, seq in enumerate(range(after + 1, 4)):
                        if drop == f"event:{sent}":
                            return
                        send({"method": "rig/event", "params": {
                            "subscription": 1, "session": "s", "seq": seq, "at_ms": 0,
                            "kind": "notice", "data": {}}})
                    continue
                send({"id": request["id"], "result": []})

    def close(self):
        self.listener.close()
        shutil.rmtree(self.home, ignore_errors=True)


class Reconnects(unittest.TestCase):
    def fake(self, drops):
        fake = Fake(drops)
        self.addCleanup(fake.close)
        return fake

    def client(self, fake):
        return Client(path=fake.home, autostart=False, retry_for=5)

    def test_a_reset_during_hello_is_retried(self):
        fake = self.fake(["hello", "hello"])
        self.assertEqual(self.client(fake).hello["clientId"], "c")
        self.assertEqual(fake.connections, 3)

    def test_a_call_lost_with_its_connection_is_resent_with_its_key(self):
        fake = self.fake(["call", "hello", None])
        rig = self.client(fake)
        self.assertEqual(rig.call("rig/session/send", session_id="s", text="hi"), [])
        sends = [c for c in fake.calls if c["method"] == "rig/session/send"]
        self.assertEqual(len(sends), 2)
        self.assertEqual(sends[0]["params"]["key"], sends[1]["params"]["key"])

    def test_a_call_without_a_key_lost_with_its_connection_is_not_resent(self):
        fake = self.fake(["call", None])
        rig = self.client(fake)
        with self.assertRaises(RigError) as raised:
            rig.call("x/test/bump", session_id="s")
        self.assertIn("may have run", str(raised.exception))
        self.assertEqual([c["method"] for c in fake.calls], ["x/test/bump"])
        self.assertEqual(rig.call("rig/session/list"), [])

    def test_a_stream_resumes_after_its_last_seq_through_a_failed_hello(self):
        # The first connection dies after one event, the next on its hello.
        fake = self.fake(["event:1", "hello", None])
        seqs = [event["seq"] for event in self.client(fake).events("s", after=0, follow=False)]
        self.assertEqual(seqs, [1, 2, 3])
        subscribes = [c["params"]["afterSeq"] for c in fake.calls if c["method"] == "rig/events/subscribe"]
        self.assertEqual(subscribes, [0, 1])

if __name__ == "__main__":
    unittest.main()

"""The conformance adapter for rig_client.py: tests/conformance/run python3 clients/python/conformance.py"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rig_client import Client, RigError  # noqa: E402

client, streams = None, {}


def handle(request):
    global client
    op = request["op"]
    if op == "connect":
        client = Client(token=request.get("token"), autostart=False, name="conformance.py")
        return client.hello
    if op == "call":
        return client._call_wire(request["method"], request.get("params", {}))
    if op == "subscribe":
        streams[request["name"]] = client.events(request["session"], request.get("after", 0),
                                                 request.get("follow", True))
        return {}
    if op == "next":
        out = []
        for event in streams[request["name"]]:
            event.pop("subscription", None)
            out.append(event)
            if event["kind"] == request.get("until") or len(out) == request.get("count"):
                break
        return out
    raise RigError({"code": -32601, "message": f"no op {op}"})


for line in sys.stdin:
    request = json.loads(line)
    try:
        answer = {"ok": handle(request)}
    except RigError as error:
        answer = {"error": {"code": error.code, "message": error.message, "data": error.data}}
    print(json.dumps(answer), flush=True)

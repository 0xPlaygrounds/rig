"""Drive rig from Python: start a session, wait for its answer, answer approvals.

    python3 drive.py "list the files here and say which is largest"
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rig_client import Client  # noqa: E402

rig = Client()
session = rig.call("rig/session/start", cwd=os.getcwd(), prompt=sys.argv[1])["sessionId"]
while True:
    result = rig.wait(session)
    if result["status"] == "needs_approval":
        ask = result["ask"]
        print(f"allowing: {ask['title']}", file=sys.stderr)
        allow = next(o for o in ask["options"] if o["kind"].startswith("allow"))
        rig.call("rig/session/answer", session_id=session, ask_id=ask["askId"], option_id=allow["optionId"])
        continue
    print(result.get("answer") or result)
    sys.exit(0 if result["status"] == "answered" else 1)

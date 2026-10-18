"""Drive rig from Python: start a session, answer its approvals and questions, print its answer.

    python3 drive.py "list the files here and say which is largest"
"""
import os, sys  # noqa: E401

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rig_client import Client  # noqa: E402

rig = Client()
session = rig.call("rig/session/start", cwd=".", prompt=sys.argv[1])["sessionId"]
while True:
    result = rig.wait(session)
    ask = result.get("ask")
    if result["status"] == "needs_approval":
        print(f"allowing: {ask['title']}", file=sys.stderr)
        allow = next(o for o in ask["options"] if o["kind"].startswith("allow"))
        rig.call("rig/session/answer", session_id=session, ask_id=ask["askId"], option_id=allow["optionId"])
    elif result["status"] == "needs_answer":
        reply = input(f"{ask['title']} {[o['optionId'] for o in ask['options']]}: ")
        choice = "option_id" if any(o["optionId"] == reply for o in ask["options"]) else "text"
        rig.call("rig/session/answer", session_id=session, ask_id=ask["askId"], **{choice: reply})
    else:
        print(result.get("answer") or f"{result['status']}: {result.get('error', '')}")
        sys.exit({"answered": 0, "timed_out": 124}.get(result["status"], 1))

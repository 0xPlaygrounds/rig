# rig acp speaks ACP v1 on stdio, as Zed drives it: initialize, a new
# session, a prompt that blocks until the turn ends, with the reply
# streamed as session/update chunks.
script '[{"reply": "PONG"}]'
python3 - "$RIG" "$RIG_CLI_WORK" <<'PY' || fail "the ACP exchange went wrong"
import json, subprocess, sys
rig, cwd = sys.argv[1], sys.argv[2]
p = subprocess.Popen([rig, "acp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
def send(frame):
    p.stdin.write(json.dumps(frame) + "\n"); p.stdin.flush()
def until(id):
    seen = []
    while True:
        line = p.stdout.readline()
        if not line:
            sys.exit(f"rig acp ended; saw {seen}")
        frame = json.loads(line)
        if frame.get("id") == id and "method" not in frame:
            return frame, seen
        seen.append(frame)
send({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": 1, "clientCapabilities": {}}})
init, _ = until(1)
assert init["result"]["protocolVersion"] == 1, init
send({"jsonrpc": "2.0", "id": 2, "method": "session/new", "params": {"cwd": cwd, "mcpServers": []}})
new, _ = until(2)
session = new["result"]["sessionId"]
send({"jsonrpc": "2.0", "id": 3, "method": "session/prompt", "params": {"sessionId": session, "prompt": [{"type": "text", "text": "say pong"}]}})
done, updates = until(3)
assert done["result"]["stopReason"] == "end_turn", done
text = "".join(u["params"]["update"]["content"]["text"] for u in updates
               if u.get("method") == "session/update"
               and u["params"]["update"]["sessionUpdate"] == "agent_message_chunk")
assert "PONG" in text, updates
p.stdin.close()
p.wait(timeout=10)
PY

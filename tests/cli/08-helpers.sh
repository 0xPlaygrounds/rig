# A parent starts two helpers, steers one, and waits on both: each report
# comes back once, as the report on the last message it was given; the
# tree shows both under the parent; stopping the parent stops them.
script '[
  {"when": "lead the work", "calls": [
    {"tool": "start_agent", "args": {"task": "helper alpha: say ALPHA", "name": "alpha"}},
    {"tool": "start_agent", "args": {"task": "helper beta: say BETA", "name": "beta"}}]},
  {"when": "lead the work", "calls": [{"tool": "wait_agent", "args": {"ids": ["alpha", "beta"]}}]},
  {"when": "lead the work", "calls": [{"tool": "wait_agent", "args": {"ids": ["alpha", "beta"]}}]},
  {"when": "lead the work", "calls": [{"tool": "wait_agent", "args": {"ids": ["alpha", "beta"]}}]},
  {"when": "lead the work", "reply": "both reported"},
  {"when": "helper alpha", "reply": "ALPHA report"},
  {"when": "helper beta", "reply": "BETA report"}
]'
expect 0 rig start --name lead "lead the work"
parent=$out
expect 0 rig wait "$parent" --timeout 120s
case "$out" in *"both reported"*) ;; *) fail "the parent's answer: $out" ;; esac
expect 0 rig log "$parent" --json
# What each wait_agent answered, from the parent's saved conversation.
python3 - "$out" <<'PY' || fail "the waits' answers: $out"
import json, sys
answers = []
for line in sys.argv[1].splitlines():
    r = json.loads(line)
    if r.get("kind") != "conversation.change":
        continue
    for entry in r["data"]["record"].get("entries", []):
        for part in entry.get("message", {}).get("content", []) or []:
            if isinstance(part, dict) and part.get("name") == "wait_agent":
                answers.append(" ".join(c.get("text", "") for c in part.get("content", [])))
assert len(answers) == 3, answers
reports = [a for a in answers if "answered:" in a]
assert sum("ALPHA report" in a.split("The others")[0] for a in reports) == 1, answers
assert sum("BETA report" in a.split("The others")[0] for a in reports) == 1, answers
assert answers[2].startswith("Nothing to wait for"), answers
PY
expect 0 rig agents --tree --json
python3 - "$out" "$parent" <<'PY' || fail "tree: $out"
import json, sys
rows = json.loads(sys.argv[1])
rows = rows if isinstance(rows, list) else rows.get("sessions", rows)
kids = [r for r in rows if r.get("parent") == sys.argv[2]]
assert len(kids) == 2, kids
PY
expect 0 rig stop "$parent"

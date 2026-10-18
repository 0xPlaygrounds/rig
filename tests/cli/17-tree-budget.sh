# A tree's token budget stops it: `--budget-tokens 1` is spent by the first
# model call, so rigd journals budget.exhausted, stops the running turns and
# refuses new messages.
script '[
  {"when": "lead the work", "tokens": 50, "calls": [{"tool": "start_agent", "args": {"task": "helper eps: work", "name": "eps"}}]},
  {"when": "lead the work", "tokens": 50, "calls": [{"tool": "wait_agent", "args": {"ids": ["eps"]}}]},
  {"when": "lead the work", "reply": "lead done"},
  {"when": "helper eps", "tokens": 50, "calls": [{"tool": "shell", "args": {"command": "sleep 30"}}]},
  {"when": "helper eps", "reply": "eps done"}
]'
expect 0 rig start --budget-tokens 1 --name lead "lead the work"
parent=$out
rig wait "$parent" --timeout 60s >/dev/null 2>&1
expect 0 rig log "$parent" --json
case "$out" in *'"budget.exhausted"'*) ;; *) fail "no budget.exhausted in the lead's journal" ;; esac
# New work in the spent tree is refused, saying what to do.
expect 6 rig send "$parent" "more work"
has_next
# The helper's running turn was stopped, not left to spend.
expect 0 rig agents --tree --json
python3 - "$out" "$parent" <<'PY' || fail "the helper still runs: $out"
import json, sys
rows = json.loads(sys.argv[1])
rows = rows if isinstance(rows, list) else rows.get("sessions", rows)
kids = [r for r in rows if r.get("parent") == sys.argv[2]]
assert kids and all(k["state"] != "running" for k in kids), kids
PY
rig stop "$parent" >/dev/null 2>&1
